"""Exercise the real subprocess and Phonopy workflow with a small ASE potential."""

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from ase.build import bulk
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication

from inspired.gui.mlff_worker import MLFFWorker
from inspired.mlff.backends import BACKENDS
from inspired.mlff.protocol import ARTIFACTS, calculation_options, read_result


@pytest.fixture(scope="module", autouse=True)
def app():
    """Keep a Qt application alive for controller and dialog construction."""
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def emt_runner(tmp_path):
    """Use the production calculation pipeline with a fast, deterministic calculator."""
    from inspired.mlff import runner

    path = tmp_path / "emt_runner.py"
    path.write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(Path(runner.__file__).parent)!r})\n"
        f"worker = runpy.run_path({runner.__file__!r})\n"
        "from ase.calculators.emt import EMT\n"
        "worker['run_job'].__globals__['create_calculator'] = lambda *args: EMT()\n"
        "worker['run_job'].__globals__['version'] = lambda name: 'test-EMT'\n"
        "sys.exit(worker['main']())\n",
        encoding="utf-8",
    )
    return path


def wait_for_job(worker, timeout=45000):
    """Collect a terminal signal while letting Qt service progress and timers."""
    loop = QEventLoop()
    outcome = []
    for signal, status in ((worker.completed, "completed"), (worker.failed, "failed"), (worker.cancelled, "cancelled")):
        signal.connect(lambda *args, state=status: (outcome.append((state, args)), loop.quit()))
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout)
    loop.exec_()
    if not outcome:
        worker.cancel()
        pytest.fail("The MLFF subprocess did not finish within the test timeout.")
    return outcome[0]


@pytest.mark.parametrize("relax_cell", [False, True])
def test_isolated_phonon_job_publishes_reloadable_results(tmp_path, monkeypatch, emt_runner, relax_cell):
    """Run a complete phonon job through QProcess and reload its portable artifacts."""
    monkeypatch.setenv("INSPIRED_MATTERSIM_PYTHON", sys.executable)
    worker = MLFFWorker(runner_path=emt_runner)
    messages = []
    worker.progress.connect(messages.append)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(10)
    worker.start(bulk("Cu", "fcc", a=3.6), "mattersim", tmp_path, lmin="1 1 1", relax_cell=relax_cell)
    status, arguments = wait_for_job(worker)
    timer.stop()
    assert status == "completed", arguments
    assert len(ticks) > 1
    assert "PROGRESS" in "".join(messages)
    result = read_result(worker.job_dir)
    assert result["supercell"] == [1, 1, 1]
    assert result["versions"]["torch"] == "test-EMT"
    assert worker.phonon.force_constants.shape[-2:] == (3, 3)
    for name in ARTIFACTS:
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "mlff-result.json").read_text())["status"] == "completed"
    assert not worker.running


def test_failed_job_preserves_previous_outputs(tmp_path, monkeypatch):
    """A worker crash must not replace outputs from an earlier calculation."""
    monkeypatch.setenv("INSPIRED_MATTERSIM_PYTHON", sys.executable)
    script = tmp_path / "fail.py"
    script.write_text("import sys\nprint('test failure', flush=True)\nsys.exit(7)\n")
    original = tmp_path / "FORCE_CONSTANTS"
    original.write_text("previous forces")
    worker = MLFFWorker(runner_path=script)
    worker.start(bulk("Cu"), "mattersim", tmp_path)
    status, arguments = wait_for_job(worker)
    assert status == "failed"
    assert "test failure" in arguments[0]
    assert original.read_text() == "previous forces"
    assert not worker.running


def test_cancellation_stops_subprocess_without_publishing(tmp_path, monkeypatch):
    """Cancel a running job and leave existing calculation files intact."""
    monkeypatch.setenv("INSPIRED_MATTERSIM_PYTHON", sys.executable)
    script = tmp_path / "wait.py"
    script.write_text("import time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    worker = MLFFWorker(runner_path=script)
    worker.progress.connect(lambda _message: worker.cancel())
    worker.start(bulk("Cu"), "mattersim", tmp_path)
    status, _ = wait_for_job(worker)
    assert status == "cancelled"
    assert not (tmp_path / "mlff-result.json").exists()
    assert not worker.running


@pytest.mark.parametrize("options", [{"lmin": "0"}, {"lmin": "1 2"}, {"lmin": "1 0 2"},
                                    {"delta": "nan"}, {"fmax": -1}, {"nmax": 0}])
def test_invalid_options_are_rejected(options):
    """Reject invalid calculation parameters before importing model libraries."""
    with pytest.raises(ValueError):
        calculation_options(**options)


def test_missing_environment_provides_setup_command(tmp_path, monkeypatch):
    """Explain exactly how to install a missing optional backend."""
    monkeypatch.setenv("INSPIRED_MATTERSIM_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(ValueError, match="pixi install --manifest-path"):
        MLFFWorker().start(bulk("Cu"), "mattersim", tmp_path)


def test_failed_to_start_releases_controller(tmp_path, monkeypatch):
    """A present executable with a missing interpreter must not strand the GUI."""
    executable = tmp_path / "broken-python"
    executable.write_text("#!/no/such/interpreter\n")
    executable.chmod(0o755)
    monkeypatch.setenv("INSPIRED_MATTERSIM_PYTHON", str(executable))
    worker = MLFFWorker()
    worker.start(bulk("Cu"), "mattersim", tmp_path)
    status, arguments = wait_for_job(worker)
    assert status == "failed"
    assert "Could not start" in arguments[0]
    assert not worker.running


def test_registry_import_does_not_load_frameworks():
    """Discovering model choices must work without any ML dependencies installed."""
    source_root = Path(__file__).parents[1] / "src"
    script = (
        f"import sys; sys.path.insert(0, {str(source_root)!r}); "
        "from inspired.mlff.backends import BACKENDS; "
        "assert len(BACKENDS) == 5; "
        "assert not any(name in sys.modules for name in ('torch', 'mattersim', 'sevenn', 'mace', 'orb_models'))"
    )
    subprocess.run([sys.executable, "-S", "-c", script], check=True)
    assert len({backend.environment for backend in BACKENDS}) == 4
