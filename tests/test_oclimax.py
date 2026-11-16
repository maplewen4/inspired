"""OCLIMAX parameter handling and failure reporting, without running the external binaries."""

import json
import os
import shutil
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication, QPushButton

from inspired.gui import oclimax_job
from inspired.gui.oclimax_job import JobError, OclimaxJob
from inspired.gui.oclimax_params import OclimaxParams, find_conf_line, set_mesh
from inspired.gui.set_paths import parse_path_config, write_path_config

MESH_CONF = """DIM = 2 0 0 0 2 0 0 0 3   # supercell

MP = 8 8 8
FC_SYMMETRY = .TRUE.
GAMMA_CENTER = .TRUE.
EIGENVECTORS = .TRUE.
FORCE_CONSTANTS = READ
"""


@pytest.fixture(scope="module", autouse=True)
def app():
    """Keep one Qt application alive while dialogs are constructed."""
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run in an empty directory, as INSPIRED does with its working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_supercell_and_mesh_are_read_without_shell_tools(in_tmp):
    """DIM and MP are parsed in Python, ignoring comments."""
    (in_tmp / "mesh.conf").write_text(MESH_CONF)
    params = OclimaxParams()
    params.get_supercell()
    params.get_default_mesh()
    assert params.supercell == [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert params.default_mesh == ["8", "8", "8"]


def test_three_value_dim_and_mesh_alias(in_tmp):
    """Three-value DIM and the MESH alias are accepted."""
    (in_tmp / "mesh.conf").write_text("DIM = 3 3 1\nMESH = 4 5 6\n")
    params = OclimaxParams()
    params.get_supercell()
    params.get_default_mesh()
    assert params.supercell == [3, 3, 1]
    assert params.default_mesh == ["4", "5", "6"]


def test_set_mesh_replaces_only_the_mesh_tag(in_tmp):
    """Only the mesh assignment changes."""
    path = in_tmp / "mesh.conf"
    path.write_text(MESH_CONF)
    set_mesh(path, ["2", "3", "4"])
    text = path.read_text()
    assert "MP = 2 3 4" in text
    # Lines that merely contain the letters "MP"/"MESH" elsewhere must be untouched.
    assert "GAMMA_CENTER = .TRUE." in text
    assert find_conf_line(path, ("DIM",)).startswith("DIM = 2 0 0")


def test_set_mesh_appends_when_missing(in_tmp):
    """A missing mesh is added rather than silently ignored."""
    path = in_tmp / "mesh.conf"
    path.write_text("DIM = 1 1 1\n")
    set_mesh(path, [5, 5, 5])
    assert find_conf_line(path, ("MP",)) == "MP = 5 5 5"


@pytest.mark.usefixtures("in_tmp")
def test_parameter_file_round_trip():
    """Written parameter files load back unchanged."""
    params = OclimaxParams()
    params.update_param("TASK", 2)
    params.update_param("Q1bin", "1.0  0.02  3.0")
    params.write_new_parameter_file("test.params")
    loaded = OclimaxParams()
    loaded.load_parameters("test.params")
    assert loaded.get_param("TASK") == 2
    assert loaded.get_param("Q1bin") == "1.0  0.02  3.0"


def test_path_config_tolerates_blank_lines_and_colons():
    """Blank lines, comments and Windows paths do not break startup."""
    text = "\npredictor_path: C:\\inspired\\model\n# comment\ndft_database_path:/data/dftdb\n"
    assert parse_path_config(text) == {
        "predictor_path": "C:\\inspired\\model",
        "dft_database_path": "/data/dftdb",
    }


def test_path_config_creates_missing_directory(tmp_path):
    """Saving preferences works on a fresh account."""
    config = tmp_path / "new" / "inspired" / "config"
    write_path_config(config, "/models", "/dftdb")
    assert parse_path_config(config.read_text()) == {"predictor_path": "/models", "dft_database_path": "/dftdb"}


def request(task=0, **params):
    """A job request as the dialog builds it."""
    values = dict(OclimaxParams().dict_params, TASK=task, **params)
    return {"params": values, "params_file": "oclimax.params", "bscale": "1.0"}


@pytest.fixture
def workdir(in_tmp):
    """The minimum input files that pass the pre-run checks."""
    for name in ("POSCAR-unitcell", "FORCE_CONSTANTS"):
        (in_tmp / name).write_text("")
    (in_tmp / "mesh.conf").write_text(MESH_CONF)
    OclimaxParams().write_new_parameter_file()
    return in_tmp


@pytest.fixture
def fake_phonopy(monkeypatch):
    """Skip the phonopy API steps so the OCLIMAX binary handling can be tested alone."""
    from inspired.gui import phonopy_steps

    monkeypatch.setattr(phonopy_steps, "mesh_phonon", lambda _conf: object())
    monkeypatch.setattr(phonopy_steps, "write_mesh_yaml", lambda *_: None)


@pytest.mark.usefixtures("workdir")
def test_failed_phonopy_step_is_reported(monkeypatch):
    """Invalid phonopy input stops the job with a message before any binary runs."""
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: pytest.fail("no process should start"))
    with pytest.raises(JobError, match="Phonopy calculation failed"):
        OclimaxJob(request()).run()  # the fixture's POSCAR-unitcell is empty


@pytest.mark.usefixtures("workdir", "fake_phonopy")
def test_failed_binary_is_reported(monkeypatch):
    """A failing OCLIMAX binary stops the job with a message, not an IndexError."""
    calls = []

    def failing_run(args, **kwargs):
        calls.append(args)
        assert kwargs.get("check") is True
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", failing_run)
    with pytest.raises(JobError, match="File format conversion failed"):
        OclimaxJob(request()).run()
    assert len(calls) == 1  # no later step runs after the converter fails


@pytest.mark.usefixtures("workdir", "fake_phonopy")
def test_missing_output_is_reported(monkeypatch):
    """A step that exits cleanly without output is reported."""
    monkeypatch.setattr(subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 0))
    with pytest.raises(JobError, match="did not produce an output file"):
        OclimaxJob(request()).run()


@pytest.mark.usefixtures("fake_phonopy")
def test_successful_job_reports_progress_and_absolute_result(workdir, monkeypatch, capsys):
    """The job prints machine-readable progress and the absolute result path for the GUI."""

    def fake_run(args, **_):
        if args[0].endswith("oclimax_run"):
            (workdir / (args[1].removesuffix(".oclimax") + "_2Dmesh_inc_0K.csv")).write_text("")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "setsid", lambda: None, raising=False)  # keep pytest in its own group
    assert oclimax_job.main(["--request", json.dumps(request())]) == 0
    output = capsys.readouterr().out.splitlines()
    assert any(line.startswith("PROGRESS 1/3 ") for line in output)
    result = [line for line in output if line.startswith("RESULT ")][0].split(" ", 1)[1]
    assert os.path.isabs(result) and result.endswith("_2Dmesh_inc_0K.csv")


@pytest.mark.usefixtures("workdir")
def test_invalid_input_is_rejected_before_starting(monkeypatch):
    """Typos and missing files are reported at once, without starting a process."""
    with pytest.raises(ValueError, match="must be numbers"):
        oclimax_job.validate(request(task=2, HKL="0 x 0"))
    os.remove("mesh.conf")
    with pytest.raises(ValueError, match="mesh.conf file in working directory is required"):
        oclimax_job.validate(request())

    from inspired.gui.oclimax import OCLIMAX

    warnings = []
    monkeypatch.setattr("inspired.gui.oclimax.QMessageBox.warning", lambda *args: warnings.append(args[2]))
    dialog = OCLIMAX()
    dialog.run_oclimax()
    assert not dialog.running
    assert "mesh.conf" in warnings[0]


# -- the real background process on the Cu example -----------------------------------------------

CU = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "Cu")


@pytest.fixture
def cu(in_tmp, monkeypatch):
    """The Cu DFT example in a working directory; a warning box fails the test instead of blocking it."""
    for name in ("FORCE_SETS", "POSCAR-unitcell", "mesh.conf", "scqe_1kk.params"):
        shutil.copy(os.path.join(CU, name), in_tmp)
    monkeypatch.setattr("inspired.gui.oclimax.QMessageBox.warning", lambda *args: pytest.fail(args[2]))
    return in_tmp


def prepared_dialog():
    """A dialog set up as 'Set INS parameters' leaves it."""
    from inspired.gui.oclimax import OCLIMAX

    dialog = OCLIMAX()
    dialog.oclimax_params.get_default_mesh()
    dialog.use_default_mesh()
    return dialog


def wait_for(dialog, *, on_progress=None, timeout=120000):
    """Run the event loop until the simulation finishes; return the status-bar message."""
    loop = QEventLoop()
    messages = []
    dialog.simulation_finished.connect(lambda message: (messages.append(message), loop.quit()))
    if on_progress:
        dialog.process.progress.connect(on_progress)
    QTimer.singleShot(timeout, loop.quit)
    loop.exec_()
    if not messages:
        dialog.process.cancel()
        pytest.fail("The INS simulation did not finish within the test timeout.")
    return messages[0]


@pytest.mark.usefixtures("cu")
def test_powder_simulation_runs_in_background_with_progress():
    """The dialog stays responsive: the run happens in a process that reports progress."""
    dialog = prepared_dialog()
    dialog.oclimax_params.update_param("MAXE", "40")  # a short spectrum keeps the test quick
    dialog.oclimax_params.write_new_parameter_file()
    progress = []
    dialog.run_oclimax()
    assert dialog.running and dialog.progress_dialog.isVisible()
    message = wait_for(dialog, on_progress=lambda done, total, text: progress.append((done, total, text)))
    assert message.startswith("INS simulation finished"), message
    assert os.path.isfile(dialog.csv_file_name) and os.path.isabs(dialog.csv_file_name)
    assert progress and progress[-1][1] == 3
    assert dialog.progress_dialog is None


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX-only")
@pytest.mark.usefixtures("cu")
def test_cancel_stops_the_job_and_its_binaries():
    """Cancel in the progress window ends the job and every OCLIMAX binary it started."""
    dialog = prepared_dialog()
    dialog.oclimax_params.load_parameters("scqe_1kk.params")
    dialog.params_file_name = "scqe_1kk.params"
    dialog.oclimax_params.update_param("Q2bin", "-1  0.002  2.5")  # many cycles, so the run is still going
    dialog.csv_file_name = previous = "previous-result.csv"
    dialog.run_oclimax()
    pid = dialog.process.process.processId()

    def cancel_during_cycles(_done, _total, text):
        if text.startswith("Cycle 2 ") and dialog.progress_dialog is not None:
            dialog.progress_dialog.findChild(QPushButton).click()  # what the user clicks

    assert wait_for(dialog, on_progress=cancel_during_cycles) == "INS simulation cancelled."
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)  # nothing is left in the job's process group
    assert dialog.csv_file_name == previous
