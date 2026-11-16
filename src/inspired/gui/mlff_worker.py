"""Qt controller for MLFF calculations in independent Python environments."""

import codecs
import os
import shutil
import tempfile
from pathlib import Path

import phonopy
from ase.io import write
from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from inspired.gui.oclimax import OCLIMAX
from inspired.mlff.backends import get_backend
from inspired.mlff.protocol import ARTIFACTS, PROTOCOL_VERSION, calculation_options, read_result, write_json
from inspired.phonon_plot import plot_band_structure_and_dos


class MLFFWorker(QObject):
    """Own one asynchronous job and publish successful calculation output."""

    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None, *, runner_path=None):
        super().__init__(parent)
        self.oclimax = OCLIMAX()
        self.nx = self.ny = self.nz = None
        self.phonon = None
        self.process = None
        self.job_dir = None
        self.workdir = None
        self._cancelling = False
        self._decoder = None
        self._log = None
        self._tail = ""
        self._runner_path = Path(runner_path) if runner_path else Path(__file__).parents[1] / "mlff" / "runner.py"

    @property
    def running(self):
        return self.process is not None

    def start(self, structure, backend_id, workdir, *, model=None, device="auto", **options):
        """Launch the selected environment's Python in a private job directory."""
        if self.running:
            raise RuntimeError("An MLFF calculation is already running.")
        backend = get_backend(backend_id)
        if not backend.available:
            raise ValueError(f"{backend.label} needs its own environment.\nRun:\n{backend.setup_hint}")
        validated = calculation_options(**options)
        self.workdir = Path(workdir).resolve()
        if model:
            model = str(Path(model).expanduser().resolve())
            if not Path(model).is_file():
                raise ValueError(f"Custom model file does not exist: {model}")
        if model and backend.id == "orb":
            raise ValueError("Custom checkpoints for ORB are not supported yet.")
        jobs_root = self.workdir / ".inspired-jobs"
        jobs_root.mkdir(exist_ok=True)
        self.job_dir = Path(tempfile.mkdtemp(prefix=f"{backend.id}-", dir=jobs_root))
        write(self.job_dir / "input.extxyz", structure, format="extxyz")
        if (self.workdir / "BORN").is_file():
            shutil.copy2(self.workdir / "BORN", self.job_dir / "BORN")
        write_json(self.job_dir / "request.json", {
            "protocol_version": PROTOCOL_VERSION, "backend": backend.id,
            "model": model, "device": device, "options": validated,
        })
        self.phonon = None
        self._cancelling = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._tail = ""
        self._log = (self.job_dir / "worker.log").open("w", encoding="utf-8")
        process = QProcess(self)
        process.setWorkingDirectory(str(self.job_dir))
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
            environment.remove(variable)
        environment.insert("PATH", str(backend.python.parent) + os.pathsep + os.environ.get("PATH", ""))
        environment.insert("OMP_NUM_THREADS", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        process.start(str(backend.python), ["-u", str(self._runner_path), "--request", str(self.job_dir / "request.json")])

    def cancel(self):
        """Terminate the worker, escalating to kill after a short grace period."""
        if not self.running:
            return
        self._cancelling = True
        process = self.process
        process.terminate()
        QTimer.singleShot(2000, lambda: self._kill_if_running(process))

    def _kill_if_running(self, process):
        if self.process is process and process.state() != QProcess.NotRunning:
            process.kill()

    def _read_output(self):
        if self.process is None:
            return
        message = self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        if message:
            self._log.write(message)
            self._log.flush()
            self._tail = (self._tail + message)[-4000:]
            self.progress.emit(message)

    def _process_error(self, error):
        if self.process is not None and error == QProcess.FailedToStart:
            message = self.process.errorString()
            self._cleanup()
            self.failed.emit(f"Could not start the model environment: {message}")

    def _cleanup(self):
        process, self.process = self.process, None
        if self._log:
            self._log.close()
            self._log = None
        if process:
            process.deleteLater()

    def _finished(self, exit_code, exit_status):
        self._read_output()
        self._cleanup()
        if self._cancelling:
            self.cancelled.emit()
            return
        if exit_status != QProcess.NormalExit or exit_code != 0:
            self.failed.emit(f"MLFF calculation failed. Log: {self.job_dir / 'worker.log'}\n{self._tail}")
            return
        try:
            result = read_result(self.job_dir)
            self.phonon = phonopy.load(str(self.job_dir / "phonopy.yaml"))
            self.nx, self.ny, self.nz = result["supercell"]
            for name in ARTIFACTS:
                staged = self.workdir / (name + ".inspired-tmp")
                shutil.copy2(self.job_dir / name, staged)
                staged.replace(self.workdir / name)
            (self.workdir / "FORCE_SETS").unlink(missing_ok=True)
            write_json(self.workdir / "mlff-result.json", result)
        except Exception as error:
            self.phonon = None
            self.failed.emit(f"Could not load MLFF results from {self.job_dir}: {error}")
            return
        self.completed.emit(result)

    def plot_results(self):
        """Plot the labelled dispersion and DOS exactly as the DFT workflow does."""
        if self.phonon is None:
            raise RuntimeError("Run the MLFF phonon calculation first.")
        plot_band_structure_and_dos(self.phonon).show()

    def generate_initial_mesh_file(self, mesh_list=(1, 1, 1)):
        """Write the standard mesh input for subsequent OCLIMAX calculations."""
        if not all((self.nx, self.ny, self.nz)):
            raise RuntimeError("Complete a phonon calculation first.")
        content = (
            f"DIM = {self.nx} 0 0 0 {self.ny} 0 0 0 {self.nz}\n\n"
            f"MP = {' '.join(map(str, mesh_list))}\n"
            "FC_SYMMETRY = .TRUE.\nGAMMA_CENTER = .TRUE.\nEIGENVECTORS = .TRUE.\nFORCE_CONSTANTS = READ\n"
        )
        (self.workdir / "mesh.conf").write_text(content, encoding="utf-8")
        self.oclimax.oclimax_params.get_default_mesh()
        self.oclimax.use_default_mesh()

    def generate_band_yaml(self):
        """Generate visualization data from the completed model without ML imports."""
        if self.phonon is None:
            raise RuntimeError("Run the MLFF phonon calculation first.")
        self.phonon.auto_band_structure(npoints=11, with_eigenvectors=True, write_yaml=True)
