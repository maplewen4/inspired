import codecs
import csv
import json
import os
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QTimer, Signal
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox, QProgressDialog

from inspired.gui import oclimax_job
from inspired.gui.oclimax_params import OclimaxParams
from inspired.gui.ui_help_oclimax import Ui_Help_OCLIMAX
from inspired.gui.ui_oclimax import Ui_OCLIMAX


class OclimaxProcess(QObject):
    """Run ``inspired.gui.oclimax_job`` asynchronously and translate its output into signals."""

    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None, *, python=sys.executable):
        super().__init__(parent)
        self.python = python
        self.process = None
        self._cancelling = False
        self._decoder = None
        self._buffer = ""
        self._errors = []
        self._tail = []
        self._result = None

    @property
    def running(self):
        return self.process is not None

    def start(self, request, workdir, tools_dir):
        if self.running:
            raise RuntimeError("An INS simulation is already running.")
        self._cancelling = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._errors, self._tail, self._result = [], [], None
        process = QProcess(self)
        process.setWorkingDirectory(str(workdir))
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        process.start(
            self.python,
            ["-u", "-m", "inspired.gui.oclimax_job", "--request", json.dumps(request), "--tools", str(tools_dir)],
        )

    def cancel(self):
        """Stop the job and the OCLIMAX binaries it started, escalating to kill after a grace period."""
        if not self.running:
            return
        self._cancelling = True
        process = self.process
        self._signal_group(process, signal.SIGTERM)
        QTimer.singleShot(2000, lambda: self.process is process and self._signal_group(process, signal.SIGKILL))

    @staticmethod
    def _signal_group(process, signum):
        # The job leads its own process group (os.setsid), which includes oclimax_run/oclimax_convert.
        pid = process.processId()
        if pid and hasattr(os, "killpg"):
            try:
                os.killpg(pid, signum)
                return
            except OSError:
                pass  # not (yet) a group leader, or already gone
        if process.state() != QProcess.NotRunning:
            process.kill() if signum == getattr(signal, "SIGKILL", None) else process.terminate()

    def _read_output(self):
        if self.process is None:
            return
        self._buffer += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._handle_line(line)

    def _handle_line(self, line):
        if line.startswith("PROGRESS "):
            counts, _, message = line[len("PROGRESS ") :].partition(" ")
            done, _, total = counts.partition("/")
            print(f"INFO: [{done}/{total}] {message}", flush=True)
            self.progress.emit(int(done), int(total), message)
            return
        if line.startswith("RESULT "):
            self._result = line[len("RESULT ") :].strip()
            return
        print(line, flush=True)
        if line.startswith("ERROR:"):
            self._errors.append(line[len("ERROR:") :].strip())
        self._tail = (self._tail + [line])[-20:]

    def _process_error(self, error):
        if self.process is not None and error == QProcess.FailedToStart:
            message = self.process.errorString()
            self._cleanup()
            self.failed.emit(f"Could not start the INS simulation: {message}")

    def _cleanup(self):
        process, self.process = self.process, None
        if process is not None:
            process.deleteLater()
        return process

    def _finished(self, exit_code, exit_status):
        self._read_output()
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        process = self._cleanup()
        if self._cancelling:
            # Make sure no binary outlives a cancelled job.
            self._signal_group(process, getattr(signal, "SIGKILL", signal.SIGTERM))
            self.cancelled.emit()
        elif exit_status == QProcess.NormalExit and exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit("\n".join(self._errors) or "\n".join(self._tail) or f"Exit code {exit_code}.")


class OCLIMAX(QDialog):
    simulation_finished = Signal(str)

    def __init__(self):
        super(OCLIMAX, self).__init__()
        self.ui = Ui_OCLIMAX()
        self.ui.setupUi(self)
        self.oclimax_params = OclimaxParams()
        self.params_file_name = "oclimax.params"
        self.csv_file_name = None
        self.oclimax_path = oclimax_job.TOOLS_DIR
        self.progress_dialog = None
        self._dialog_parent = None
        self.process = OclimaxProcess(self)
        self.process.progress.connect(self._job_progress)
        self.process.completed.connect(self._job_completed)
        self.process.failed.connect(self._job_failed)
        self.process.cancelled.connect(self._job_cancelled)

        self.enable_disable_single_crystal_parameters(False)

        self.display_parameters()

    def set_oclimax_cwd(self, path):
        os.chdir(path)

    def use_default_mesh(self):
        self.ui.lineEdit_bscale.setText("1.0")
        self.ui.spinBox_mesh1.setValue(int(self.oclimax_params.default_mesh[0]))
        self.ui.spinBox_mesh2.setValue(int(self.oclimax_params.default_mesh[1]))
        self.ui.spinBox_mesh3.setValue(int(self.oclimax_params.default_mesh[2]))
        self.oclimax_params.generate_mesh_file(mesh_list=self.oclimax_params.default_mesh)

    def save_mesh(self):
        mesh_list = []
        mesh_list.append(str(self.ui.spinBox_mesh1.value()))
        mesh_list.append(str(self.ui.spinBox_mesh2.value()))
        mesh_list.append(str(self.ui.spinBox_mesh3.value()))
        self.oclimax_params.generate_mesh_file(mesh_list=mesh_list)

    def load_parameters(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", "", "OCLIMAX parameter files (*.params)")
        if fname:
            self.oclimax_params.load_parameters(fname)
            self.display_parameters()
            self.params_file_name = fname

    def use_default_parameters(self):
        self.oclimax_params.reset_params_default()
        self.display_parameters()
        self.oclimax_params.write_new_parameter_file()
        self.params_file_name = "oclimax.params"

    def save_parameters(self):
        self.get_params_from_display()
        fname, _ = QFileDialog.getSaveFileName(self, "Save file", "", "OCLIMAX parameter files (*.params)")
        if fname:
            if not fname.endswith(".params"):
                fname += ".params"
            self.params_file_name = fname
            self.oclimax_params.write_new_parameter_file(fname)
        else:
            self.oclimax_params.write_new_parameter_file()

        self.close()

    def get_task_index(self):
        """Translate oclimax task/instr to task index in inspired"""
        task = self.oclimax_params.get_param("TASK")
        instrument = self.oclimax_params.get_param("INSTR")
        if task == 0 and instrument == 0:
            ind = 0
        elif task == 0 and instrument == 3:
            ind = 1
        elif task == 1 and instrument == 0:
            ind = 2
        elif task == 1 and instrument == 3:
            ind = 3
        elif task == 2:
            ind = 4
        elif task == 3:
            ind = 5
        else:
            ind = 6
        return ind

    def set_task_instr(self, ind):
        """Translate task index in inspired to oclimax task/instr"""
        if ind == 0:
            self.oclimax_params.update_param("TASK", 0)
            self.oclimax_params.update_param("INSTR", 0)
        elif ind == 1:
            self.oclimax_params.update_param("TASK", 0)
            self.oclimax_params.update_param("INSTR", 3)
        elif ind == 2:
            self.oclimax_params.update_param("TASK", 1)
            self.oclimax_params.update_param("INSTR", 0)
        elif ind == 3:
            self.oclimax_params.update_param("TASK", 1)
            self.oclimax_params.update_param("INSTR", 3)
        elif ind == 4:
            self.oclimax_params.update_param("TASK", 2)
            self.oclimax_params.update_param("INSTR", 3)
        elif ind == 5:
            self.oclimax_params.update_param("TASK", 3)
            self.oclimax_params.update_param("INSTR", 3)

    def display_parameters(self):
        self.ui.comboBox_task.setCurrentIndex(self.get_task_index())
        self.ui.lineEdit_output.setText(self.oclimax_params.get_param("OUTPUT"))
        self.ui.lineEdit_temp.setText(self.oclimax_params.get_param("TEMP"))
        self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param("MAXO"))
        self.ui.lineEdit_mask.setText(self.oclimax_params.get_param("MASK"))
        self.ui.lineEdit_theta.setText(self.oclimax_params.get_param("THETA"))
        self.ui.comboBox_Eunit.setCurrentIndex(self.oclimax_params.get_param("E_UNIT"))
        self.ui.lineEdit_norm.setText(self.oclimax_params.get_param("NORM"))
        self.ui.lineEdit_Ei.setText(self.oclimax_params.get_param("Ei"))
        self.ui.lineEdit_Ecut.setText(self.oclimax_params.get_param("ECUT"))
        self.ui.lineEdit_Emin.setText(self.oclimax_params.get_param("MINE"))
        self.ui.lineEdit_Emax.setText(self.oclimax_params.get_param("MAXE"))
        self.ui.lineEdit_Qmin.setText(self.oclimax_params.get_param("MINQ"))
        self.ui.lineEdit_Qmax.setText(self.oclimax_params.get_param("MAXQ"))
        self.ui.lineEdit_dE.setText(self.oclimax_params.get_param("dE"))
        self.ui.lineEdit_dQ.setText(self.oclimax_params.get_param("dQ"))
        self.ui.lineEdit_Eres.setText(self.oclimax_params.get_param("ERES"))
        self.ui.lineEdit_Qres.setText(self.oclimax_params.get_param("QRES"))
        self.ui.lineEdit_Q1.setText(self.oclimax_params.get_param("Q1"))
        self.ui.lineEdit_Q2.setText(self.oclimax_params.get_param("Q2"))
        self.ui.lineEdit_Q3.setText(self.oclimax_params.get_param("Q3"))
        self.ui.lineEdit_Q1bin.setText(self.oclimax_params.get_param("Q1bin"))
        self.ui.lineEdit_Q2bin.setText(self.oclimax_params.get_param("Q2bin"))
        self.ui.lineEdit_Q3bin.setText(self.oclimax_params.get_param("Q3bin"))
        self.ui.lineEdit_HKL.setText(self.oclimax_params.get_param("HKL"))
        self.ui.lineEdit_Ebin.setText(self.oclimax_params.get_param("Ebin"))
        self.ui.comboBox_xaxis.setCurrentIndex(self.oclimax_params.get_param("x-axis"))
        self.ui.comboBox_yaxis.setCurrentIndex(self.oclimax_params.get_param("y-axis"))

    def get_params_from_display(self):
        self.set_task_instr(self.ui.comboBox_task.currentIndex())
        self.oclimax_params.update_param("OUTPUT", self.ui.lineEdit_output.displayText())
        self.oclimax_params.update_param("TEMP", self.ui.lineEdit_temp.displayText())
        self.oclimax_params.update_param("MAXO", self.ui.lineEdit_maxo.displayText())
        self.oclimax_params.update_param("MASK", self.ui.lineEdit_mask.displayText())
        self.oclimax_params.update_param("THETA", self.ui.lineEdit_theta.displayText())
        self.oclimax_params.update_param("E_UNIT", self.ui.comboBox_Eunit.currentIndex())
        self.oclimax_params.update_param("NORM", self.ui.lineEdit_norm.displayText())
        self.oclimax_params.update_param("Ei", self.ui.lineEdit_Ei.displayText())
        self.oclimax_params.update_param("ECUT", self.ui.lineEdit_Ecut.displayText())
        self.oclimax_params.update_param("MINE", self.ui.lineEdit_Emin.displayText())
        self.oclimax_params.update_param("MAXE", self.ui.lineEdit_Emax.displayText())
        self.oclimax_params.update_param("MINQ", self.ui.lineEdit_Qmin.displayText())
        self.oclimax_params.update_param("MAXQ", self.ui.lineEdit_Qmax.displayText())
        self.oclimax_params.update_param("dE", self.ui.lineEdit_dE.displayText())
        self.oclimax_params.update_param("dQ", self.ui.lineEdit_dQ.displayText())
        self.oclimax_params.update_param("ERES", self.ui.lineEdit_Eres.displayText())
        self.oclimax_params.update_param("QRES", self.ui.lineEdit_Qres.displayText())
        self.oclimax_params.update_param("Q1", self.ui.lineEdit_Q1.displayText())
        self.oclimax_params.update_param("Q2", self.ui.lineEdit_Q2.displayText())
        self.oclimax_params.update_param("Q3", self.ui.lineEdit_Q3.displayText())
        self.oclimax_params.update_param("Q1bin", self.ui.lineEdit_Q1bin.displayText())
        self.oclimax_params.update_param("Q2bin", self.ui.lineEdit_Q2bin.displayText())
        self.oclimax_params.update_param("Q3bin", self.ui.lineEdit_Q3bin.displayText())
        self.oclimax_params.update_param("HKL", self.ui.lineEdit_HKL.displayText())
        self.oclimax_params.update_param("Ebin", self.ui.lineEdit_Ebin.displayText())
        self.oclimax_params.update_param("x-axis", self.ui.comboBox_xaxis.currentIndex())
        self.oclimax_params.update_param("y-axis", self.ui.comboBox_yaxis.currentIndex())

    def enable_disable_single_crystal_parameters(self, enable=False):
        self.ui.lineEdit_Q1.setEnabled(enable)
        self.ui.lineEdit_Q2.setEnabled(enable)
        self.ui.lineEdit_Q3.setEnabled(enable)
        self.ui.lineEdit_Q1bin.setEnabled(enable)
        self.ui.lineEdit_Q2bin.setEnabled(enable)
        self.ui.lineEdit_Q3bin.setEnabled(enable)
        self.ui.lineEdit_HKL.setEnabled(enable)
        self.ui.lineEdit_Ebin.setEnabled(enable)
        self.ui.comboBox_xaxis.setEnabled(enable)
        self.ui.comboBox_yaxis.setEnabled(enable)

        self.ui.lineEdit_Emin.setEnabled(not enable)
        self.ui.lineEdit_Emax.setEnabled(not enable)
        self.ui.lineEdit_dE.setEnabled(not enable)
        self.ui.lineEdit_Qmin.setEnabled(not enable)
        self.ui.lineEdit_Qmax.setEnabled(not enable)
        self.ui.lineEdit_dQ.setEnabled(not enable)
        self.ui.lineEdit_maxo.setEnabled(not enable)

    def task_changed(self):
        if self.ui.comboBox_task.currentIndex() == 4 or self.ui.comboBox_task.currentIndex() == 5:
            self.oclimax_params.update_param("MAXO", "1")
            self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param("MAXO"))
            self.enable_disable_single_crystal_parameters(True)
            if self.ui.comboBox_task.currentIndex() == 4:
                self.ui.comboBox_yaxis.setCurrentIndex(0)
            elif self.ui.comboBox_task.currentIndex() == 5:
                self.ui.comboBox_yaxis.setCurrentIndex(2)

        else:
            self.oclimax_params.update_param("MAXO", "10")
            self.ui.lineEdit_maxo.setText(self.oclimax_params.get_param("MAXO"))
            self.enable_disable_single_crystal_parameters(False)

    def unit_changed(self):  # No action for now, reserved for future development
        if self.ui.comboBox_Eunit.currentIndex() == 1:
            return

    @property
    def running(self):
        return self.process.running

    def run_oclimax(self, parent=None):
        """Validate the inputs, then run the simulation in a cancellable background process."""
        if self.running:
            return
        request = {
            "params": dict(self.oclimax_params.dict_params),
            "params_file": self.params_file_name,
            "bscale": self.ui.lineEdit_bscale.displayText(),
        }
        try:
            oclimax_job.validate(request)
        except ValueError as error:
            for line in str(error).splitlines():
                print(f"ERROR: {line}")
            QMessageBox.warning(parent, "INS simulation", str(error))
            return
        dialog = QProgressDialog("Starting INS simulation…", "Cancel", 0, 0, parent)
        dialog.setWindowTitle("INS simulation (OCLIMAX)")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self.process.cancel)
        self.progress_dialog = dialog
        self._dialog_parent = parent
        print("INFO: INS simulation started. Use Cancel in the progress window to stop it.")
        self.process.start(request, os.getcwd(), self.oclimax_path)
        dialog.show()

    def _job_progress(self, done, total, message):
        if self.progress_dialog is not None:
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setValue(done)
            self.progress_dialog.setLabelText(message)

    def _close_progress(self):
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None

    def _job_completed(self, result):
        self._close_progress()
        self.csv_file_name = result
        self.simulation_finished.emit(f"INS simulation finished: {result}")

    def _job_failed(self, message):
        self._close_progress()
        print(f"ERROR: INS simulation failed. {message}")
        self.simulation_finished.emit("INS simulation failed. See the terminal for details.")
        QMessageBox.warning(self._dialog_parent, "INS simulation failed", message)

    def _job_cancelled(self):
        self._close_progress()
        print("INFO: INS simulation cancelled. The previous result is kept for plotting.")
        self.simulation_finished.emit("INS simulation cancelled.")

    def plot_spec(
        self,
        unit=0,
        setrange=False,
        interactive=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
    ):
        if not self.csv_file_name:
            print("ERROR: No result to plot. Please run INS simulation first.")
            return

        try:
            with open(self.csv_file_name, "r") as csv_file:
                sim = list(csv.reader(csv_file))
        except OSError as error:
            print(f"ERROR: Could not read simulation result {self.csv_file_name}: {error}")
            return

        TASK = sim[2][0].split("#")[1]
        INSTR = sim[2][1]
        E_UNIT = sim[2][4]
        T = float(sim[2][2])

        if int(E_UNIT) == 0:
            csv_unit = r"cm$^{-1}$"
        elif int(E_UNIT) == 1:
            csv_unit = "meV"
        elif int(E_UNIT) == 2:
            csv_unit = "THz"
        if int(unit) == 1:
            plt_unit = r"cm$^{-1}$"
        elif int(unit) == 0:
            plt_unit = "meV"
        elif int(unit) == 2:
            plt_unit = "THz"

        if plt_unit == csv_unit:
            scalex = 1.0
        elif plt_unit == r"cm$^{-1}$" and csv_unit == "meV":
            scalex = 8.06554
        elif plt_unit == r"cm$^{-1}$" and csv_unit == "THz":
            scalex = 33.35641
        elif plt_unit == "meV" and csv_unit == r"cm$^{-1}$":
            scalex = 0.12399
        elif plt_unit == "meV" and csv_unit == "THz":
            scalex = 4.13567
        elif plt_unit == "THz" and csv_unit == r"cm$^{-1}$":
            scalex = 0.02998
        elif plt_unit == "THz" and csv_unit == "meV":
            scalex = 0.24180

        i = 0
        while i < len(sim):
            if not sim[i]:
                del sim[i]
                continue
            try:
                float(sim[i][0])
            except ValueError:
                del sim[i]
                continue
            i = i + 1

        for row in range(len(sim)):
            for col in range(len(sim[0])):
                try:
                    float(sim[row][col])
                except ValueError:
                    sim[row][col] = "0.0"
                    continue

        if int(INSTR) in [0, 1, 2]:
            fig, ax = plt.subplots(figsize=(6, 5))
            tsim = [[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
            tsim[0] = [tsim[0][i] * scalex for i in range(len(tsim[0]))]
            ns = len(sim[0]) - 1
            x_min = min(tsim[0])
            x_max = max(tsim[0])
            y_min = min(tsim[1])
            y_max = max(tsim[1])
            if setrange:
                try:
                    x_min = float(xmin)  # x_min = max([float(xmin),x_min])
                except:
                    pass
                try:
                    x_max = float(xmax)  # x_max = min([float(xmax),x_max])
                except:
                    pass
                try:
                    y_min = float(ymin)  # y_min = max([float(ymin),y_min])
                except:
                    pass
                try:
                    y_max = float(ymax)  # y_max = min([float(ymax),y_max])
                except:
                    pass
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            for i in range(ns):
                tsim[i + 1] = [tsim[i + 1][j] for j in range(len(tsim[i + 1]))]
            if int(INSTR) == 0:
                ax.plot(tsim[0], tsim[1], label="High Q total")
                ax.plot(tsim[0], tsim[2], label="Low Q total")
                ax.plot(tsim[0], tsim[3], label="High Q 0-1")
                ax.plot(tsim[0], tsim[4], label="High Q 0-2")
                ax.plot(tsim[0], tsim[5], label="High Q 0-3")
                ax.plot(tsim[0], tsim[6], label="High Q 0-4+")
            else:
                for i in range(ns):
                    ax.plot(tsim[0], tsim[i + 1], label="Spec" + str(i + 1))
            ax.set_xlabel("Energy transfer (" + plt_unit + ")")
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend(frameon=False)
            fig.show()

        elif int(INSTR) == 3:
            tsim = [[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
            x = [tsim[0][i] for i in range(len(tsim[0]))]
            if int(TASK) == 3:
                y = [tsim[1][i] for i in range(len(tsim[1]))]
            else:
                y = [tsim[1][i] * scalex for i in range(len(tsim[1]))]
            z = [tsim[2][i] for i in range(len(tsim[2]))]
            xi = np.sort(np.asarray(list(set(x))))
            yi = np.sort(np.asarray(list(set(y))))
            zi = np.zeros((len(yi), len(xi)))
            for i in range(len(yi)):
                for j in range(len(xi)):
                    k = i + j * len(yi)
                    if z[k] >= 0.0:
                        zi[i][j] = z[k]
                    elif z[k] <= -1:
                        zi[i][j] = np.nan
                    else:
                        zi[i][j] = 0.0
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
            z_min = 0
            z_max = max(z)
            if setrange:
                try:
                    x_min = max([float(xmin), x_min])
                except:
                    pass
                try:
                    x_max = min([float(xmax), x_max])
                except:
                    pass
                try:
                    y_min = max([float(ymin), y_min])
                except:
                    pass
                try:
                    y_max = min([float(ymax), y_max])
                except:
                    pass
                try:
                    z_min = float(zmin)  # z_min = max([float(zmin),z_min])
                except:
                    pass
                try:
                    z_max = float(zmax)  # z_max = min([float(zmax),z_max])
                except:
                    pass
            x1 = np.where(np.logical_and(xi >= x_min, xi <= x_max))[0][0]
            x2 = np.where(np.logical_and(xi >= x_min, xi <= x_max))[0][-1]
            y1 = np.where(np.logical_and(yi >= y_min, yi <= y_max))[0][0]
            y2 = np.where(np.logical_and(yi >= y_min, yi <= y_max))[0][-1]
            Z = zi[y1 : y2 + 1, x1 : x2 + 1]
            asp = (x_max - x_min) / (y_max - y_min) * 1.0

            if interactive:
                fig, axs = plt.subplots(
                    2,
                    2,
                    figsize=(6, 6),
                    sharex="col",
                    sharey="row",
                    gridspec_kw=dict(height_ratios=[1, 3], width_ratios=[3, 1]),
                )
                axs[0, 1].set_visible(False)
                axs[0, 0].set_box_aspect(1 / 3)
                axs[1, 0].set_box_aspect(1)
                axs[1, 1].set_box_aspect(3 / 1)

                axs[0, 0].set_ylabel("Z profile")
                axs[1, 1].set_xlabel("Z profile")
                axs[1, 1].set_xlim(right=z_max)
                axs[0, 0].set_ylim(top=z_max)

                axs[1, 0].imshow(
                    Z,
                    origin="lower",
                    extent=[x_min, x_max, y_min, y_max],
                    vmin=z_min,
                    vmax=z_max,
                    interpolation="gaussian",
                    cmap="jet",
                )
                axs[1, 0].set_aspect(aspect=asp, share=False)

                v_line = axs[1, 0].axvline(np.nan, color="r")
                h_line = axs[1, 0].axhline(np.nan, color="g")
                (v_prof,) = axs[1, 1].plot(
                    np.zeros(len(Z[:, 0])),
                    np.linspace(y_min, y_max, len(Z[:, 0])),
                    "r-",
                )
                (h_prof,) = axs[0, 0].plot(
                    np.linspace(x_min, x_max, len(Z[0, :])),
                    np.zeros(len(Z[0, :])),
                    "g-",
                )

                dx = xi[1] - xi[0]
                dy = yi[1] - yi[0]

                def on_move(event):
                    if event.inaxes is axs[1, 0]:
                        cur_x = event.xdata
                        cur_y = event.ydata

                        v_line.set_xdata([cur_x, cur_x])
                        h_line.set_ydata([cur_y, cur_y])
                        # Clamp so the cursor at the plot edge cannot index past the data.
                        col = min(max(int((cur_x - x_min) / dx), 0), Z.shape[1] - 1)
                        row = min(max(int((cur_y - y_min) / dy), 0), Z.shape[0] - 1)
                        v_prof.set_xdata(Z[:, col])
                        h_prof.set_ydata(Z[row, :])

                        fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", on_move)
                ax = axs[1, 0]

            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                pos = ax.imshow(
                    Z,
                    origin="lower",
                    extent=[x_min, x_max, y_min, y_max],
                    vmin=z_min,
                    vmax=z_max,
                    aspect=asp,
                    interpolation="gaussian",
                    cmap="jet",
                )
                fig.colorbar(pos, ax=ax)

            if int(TASK) == 2:
                ax.set_xlabel(r"Q (r.l.u.)")
                ax.set_ylabel("E (" + plt_unit + ")")
            elif int(TASK) == 3:
                ax.set_xlabel(r"Qx (r.l.u.)")
                ax.set_ylabel(r"Qy (r.l.u.)")
            else:
                ax.set_xlabel(r"|Q| (${\rm \AA}^{-1}$)")
                ax.set_ylabel("E (" + plt_unit + ")")
            fig.show()

    def open_help_oclimax(self):
        help_oclimax = QDialog()
        self.setup_help = Ui_Help_OCLIMAX()
        self.setup_help.setupUi(help_oclimax)
        help_oclimax.exec()
