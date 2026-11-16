import warnings

warnings.simplefilter("ignore")
import os
import shutil
import sys
import traceback

from ase.formula import Formula
from ase.io import read
from ase.spacegroup import get_spacegroup
from ase.visualize import view
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
)

from inspired.gui.dft_worker import DFTWorker
from inspired.gui.dp_worker import DPWorker
from inspired.gui.mlff_worker import MLFFWorker
from inspired.gui.pandas_model import PandasModel
from inspired.gui.set_paths import SetPaths, parse_path_config, user_config_path
from inspired.gui.ui_help_dft import Ui_Help_DFT
from inspired.gui.ui_help_dp import Ui_Help_DP
from inspired.gui.ui_help_mlff import Ui_Help_MLFF
from inspired.gui.ui_inspired import Ui_INSPIRED
from inspired.mlff.backends import BACKENDS
from inspired.version import __version__
from inspired.visualization import PhononVisualizer, VisualizationError, VisualizationTarget


class INSPIRED(QMainWindow):
    def __init__(self, parent=None):
        super(INSPIRED, self).__init__(parent)
        self.ui = Ui_INSPIRED()
        self.ui.setupUi(self)
        self.dp_worker = DPWorker()
        self.dft_worker = DFTWorker()
        self.mlff_worker = MLFFWorker(self)
        self.mlff_worker.progress.connect(self._mlff_progress)
        self.mlff_worker.completed.connect(self._mlff_completed)
        self.mlff_worker.failed.connect(self._mlff_failed)
        self.mlff_worker.cancelled.connect(self._mlff_cancelled)
        self.mlff_progress_dialog = None
        for oclimax in (self.dft_worker.oclimax, self.mlff_worker.oclimax):
            oclimax.simulation_finished.connect(lambda message: self.ui.statusbar.showMessage(message, 10000))
        self.set_paths = SetPaths()
        self.root_path = os.path.dirname(__file__)
        self.cwd_path = os.getcwd()
        self.ui.label_cwd.setText("Current Working Directory: " + self.cwd_path)
        self.userdft_path = self.cwd_path
        self.setup_paths()
        self.check_paths()
        self.show_df = self.dft_worker.get_initial_search_df(dft_path=self.dft_database_path)
        self.ui.tableView.setModel(PandasModel(self.show_df))
        self.pred_type = "Phonon DOS"
        self.atoms_dp = None
        self.atoms_dft = None
        self.atoms_mlff = None
        self.mlff_opt = False

        for i in range(3):
            self.ui.tableView.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.ui.tableView.verticalHeader().setVisible(False)
        self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.radioButton_database.setChecked(True)

        self.ui.lineEdit_lmin_mlff.setText("12")
        self.ui.lineEdit_fmax_mlff.setText("0.01")
        self.ui.lineEdit_nmax_mlff.setText("100")
        self.ui.lineEdit_delta_mlff.setText("0.03")

        self._set_visualization_enabled("dft", False)
        self._set_visualization_enabled("mlff", False)
        self.ui.comboBox_mlff_model.clear()
        for backend in BACKENDS:
            status = "Environment installed" if backend.available else "Setup required"
            self.ui.comboBox_mlff_model.addItem(f"{backend.label} — {status}", backend.id)
            index = self.ui.comboBox_mlff_model.count() - 1
            self.ui.comboBox_mlff_model.setItemData(index, backend.setup_hint, Qt.ToolTipRole)

    def setup_working_directory(self):
        """Set working directory
        This is the directory where all input and output files are located
        Default is the location where inspired is started
        """
        if self.mlff_worker.running:
            return
        wd = QFileDialog.getExistingDirectory(self, "Working directory", "")
        if wd and os.path.realpath(wd) != os.path.realpath(self.cwd_path):
            self.cwd_path = wd
            # Phonon files of loaded models live in the previous directory.
            self._invalidate_dft("the working directory changed")
            self._invalidate_mlff("the working directory changed")
        self.ui.statusbar.showMessage(f"Working directory is set to {self.cwd_path}.", 1000)
        self.ui.label_cwd.setText("Current Working Directory: " + self.cwd_path)
        os.chdir(self.cwd_path)

    def setup_paths(self):
        """Set paths to database and model files
        Default is root dir of inspired but can be customized
        Will be saved in ~/.config/inspired
        """
        user_config = user_config_path()
        sys_config = os.path.join(self.root_path, "config")
        self.predictor_path = os.path.join(self.root_path, "model")
        self.dft_database_path = os.path.join(self.root_path, "dftdb")
        if not user_config.is_file():
            if not os.path.isfile(sys_config):
                print("INFO: config file not found, using default paths.")
                return
            user_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sys_config, user_config)

        path_dict = parse_path_config(user_config.read_text(encoding="utf-8"))
        if path_dict.get("predictor_path", "default") != "default":
            self.predictor_path = path_dict["predictor_path"]
        if path_dict.get("dft_database_path", "default") != "default":
            self.dft_database_path = path_dict["dft_database_path"]

    def path_problems(self):
        """Return a description of every missing required model or database file."""
        problems = []
        if not os.path.isdir(self.predictor_path):
            problems.append(f"Predictor model path does not exist: {self.predictor_path}")
        else:
            for dp_file in (
                "crys_dos_predictor.torch",
                "crys_vis_predictor.torch",
                "latent_space_predictor.torch",
                "decoder.pt",
            ):
                if not os.path.isfile(os.path.join(self.predictor_path, dp_file)):
                    problems.append(f"Predictor model path is missing {dp_file}")
        if not os.path.isdir(self.dft_database_path):
            problems.append(f"DFT database path does not exist: {self.dft_database_path}")
        elif not os.path.isfile(os.path.join(self.dft_database_path, "crystals.dat")):
            problems.append(f"DFT database file crystals.dat was not found in {self.dft_database_path}")
        return problems

    def check_paths(self):
        """Check if paths are valid
        Inspired will not run without all paths set properly
        Asks for new paths until all required files are found
        Users can choose to quit
        """
        while problems := self.path_problems():
            for problem in problems:
                print(f"ERROR: {problem}. Please reset in config file or Preferences in the Menu.")
            QMessageBox.warning(
                None,
                "INSPIRED paths",
                "INSPIRED cannot find its model and database files:\n\n" + "\n".join(problems),
            )
            self.set_paths.init_paths(self.predictor_path, self.dft_database_path)
            if self.set_paths.exec() != QDialog.Accepted:
                print("ERROR: Required paths were not configured. Exiting.")
                sys.exit(1)
            self.setup_paths()

    def set_preferences(self):
        """Open preferences window
        Set and check paths
        """
        self.set_paths.init_paths(self.predictor_path, self.dft_database_path)
        if self.set_paths.exec() == QDialog.Accepted:
            self.setup_paths()
            self.check_paths()

    def select_structure_file(self):
        """Choose a structure file in browser
        Currently support cif and VASP POSCAR/CONTCAR
        """
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", "", "Structure files (*)")
        #  '',"CIF files (*.cif)")
        if fname:
            return fname
        else:
            return None

    def read_structure_file(self, filename):
        """Read and check structure with a given filename
        Not supporting fractional occupancies
        Display crystal information after successful loading
        """
        basename = os.path.basename(filename)
        if filename.lower().endswith(".cif"):
            atoms = read(filename, format="cif")
            occ = atoms.info.get("occupancy", {})
            fra = [float(value) for site in occ.values() for value in site.values()]
            if fra and min(fra) < 0.999:
                print("WARNING: The structure in the CIF file must be a physical representation of the actual atoms.")
                print(
                    "WARNING: Statistical representation in the form of fractional/partial occupancies is not supported."
                )
                print("WARNING: A supercell may be created to account for disorder and remove partial occupancies.")
                print("WARNING: All partial occupancies will be treated as full atoms in the following calculations.")

        elif "POSCAR" in basename or "CONTCAR" in basename or basename.lower().endswith(".vasp"):
            atoms = read(filename, format="vasp")
        else:
            print("ERROR: File format not supported yet (currently support cif and VASP format)")
            return None
        if not all(atoms.pbc) or atoms.cell.volume <= 0:
            print("ERROR: The structure must be a periodic crystal with a unit cell.")
            return None
        if atoms:
            sg = get_spacegroup(atoms)
            form = atoms.get_chemical_formula()
            cell = atoms.cell
            print("INFO: Crystal structure loaded successfully")
            print("INFO: Cell lengths: ", cell.lengths())
            print("INFO: Cell angles: ", cell.angles())
            print("INFO: Cell volume: ", cell.volume)
            print("INFO: Number of atoms in cell: ", len(atoms.numbers))
            print("INFO: Chemical formula: ", Formula(form).reduce()[0])
            print("INFO: Space group: ", sg.no, sg.symbol)

        return atoms

    # Predictor tab
    # The following functions define actions in the Predictor tab

    def upload_structure_dp(self):
        fname = self.select_structure_file()
        if fname:
            try:
                self.atoms_dp = self.read_structure_file(fname)
                if self.atoms_dp is None:
                    self.ui.statusbar.showMessage(f"Could not load {fname}. See the terminal for details.", 10000)
                    return
                self.ui.label_structure_dp.setText(f"Structure file: {fname}")
                self.ui.statusbar.showMessage("")
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error: {e.__repr__()} when loading {fname}")

    def view_structure_dp(self):
        if self.atoms_dp:
            view(self.atoms_dp)
        else:
            print("ERROR: Select structure file first.")

    def predict_selection_changed(self, *args, **kwargs):
        sender_name = self.sender().objectName()
        if sender_name == "radioButton_predict_phonon_dos":
            self.pred_type = "Phonon DOS"
        elif sender_name == "radioButton_predict_vision":
            self.pred_type = "VISION spectrum"
        else:
            self.pred_type = "S(|Q|, E)"

    def run_prediction(self):
        if self.atoms_dp:
            print("INFO: Prediction started.")
            self.dp_worker.predict(
                model_path=self.predictor_path,
                struc=self.atoms_dp,
                pred_type=self.pred_type,
                partial_dos=self.ui.checkBox_partial_dos.isChecked(),
            )
            print("INFO: Prediction finished.")
            # self.ui.statusbar.showMessage("Prediction done")
        else:
            print("ERROR: Select structure file first.")

    def plot_spec_dp(self):
        self.dp_worker.savenplot(
            model_path=self.predictor_path,
            cwd=self.cwd_path,
            partial_dos=self.ui.checkBox_partial_dos.isChecked(),
            unit=self.ui.comboBox_plot_unit_dp.currentIndex(),
            setrange=self.ui.checkBox_range_dp.isChecked(),
            interactive=self.ui.checkBox_interactive_dp.isChecked(),
            xmin=self.ui.lineEdit_xmin_dp.displayText(),
            xmax=self.ui.lineEdit_xmax_dp.displayText(),
            ymin=self.ui.lineEdit_ymin_dp.displayText(),
            ymax=self.ui.lineEdit_ymax_dp.displayText(),
            zmin=self.ui.lineEdit_zmin_dp.displayText(),
            zmax=self.ui.lineEdit_zmax_dp.displayText(),
        )

    # DFT tab
    # The following functions define actions in the DFT tab

    def cal_with_dftdb(self):
        self.ui.comboBox_search_type.setEnabled(True)
        self.ui.lineEdit_search_value.setEnabled(True)
        self.ui.pushButton_search.setEnabled(True)
        self.ui.tableView.setEnabled(True)
        self.ui.pushButton_dft_folder.setEnabled(False)

    def cal_with_userfile(self):
        self.ui.comboBox_search_type.setEnabled(False)
        self.ui.lineEdit_search_value.setEnabled(False)
        self.ui.pushButton_search.setEnabled(False)
        self.ui.tableView.setEnabled(False)
        self.ui.pushButton_dft_folder.setEnabled(True)

    def choose_dft_folder(self):
        selected_path = QFileDialog.getExistingDirectory(self, "User DFT directory", "")
        if selected_path:
            self.userdft_path = selected_path
            self.ui.label_dft_folder.setText("DFT folder: " + self.userdft_path)

    def search_crystal_structure(self):
        search_type = self.ui.comboBox_search_type.currentText()
        search_keyword = self.ui.lineEdit_search_value.displayText()
        match_exact = self.ui.checkBox_exact.isChecked()
        self.show_df = self.dft_worker.search_structure(search_type, search_keyword, match_exact)
        self.ui.tableView.setModel(PandasModel(self.show_df))

    def load_dft_model(self):
        self.atoms_dft = None
        self._set_visualization_enabled("dft", False)
        if self.ui.radioButton_database.isChecked():
            mp_dfindex = self.ui.tableView.currentIndex().row()
            if mp_dfindex < 0:
                self._show_visualization_error("Select a DFT database result before loading the model.")
                return
            mp_id = self.show_df.iat[mp_dfindex, 0]
            filename = os.path.join(self.dft_database_path, mp_id, "POSCAR-unitcell")
            self.dft_worker.prepare_dft_files(self.cwd_path, self.dft_database_path, mp_id)
        elif self.ui.radioButton_user_dft.isChecked():
            filename = os.path.join(self.userdft_path, "POSCAR-unitcell")
            self.dft_worker.prepare_dft_files(self.cwd_path, self.userdft_path)
        # The DFT files just replaced any MLFF phonon results in the working directory.
        self._invalidate_mlff("DFT files were copied into the working directory")
        try:
            self.atoms_dft = read(filename, format="vasp")
        except:
            print("ERROR: Failed to load structure model")
            print("INFO: The file containing the unit cell structure must be POSCAR-unitcell")
            print("INFO: If it is POSCAR or CONTCAR, please rename it to POSCAR-unitcell")
        else:
            sg = get_spacegroup(self.atoms_dft)
            form = self.atoms_dft.get_chemical_formula()
            cell = self.atoms_dft.cell
            print("INFO: Crystal structure loaded successfully")
            print("INFO: Cell lengths: ", cell.lengths())
            print("INFO: Cell angles: ", cell.angles())
            print("INFO: Cell volume: ", cell.volume)
            print("INFO: Number of atoms in cell: ", len(self.atoms_dft.numbers))
            print("INFO: Chemical formula: ", Formula(form).reduce()[0])
            print("INFO: Space group: ", sg.no, sg.symbol)
            self._set_visualization_enabled("dft", True)

    def view_structure_dft(self):
        if self.atoms_dft:
            view(self.atoms_dft)
        else:
            print("ERROR: Select and load model first.")

    def plot_dos_disp_dft(self):
        if self.atoms_dft:
            self.dft_worker.plot_dos_dispersion()
        else:
            print("ERROR: Select and load model first.")

    def setup_oclimax_dft(self):
        if self.atoms_dft:
            self.dft_worker.oclimax.set_oclimax_cwd(self.cwd_path)
            self.dft_worker.oclimax.exec()
            self.ui.label.setText(f"Parameter file: {self.dft_worker.oclimax.params_file_name}")
        else:
            print("ERROR: Select and load model first.")

    def run_oclimax_dft(self):
        if self.atoms_dft:
            self.dft_worker.oclimax.run_oclimax(parent=self)
        else:
            print("ERROR: Select and load model file first.")

    def plot_spec_dft(self):
        self.dft_worker.oclimax.plot_spec(
            unit=self.ui.comboBox_plot_unit_dft.currentIndex(),
            setrange=self.ui.checkBox_range_dft.isChecked(),
            interactive=self.ui.checkBox_interactive_dft.isChecked(),
            xmin=self.ui.lineEdit_xmin_dft.displayText(),
            xmax=self.ui.lineEdit_xmax_dft.displayText(),
            ymin=self.ui.lineEdit_ymin_dft.displayText(),
            ymax=self.ui.lineEdit_ymax_dft.displayText(),
            zmin=self.ui.lineEdit_zmin_dft.displayText(),
            zmax=self.ui.lineEdit_zmax_dft.displayText(),
        )

    # MLFF tab
    # The following functions define actions in the MLFF tab

    def upload_structure_mlff(self):
        fname = self.select_structure_file()
        if fname:
            self.atoms_mlff = None
            self.mlff_opt = False
            self._set_visualization_enabled("mlff", False)
            try:
                self.atoms_mlff = self.read_structure_file(fname)
                if self.atoms_mlff is None:
                    self.ui.statusbar.showMessage(f"Could not load {fname}. See the terminal for details.", 10000)
                    return
                self.ui.label_structure_mlff.setText(f"Structure file: {fname}")
                self.ui.statusbar.showMessage("")
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error: {e.__repr__()} when loading {fname}")

    def view_structure_mlff(self):
        if self.atoms_mlff:
            view(self.atoms_mlff)
        else:
            print("ERROR: Select structure file first.")

    def opt_and_dos_cal_mlff(self):
        if self.mlff_worker.running:
            self.mlff_worker.cancel()
            return
        if not self.atoms_mlff:
            print("ERROR: No structure loaded.")
            return
        self.mlff_opt = False
        self._set_visualization_enabled("mlff", False)
        try:
            custom = self.ui.checkBox_mlff_model.isChecked()
            model = self.ui.lineEdit_mlff_model_name.text().strip() if custom else None
            if custom and not model:
                raise ValueError("Select a custom model file or uncheck the custom model option.")
            self.mlff_worker.start(
                self.atoms_mlff, self.ui.comboBox_mlff_model.currentData(), self.cwd_path,
                model=model,
                lmin=self.ui.lineEdit_lmin_mlff.text(),
                fmax=self.ui.lineEdit_fmax_mlff.text(),
                nmax=self.ui.lineEdit_nmax_mlff.text(),
                delta=self.ui.lineEdit_delta_mlff.text(),
                relax_cell=self.ui.checkBox_relax_cell.isChecked(),
            )
        except (ValueError, RuntimeError, OSError) as error:
            QMessageBox.warning(self, "MLFF setup", str(error))
            return
        self._set_mlff_busy(True)
        self.mlff_progress_dialog = QProgressDialog("Starting model environment…", "Cancel", 0, 0, self)
        self.mlff_progress_dialog.setWindowTitle("MLFF calculation")
        self.mlff_progress_dialog.setAutoClose(False)
        self.mlff_progress_dialog.canceled.connect(self.mlff_worker.cancel)
        self.mlff_progress_dialog.show()

    def _set_mlff_busy(self, busy):
        for name in ("pushButton_upload_structure_mlff", "groupBox_6", "actionSet_CWD",
                     "pushButton_setup_oclimax_mlff", "pushButton_run_oclimax_mlff", "pushButton_plot_mlff"):
            getattr(self.ui, name).setEnabled(not busy)
        for tab in (self.ui.DFTtab, self.ui.PREDtab):
            self.ui.tabWidget.setTabEnabled(self.ui.tabWidget.indexOf(tab), not busy)
        self.ui.pushButton_opt_phonon_mlff.setText(
            "Cancel calculation" if busy else "Structure optimization + Phonon calculation"
        )

    def _mlff_progress(self, message):
        print(message, end="", flush=True)
        lines = message.strip().splitlines()
        if lines:
            self.ui.statusbar.showMessage(lines[-1][:250])
            if self.mlff_progress_dialog:
                self.mlff_progress_dialog.setLabelText(lines[-1][:250])

    def _finish_mlff_ui(self):
        self._set_mlff_busy(False)
        if self.mlff_progress_dialog:
            self.mlff_progress_dialog.close()
            self.mlff_progress_dialog.deleteLater()
            self.mlff_progress_dialog = None

    def _mlff_completed(self, result):
        self._finish_mlff_ui()
        # MLFF output replaced FORCE_CONSTANTS and removed FORCE_SETS of a loaded DFT model.
        self._invalidate_dft("MLFF results replaced the phonon files in the working directory")
        self.mlff_opt = True
        self.atoms_mlff = read(os.path.join(self.cwd_path, "POSCAR-unitcell"), format="vasp")
        self._set_visualization_enabled("mlff", True)
        self.ui.statusbar.showMessage(f"{result['backend']} calculation complete. Results saved in {self.cwd_path}", 10000)
        try:
            self.mlff_worker.plot_results()
        except Exception as error:
            QMessageBox.warning(self, "Plot unavailable", f"Calculation completed, but the plot could not be opened: {error}")

    def _mlff_failed(self, message):
        self._finish_mlff_ui()
        self.mlff_opt = False
        self._set_visualization_enabled("mlff", False)
        self.ui.statusbar.showMessage("MLFF calculation failed. See the job log.", 10000)
        QMessageBox.warning(self, "MLFF calculation failed", message)

    def _mlff_cancelled(self):
        self._finish_mlff_ui()
        self.mlff_opt = False
        self._set_visualization_enabled("mlff", False)
        self.ui.statusbar.showMessage("MLFF calculation cancelled. Previous output files were preserved.", 10000)

    def closeEvent(self, event):
        running = False
        if self.mlff_worker.running:
            self.mlff_worker.cancel()
            running = True
        for oclimax in (self.dft_worker.oclimax, self.mlff_worker.oclimax):
            if oclimax.running:
                oclimax.process.cancel()
                running = True
        if running:
            self.ui.statusbar.showMessage("Stopping the calculation. Close the window again once it has stopped.")
            event.ignore()
        else:
            event.accept()

    def setup_oclimax_mlff(self):
        if not self.atoms_mlff:
            print("ERROR: No structure loaded.")
            return
        if not self.mlff_opt:
            print("ERROR: Must run structure optimization and phonon calculation first.")
            return
        qden = 35.0
        abc = self.atoms_mlff.cell.cellpar()[0:3]
        nx = int(qden / abc[0]) + 1
        ny = int(qden / abc[1]) + 1
        nz = int(qden / abc[2]) + 1

        self.mlff_worker.oclimax.set_oclimax_cwd(self.cwd_path)
        self.mlff_worker.generate_initial_mesh_file([nx, ny, nz])
        self.mlff_worker.oclimax.exec()
        self.ui.label_param_file_mlff.setText(f"Parameter file: {self.mlff_worker.oclimax.params_file_name}")

    def run_oclimax_mlff(self):
        if not self.atoms_mlff:
            print("ERROR: No structure loaded.")
            return
        if not self.mlff_opt:
            print("ERROR: Must run structure optimization and phonon calculation first.")
            return
        self.mlff_worker.oclimax.run_oclimax(parent=self)

    def plot_spec_mlff(self):
        self.mlff_worker.oclimax.plot_spec(
            unit=self.ui.comboBox_plot_unit_mlff.currentIndex(),
            setrange=self.ui.checkBox_range_mlff.isChecked(),
            interactive=self.ui.checkBox_interactive_mlff.isChecked(),
            xmin=self.ui.lineEdit_xmin_mlff.displayText(),
            xmax=self.ui.lineEdit_xmax_mlff.displayText(),
            ymin=self.ui.lineEdit_ymin_mlff.displayText(),
            ymax=self.ui.lineEdit_ymax_mlff.displayText(),
            zmin=self.ui.lineEdit_zmin_mlff.displayText(),
            zmax=self.ui.lineEdit_zmax_mlff.displayText(),
        )

    def _invalidate_dft(self, reason):
        if self.atoms_dft is not None:
            print(f"INFO: DFT model unloaded because {reason}. Load it again to continue in the DFT tab.")
        self.atoms_dft = None
        self._set_visualization_enabled("dft", False)

    def _invalidate_mlff(self, reason):
        if self.mlff_opt:
            print(f"INFO: MLFF results unloaded because {reason}. Rerun the calculation to continue in the MLFF tab.")
        self.mlff_opt = False
        self._set_visualization_enabled("mlff", False)

    # Visualization tools

    def _set_visualization_enabled(self, workflow, enabled):
        """Enable visualization actions only when their source data is ready."""
        getattr(self.ui, f"pushButton_jmol_{workflow}").setEnabled(enabled)
        getattr(self.ui, f"pushButton_phononweb_{workflow}").setEnabled(enabled)

    def _visualization_context(self):
        sender = self.sender()
        sender_name = sender.objectName() if sender is not None else ""
        if sender_name.endswith("_dft"):
            if self.atoms_dft is None:
                raise VisualizationError("Load a DFT model before opening a visualization.")
            return self.dft_worker, ("POSCAR-unitcell", "FORCE_CONSTANTS", "FORCE_SETS", "BORN", "mesh.conf")
        if sender_name.endswith("_mlff"):
            if not self.mlff_opt:
                raise VisualizationError("Complete the MLFF phonon calculation before opening a visualization.")
            return self.mlff_worker, ("POSCAR-unitcell", "FORCE_CONSTANTS", "BORN")
        raise VisualizationError("Could not determine which calculation should be visualized.")

    def _open_visualization(self, target):
        try:
            worker, source_files = self._visualization_context()
            result = PhononVisualizer(self.cwd_path).visualize(
                target,
                worker.generate_band_yaml,
                source_files=source_files,
            )
        except VisualizationError as error:
            self._show_visualization_error(str(error))
            return
        print(f"INFO: {result.message}")
        self.ui.statusbar.showMessage(result.message, 10000)

    def _show_visualization_error(self, message):
        print(f"ERROR: {message}")
        self.ui.statusbar.showMessage(message, 10000)
        QMessageBox.warning(self, "Visualization unavailable", message)

    def open_jmol(self):
        """Open Gamma-point vibration modes in Jmol."""
        self._open_visualization(VisualizationTarget.JMOL)

    def open_phononwebsite(self):
        """Open the Phonon Website with a current band.yaml ready to upload."""
        self._open_visualization(VisualizationTarget.PHONON_WEBSITE)

    # Help dialogs in the tabs

    def open_help_dp(self):
        help_dp = QDialog()
        self.setup_help = Ui_Help_DP()
        self.setup_help.setupUi(help_dp)
        help_dp.exec()

    def open_help_dft(self):
        help_dft = QDialog()
        self.setup_help = Ui_Help_DFT()
        self.setup_help.setupUi(help_dft)
        help_dft.exec()

    def open_help_mlff(self):
        help_mlff = QDialog()
        self.setup_help = Ui_Help_MLFF()
        self.setup_help.setupUi(help_mlff)
        help_mlff.exec()


def _report_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Keep the GUI alive: PyQt5 otherwise aborts the application on any error in a slot."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"ERROR: Unexpected error\n{details}", file=sys.stderr)
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "INSPIRED error", f"{exc_type.__name__}: {exc_value}\n\nSee the terminal for details.")


def gui():
    """Main entry point for Qt application"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
    # print version and exit
    input_flags = sys.argv[1::]
    if "--v" in input_flags or "--version" in input_flags:
        print(__version__)
        sys.exit()
    # start the app
    print("********************************************************************************************************")
    print("* Inelastic Neutron Scattering Prediction for Instantaneous Results and Experimental Design (INSPIRED) *")
    print("********************************************************************************************************")
    print(f"INFO: version {__version__}")
    print("INFO: Initializing ...")
    sys.excepthook = _report_unhandled_exception
    app = QApplication(sys.argv)
    main_window = INSPIRED()
    print("INFO: Starting GUI ...")
    main_window.show()
    print("INFO: Ready. Set your working directory in Menu, or use current directory by default.")
    print(
        "INFO: It is strongly recommended that a new/empty directory is created/assigned as working directory for each project."
    )
    print(
        "INFO: Existing files from a previous/different model in the working directory may interfere with the calculation and lead to incorrect results."
    )
    print(
        "INFO: If you choose not to change working directory for multiple calculations, please follow through all steps for each calculation (do not skip steps)."
    )
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui()
