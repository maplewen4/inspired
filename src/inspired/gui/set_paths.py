import os
import sys
from pathlib import Path

from qtpy.QtWidgets import QDialog, QFileDialog

from inspired.gui.ui_set_paths import Ui_SetPaths

PATH_KEYS = ("predictor_path", "dft_database_path")


def user_config_path():
    """Location of the per-user path configuration."""
    return Path.home() / ".config" / "inspired" / "config"


def parse_path_config(text):
    """Parse ``key: value`` lines; tolerate blank lines, comments and colons in paths."""
    paths = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        paths[key.strip()] = value.strip()
    return paths


def write_path_config(path, predictor_path, dft_database_path):
    """Write the configuration, creating its directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"predictor_path:{predictor_path}\ndft_database_path:{dft_database_path}\n",
        encoding="utf-8",
    )


class SetPaths(QDialog):
    def __init__(self):
        super(SetPaths, self).__init__()
        self.ui = Ui_SetPaths()
        self.ui.setupUi(self)

    def init_paths(self, predictor_path, dft_database_path):
        self.predictor_path = predictor_path
        self.dft_database_path = dft_database_path

        self.ui.lineEdit_dp_path.setText(predictor_path)
        self.ui.lineEdit_dft_path.setText(dft_database_path)

    def browse_dftdb(self):
        wd = QFileDialog.getExistingDirectory(self, "Path to DFT database", "")
        if wd:
            self.dft_database_path = wd
            self.ui.lineEdit_dft_path.setText(wd)

    def browse_dp_model(self):
        wd = QFileDialog.getExistingDirectory(self, "Path to predictor models", "")
        if wd:
            self.predictor_path = wd
            self.ui.lineEdit_dp_path.setText(wd)

    def save_paths(self):
        self.predictor_path = os.path.expanduser(self.ui.lineEdit_dp_path.displayText().strip())
        self.dft_database_path = os.path.expanduser(self.ui.lineEdit_dft_path.displayText().strip())
        write_path_config(user_config_path(), self.predictor_path, self.dft_database_path)
        self.accept()

    def quit_inspired(self):
        sys.exit()
