"""Initialize and access OCLIMAX parameters and the phonopy mesh.conf file."""

import os
from pathlib import Path

MESH_CONF = "mesh.conf"
# Phonopy accepts any of these tags for the sampling mesh.
MESH_TAGS = ("MP", "MESH", "MESH_NUMBERS")


def conf_key(line):
    """Return the upper-case tag of a phonopy ``TAG = value`` line, or None."""
    content = line.split("#", 1)[0]
    if "=" not in content:
        return None
    return content.split("=", 1)[0].strip().upper()


def conf_value(line):
    """Return the value of a phonopy ``TAG = value`` line without comments."""
    return line.split("#", 1)[0].split("=", 1)[1].strip()


def find_conf_line(path, tags):
    """Return the first line of *path* assigning one of *tags*, or None."""
    path = Path(path)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if conf_key(line) in tags:
            return line.strip()
    return None


def set_mesh(path, mesh_list):
    """Replace (or append) the sampling mesh in a phonopy configuration file."""
    path = Path(path)
    mesh = " ".join(str(value) for value in mesh_list)
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        key = conf_key(line)
        if key in MESH_TAGS:
            lines[index] = f"{key} = {mesh}"
            replaced = True
    if not replaced:
        lines.append(f"MP = {mesh}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


class OclimaxParams:
    def __init__(self):
        self.dict_default_params = {
            "TASK": 0,
            "INSTR": 0,
            "TEMP": "0.00",
            "E_UNIT": 1,
            "OUTPUT": "0",
            "MAXO": "10",
            "CONV": "2",
            "PHASE": "0",
            "MASK": "0",
            "NORM": "0",
            "ELASTIC": "-1 -1",
            "MINE": "0.1",
            "MAXE": "400.00",
            "dE": "0.1",
            "ECUT": "0.01",
            "ERES": "0.31  0.5E-02  0.8E-06",
            "MINQ": "0.10",
            "MAXQ": "10.00",
            "dQ": "0.10",
            "QRES": "-1",
            "THETA": "0.0  60.0",
            "Ef": "4.00",
            "Ei": "400.00",
            "L1": "11.60",
            "L2": "2.00",
            "L3": "3.00",
            "dt_m": "3.91",
            "dt_ch": "5.95",
            "dL3": "3.50",
            "MINQ_y": "1.00",
            "MAXQ_y": "2.00",
            "dQ_y": "0.02",
            "WING": "0",
            "A_ISO": "0.0350",
            "W_WIDTH": "20.0",
            "Q1": "1  0  0",
            "Q2": "0  1  0",
            "Q3": "0  0  1",
            "HKL": "0  0  0",
            "Q1bin": "2.0  0.01  5.0",
            "Q2bin": "-0.05  0.05  0.05",
            "Q3bin": "-0.05  0.05  0.05",
            "Ebin": "0.1  0.1  100.0",
            "x-axis": 0,
            "y-axis": 0,
        }

        self.dict_params = self.dict_default_params.copy()
        self.default_mesh = ["1", "1", "1"]
        self.supercell = None

    def get_supercell(self):
        line = find_conf_line(MESH_CONF, ("DIM",))
        if line is None:
            print(
                "ERROR: Could not find DIM in mesh.conf. This file is required to continue the calculation correctly."
            )
            return
        try:
            scm = [int(value) for value in conf_value(line).split()]
        except ValueError:
            scm = []
        if len(scm) == 9:
            self.supercell = [scm[0:3], scm[3:6], scm[6:9]]
        elif len(scm) == 3:
            self.supercell = scm
        else:
            print("ERROR: Check supercell dimension in mesh.conf file.")

    def get_default_mesh(self):
        line = find_conf_line(MESH_CONF, MESH_TAGS)
        if line is None:
            print(
                "ERROR: Could not find MP/MESH in mesh.conf. This file is required to continue the calculation correctly."
            )
            return
        self.default_mesh = conf_value(line).split()[-3:]

    def generate_mesh_file(self, mesh_list=("1", "1", "1")):
        if not os.path.isfile(MESH_CONF):
            print(
                "ERROR: Could not open/read mesh.conf file. This file is required to continue the calculation correctly."
            )
            return
        set_mesh(MESH_CONF, mesh_list)

    def load_parameters(self, params_file_path="oclimax.params"):
        try:
            with open(params_file_path, "r") as pf:
                para_lines = pf.readlines()
        except OSError:
            print("INFO: Could not open/read params file, make sure it is in current working directory.")
            return
        for s in para_lines:
            if s.startswith("#!"):
                s = s[2:]
            param = [p.strip() for p in s.split("#", 1)[0].split("=", 1)]
            if len(param) == 2 and param[0] in list(self.dict_params.keys()):
                if param[0] in ["TASK", "INSTR", "E_UNIT", "x-axis", "y-axis"]:
                    try:
                        self.dict_params[param[0]] = int(param[1])
                    except ValueError:
                        print(f"WARNING: Ignoring non-integer {param[0]} value in {params_file_path}: {param[1]}")
                else:
                    self.dict_params[param[0]] = param[1]

    def write_new_parameter_file(self, save_params_file_path="oclimax.params"):
        list_file_lines = []
        list_file_lines.append("# This file was generated by INSPIRED and should be used within INSPIRED.")
        list_file_lines.append(
            "# Using this file directly with the standalone OCLIMAX is discouraged and may not produce the intended results."
        )
        list_file_lines.append("TASK    =   " + str(self.dict_params["TASK"]))
        list_file_lines.append("INSTR   =   " + str(self.dict_params["INSTR"]))
        list_file_lines.append("TEMP    =   " + self.dict_params["TEMP"])
        list_file_lines.append("E_UNIT  =   " + str(self.dict_params["E_UNIT"]))
        list_file_lines.append("OUTPUT  =   " + self.dict_params["OUTPUT"])
        list_file_lines.append("MAXO    =   " + self.dict_params["MAXO"])
        list_file_lines.append("CONV    =   " + self.dict_params["CONV"])
        list_file_lines.append("PHASE   =   " + self.dict_params["PHASE"])
        list_file_lines.append("MASK    =   " + self.dict_params["MASK"])
        list_file_lines.append("NORM    =   " + self.dict_params["NORM"])
        list_file_lines.append("ELASTIC =   " + self.dict_params["ELASTIC"])
        list_file_lines.append("MINE    =   " + self.dict_params["MINE"])
        list_file_lines.append("MAXE    =   " + self.dict_params["MAXE"])
        list_file_lines.append("dE      =   " + self.dict_params["dE"])
        list_file_lines.append("ECUT    =   " + self.dict_params["ECUT"])
        list_file_lines.append("ERES    =   " + self.dict_params["ERES"])
        list_file_lines.append("MINQ    =   " + self.dict_params["MINQ"])
        list_file_lines.append("MAXQ    =   " + self.dict_params["MAXQ"])
        list_file_lines.append("dQ      =   " + self.dict_params["dQ"])
        list_file_lines.append("QRES    =   " + self.dict_params["QRES"])
        list_file_lines.append("THETA   =   " + self.dict_params["THETA"])
        list_file_lines.append("Ef      =   " + self.dict_params["Ef"])
        list_file_lines.append("Ei      =   " + self.dict_params["Ei"])
        list_file_lines.append("L1      =   " + self.dict_params["L1"])
        list_file_lines.append("L2      =   " + self.dict_params["L2"])
        list_file_lines.append("L3      =   " + self.dict_params["L3"])
        list_file_lines.append("dt_m    =   " + self.dict_params["dt_m"])
        list_file_lines.append("dt_ch   =   " + self.dict_params["dt_ch"])
        list_file_lines.append("dL3     =   " + self.dict_params["dL3"])
        list_file_lines.append("MINQ_y  =   " + self.dict_params["MINQ_y"])
        list_file_lines.append("MAXQ_y  =   " + self.dict_params["MAXQ_y"])
        list_file_lines.append("dQ_y    =   " + self.dict_params["dQ_y"])
        list_file_lines.append("WING    =   " + self.dict_params["WING"])
        list_file_lines.append("A_ISO   =   " + self.dict_params["A_ISO"])
        list_file_lines.append("W_WIDTH =   " + self.dict_params["W_WIDTH"])
        list_file_lines.append("#! Q1 =   " + self.dict_params["Q1"])
        list_file_lines.append("#! Q2 =   " + self.dict_params["Q2"])
        list_file_lines.append("#! Q3 =   " + self.dict_params["Q3"])
        list_file_lines.append("#! Q1bin =   " + self.dict_params["Q1bin"])
        list_file_lines.append("#! Q2bin =   " + self.dict_params["Q2bin"])
        list_file_lines.append("#! Q3bin =   " + self.dict_params["Q3bin"])
        list_file_lines.append("#! HKL =   " + self.dict_params["HKL"])
        list_file_lines.append("#! Ebin =   " + self.dict_params["Ebin"])
        list_file_lines.append("#! x-axis =   " + str(self.dict_params["x-axis"]))
        list_file_lines.append("#! y-axis =   " + str(self.dict_params["y-axis"]))

        with open(save_params_file_path, "w") as pf:
            pf.write("\n".join(list_file_lines))

    def update_param(self, param_key, param):
        self.dict_params[param_key] = param

    def get_param(self, param_key):
        return self.dict_params[param_key]

    def reset_params_default(self):
        self.dict_params = self.dict_default_params.copy()
