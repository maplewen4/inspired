"""OCLIMAX INS simulation as a standalone, cancellable process.

The GUI validates the request, then runs ``python -m inspired.gui.oclimax_job``
in the working directory and reads its output:

* ``PROGRESS <done>/<total> <message>`` lines drive the progress dialog,
* ``RESULT <path>`` names the simulated spectrum on success,
* ``ERROR: ...`` lines explain a failure (exit status 1).

The job starts its own process group, so cancelling stops the bundled
``oclimax_run``/``oclimax_convert`` binaries together with this process.
No Qt is imported here.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from inspired.gui import phonopy_steps

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oclimax")
REQUIRED_FILES = ("POSCAR-unitcell", "mesh.conf")


class JobError(RuntimeError):
    """A step failed; the message is shown to the user."""


def check_inputs(task, params_file):
    """Return a list of problems with the working directory; warnings are printed."""
    problems = [
        f"{name} file in working directory is required for this calculation."
        for name in REQUIRED_FILES
        if not os.path.isfile(name)
    ]
    if not (os.path.isfile("FORCE_SETS") or os.path.isfile("FORCE_CONSTANTS")):
        problems.append("FORCE_SETS or FORCE_CONSTANTS file is required for this calculation.")
    elif os.path.isfile("FORCE_SETS") and os.path.isfile("FORCE_CONSTANTS"):
        print("WARNING: Found both FORCE_SETS and FORCE_CONSTANTS files. Inconsistency issue possible.")
    if task in (0, 1) and not os.path.isfile(params_file):
        problems.append(f"OCLIMAX parameter file {params_file} not found. Set the INS parameters first.")
    if task not in (0, 1, 2, 3):
        problems.append(f"Unsupported OCLIMAX task {task}.")
    return problems


def single_crystal_bins(params, bscale):
    """Parse and validate single-crystal Q/E binning; raise ValueError with a user message."""
    task = params["TASK"]
    try:
        Q = [params[key].split() for key in ("Q1", "Q2", "Q3")]
        Qb = [params[key].split() for key in ("Q1bin", "Q2bin", "Q3bin")]
        if any(len(param) != 3 for param in Q + Qb):
            raise ValueError(
                'Check Q and Q_bin parameters (each should have three numbers, click "Help" for more information)'
            )
        Q = np.array(Q, dtype=float)
        Qb = np.array(Qb, dtype=float)
        HKL = np.array(params["HKL"].split(), dtype=float)
        Eb = np.array(params["Ebin"].split(), dtype=float)
        bscale = float(bscale)
    except ValueError as error:
        if "Check" in str(error):
            raise
        raise ValueError(f"Single crystal parameters must be numbers ({error})") from None
    if len(HKL) != 3:
        raise ValueError("Check HKL parameter (three numbers are required)")
    for param in Qb:
        if param[0] > param[2]:
            raise ValueError("Check Qbin parameters ([Qmin,Qstep,Qmax])")
        if param[1] < 0:
            raise ValueError("Check Qbin parameters (Qstep cannot be negative)")
        if param[1] == 0:
            if param[0] != param[2]:
                raise ValueError("Check Qbin parameters (Qstep can be zero only if Qmin=Qmax)")
            param[1] = 1
    if len(Eb) == 2 and task == 3:
        if Eb[1] <= Eb[0]:
            raise ValueError("Check E_bin parameters (Emax must be greater than Emin)")
        Eb = np.array([Eb[0], (Eb[1] - Eb[0]) / 2.0, Eb[1]])
    elif len(Eb) != 3:
        raise ValueError("Check E_bin parameters ([Emin,Estep,Emax] for the task)")
    elif Eb[2] <= Eb[0]:
        raise ValueError("Check E_bin parameters (Emax must be greater than Emin)")
    elif Eb[1] <= 0:
        raise ValueError("Check E_bin parameters (Estep must be greater than zero)")
    if bscale <= 0:
        raise ValueError("BAND_POINTS scaling factor must be positive")
    if task == 3:
        xaxis, yaxis = params["x-axis"], params["y-axis"] - 1
        if yaxis < 0 or yaxis == xaxis:
            raise ValueError("Choose different Q directions for the x and y axes.")
    return Q, Qb, HKL, Eb, bscale


def validate(request):
    """Check a request before starting a process; raise ValueError with all problems."""
    problems = check_inputs(request["params"]["TASK"], request["params_file"])
    if not problems and request["params"]["TASK"] in (2, 3):
        try:
            single_crystal_bins(request["params"], request["bscale"])
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("\n".join(problems))


def _bins(start, stop, step):
    return np.linspace(start, stop, int(round((stop - start) / step)) + 1)


class OclimaxJob:
    """Run phonopy (Python API) and the OCLIMAX binaries for one request."""

    def __init__(self, request, tools_dir=TOOLS_DIR):
        self.params = request["params"]
        self.params_file = request["params_file"]
        self.bscale = request["bscale"]
        self.tools_dir = tools_dir
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.done = 0
        self.total = 1

    # -- reporting -----------------------------------------------------------------------------
    def progress(self, message, advance=True):
        if advance:
            self.done += 1
        print(f"PROGRESS {self.done}/{self.total} {message}", flush=True)

    # -- steps ---------------------------------------------------------------------------------
    def tool(self, name):
        return os.path.join(self.tools_dir, name)

    def run_program(self, args, failure):
        try:
            subprocess.run(args, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise JobError(f"{failure} ({error})") from error

    @staticmethod
    def phonopy_step(action, *args):
        try:
            return action(*args)
        except Exception as error:
            raise JobError(f"Phonopy calculation failed ({error})") from error

    @staticmethod
    def single_output(pattern, step):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise JobError(f"{step} did not produce an output file matching {pattern}. See messages above.")
        return matches[0]

    def mesh_step(self, output):
        """Write mesh.yaml with the phonopy API and convert it to an OCLIMAX input file."""
        self.progress("Calculating phonons on the sampling mesh", advance=False)
        conf = self.phonopy_step(phonopy_steps.read_mesh_conf)
        self.phonopy_step(phonopy_steps.write_mesh_yaml, self.phonopy_step(phonopy_steps.mesh_phonon, conf), conf)
        self.progress("Converting mesh.yaml for OCLIMAX")
        self.run_program(
            [self.tool("oclimax_convert"), "-yaml", "mesh.yaml", "-o", output],
            "File format conversion failed.",
        )
        self.progress("Phonon mesh ready")

    def merge(self, output, cuts):
        self.progress("Integrating cuts", advance=False)
        # Use this interpreter: a bare "python" on PATH may lack numpy or be a different version.
        self.run_program([sys.executable, self.tool("merge_csv.py"), output, *cuts], "Merging files failed.")
        merged = self.single_output(output + "*.csv", "Merging cuts")
        self.progress("Cuts integrated")
        return merged

    def write_sc_params(self, task, Eb, Qxb, p4, Qx, extra=()):
        get = self.params.get
        with open("oclimax-sc.params", "w") as f:
            print("TASK    =         " + str(task), file=f)
            print("INSTR   =         3", file=f)
            print("TEMP    =     ", get("TEMP"), file=f)
            print("E_UNIT  =     ", get("E_UNIT"), file=f)
            print("OUTPUT  =     ", get("OUTPUT"), file=f)
            print("MAXO    =     ", get("MAXO"), file=f)
            print("MASK    =     ", get("MASK"), file=f)
            print("NORM    =     ", get("NORM"), file=f)
            print("MINE    =     ", Eb[0], file=f)
            print("MAXE    =     ", Eb[2], file=f)
            print("dE      =     ", Eb[1], file=f)
            print("ECUT    =     ", get("ECUT"), file=f)
            print("ERES    =     ", get("ERES"), file=f)
            print("MINQ    =     ", Qxb[0], file=f)
            print("MAXQ    =     ", Qxb[2], file=f)
            print("dQ      =     ", Qxb[1], file=f)
            print("QRES    =     ", get("QRES"), file=f)
            print("THETA   =     ", get("THETA"), file=f)
            print("Ei      =     ", get("Ei"), file=f)
            print("HKL     =     ", p4[0], p4[1], p4[2], file=f)
            print("Q_vec   =     ", Qx[0], Qx[1], Qx[2], file=f)
            for line in extra:
                print(line, file=f)

    def run_cut(self, generated_pattern, outfile):
        self.run_program(
            [self.tool("oclimax_run"), "mesh.oclimax", "oclimax-sc.params", "band_" + self.timestr + ".oclimax"],
            "OCLIMAX simulation failed.",
        )
        os.replace(self.single_output(generated_pattern, "OCLIMAX"), outfile)
        return outfile

    # -- tasks ---------------------------------------------------------------------------------
    def run(self):
        task = self.params["TASK"]
        if task in (0, 1):
            return self.run_powder()
        return self.run_single_crystal(task)

    def run_powder(self):
        self.total = 3
        output = "oclimax_" + self.timestr
        self.mesh_step(output)
        self.progress("Running OCLIMAX simulation", advance=False)
        self.run_program(
            [self.tool("oclimax_run"), output + ".oclimax", self.params_file], "OCLIMAX simulation failed."
        )
        return self.single_output(output + "*K.csv", "OCLIMAX")

    def run_single_crystal(self, task):
        Q, Qb, HKL, Eb, bscale = single_crystal_bins(self.params, self.bscale)
        print("INFO: BAND_POINTS scaling factor set to " + str(bscale), flush=True)
        xaxis = self.params["x-axis"]
        if task == 2:
            s = [0, 1, 2]
            s.remove(xaxis)
            q1 = _bins(*Qb[s[0]][[0, 2, 1]])
            q2 = _bins(*Qb[s[1]][[0, 2, 1]])
            cycles = [(i, j) for i in range(len(q1)) for j in range(len(q2))]
        else:
            yaxis = self.params["y-axis"] - 1
            s = [0, 1, 2]
            s.remove(xaxis)
            s.remove(yaxis)
            q1 = _bins(*Qb[s[0]][[0, 2, 1]])
            cycles = [(i, None) for i in range(len(q1))]
        self.total = len(cycles) + 4

        self.progress("Loading force constants", advance=False)
        # One model for every cut: force constants are built (and symmetrized) once, not once per cycle.
        phonon = self.phonopy_step(phonopy_steps.single_crystal_phonon, self.phonopy_step(phonopy_steps.read_mesh_conf))
        self.progress("Force constants ready")
        self.mesh_step("mesh")

        Qx, Qxb = Q[xaxis], Qb[xaxis]
        cuts = []
        for count, (i, j) in enumerate(cycles, 1):
            self.progress(f"Cycle {count} of {len(cycles)}", advance=False)
            if task == 2:
                qQ1, qQ2 = Q[s[0]], Q[s[1]]
                npoint = max(int(bscale * 1.0 / Qxb[1]) + 1, int(bscale * (Eb[2] - Eb[0]) / Eb[1]) + 1)
                p1 = -0.5 * Qx + q1[i] * qQ1 + q2[j] * qQ2
                p2 = q1[i] * qQ1 + q2[j] * qQ2
                p3 = 0.5 * Qx + q1[i] * qQ1 + q2[j] * qQ2
                p4 = p2 + HKL[0] * Q[0] + HKL[1] * Q[1] + HKL[2] * Q[2]
                self.phonopy_step(phonopy_steps.write_band_yaml, phonon, [p1, p2, p3], npoint)
                self.run_program(
                    [self.tool("oclimax_convert"), "-yamld", "band.yaml", "-o", "band_" + self.timestr],
                    "File format conversion failed.",
                )
                self.write_sc_params(2, Eb, Qxb, p4, Qx)
                cuts.append(
                    self.run_cut("band_" + self.timestr + "_2Dmesh_scqw_*K.csv", f"cut_qe_{self.timestr}_{i}_{j}.csv")
                )
            else:
                Qy, Qyb, qQ1 = Q[yaxis], Qb[yaxis], Q[s[0]]
                qx = np.linspace(-0.5, 0.5, int(bscale * 1.0 / Qxb[1]) + 1)
                qy = np.linspace(-0.5, 0.5, int(bscale * 1.0 / Qyb[1]) + 1)
                shft = q1[i] * qQ1
                p4 = shft + HKL[0] * Q[0] + HKL[1] * Q[1] + HKL[2] * Q[2]
                qpoints = [qx[a] * Qx + qy[b] * Qy + shft for a in range(len(qx)) for b in range(len(qy))]
                self.phonopy_step(phonopy_steps.write_qpoints_yaml, phonon, qpoints)
                self.run_program(
                    [
                        self.tool("oclimax_convert"),
                        "-vpd",
                        "POSCAR-unitcell",
                        "qpoints.yaml",
                        "-o",
                        "band_" + self.timestr,
                    ],
                    "File format conversion failed.",
                )
                extra = (
                    f"Q_vec_y =      {Qy[0]} {Qy[1]} {Qy[2]}",
                    f"MINQ_y  =      {Qyb[0]}",
                    f"MAXQ_y  =      {Qyb[2]}",
                    f"dQ_y    =      {Qyb[1]}",
                )
                self.write_sc_params(3, Eb, Qxb, p4, Qx, extra=extra)
                cuts.append(
                    self.run_cut("band_" + self.timestr + "_2Dmesh_scqq_*K.csv", f"cut_qq_{self.timestr}_{i}.csv")
                )
            self.progress(f"Cycle {count} of {len(cycles)} finished")
        kind = "qe" if task == 2 else "qq"
        return self.merge(f"oclimax_sc_{kind}_{self.timestr}", cuts)


def main(argv=None):
    """Run one request in the current directory; print RESULT on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="JSON request written by the GUI")
    parser.add_argument("--tools", default=TOOLS_DIR, help="directory holding the OCLIMAX binaries")
    args = parser.parse_args(argv)
    if hasattr(os, "setsid"):
        try:
            os.setsid()  # own process group, so cancelling also stops the binaries started below
        except OSError:
            pass
    try:
        request = json.loads(args.request)
        problems = check_inputs(request["params"]["TASK"], request["params_file"])
        if problems:
            raise JobError("\n".join(problems))
        result = OclimaxJob(request, tools_dir=args.tools).run()
    except (JobError, ValueError) as error:
        for line in str(error).splitlines():
            print(f"ERROR: {line}", flush=True)
        return 1
    print(f"INFO: OCLIMAX simulation finished. Output file: {result}", flush=True)
    print(f"RESULT {os.path.abspath(result)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
