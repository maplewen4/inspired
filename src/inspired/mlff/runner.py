"""Standalone MLFF worker, runnable by an independent environment's Python."""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

# Running this file by absolute path does not require installing the GUI package.
if __package__:
    from .backends import create_calculator, get_backend
    from .protocol import PROTOCOL_VERSION, write_json
else:
    from backends import create_calculator, get_backend
    from protocol import PROTOCOL_VERSION, write_json


def run_job(request):
    """Load a model once and reuse it for relaxation and all displacements."""
    import numpy as np
    import torch
    from ase import Atoms
    from ase.filters import ExpCellFilter
    from ase.io import read, write
    from ase.optimize import FIRE
    from phonopy import Phonopy
    from phonopy.file_IO import parse_BORN, write_FORCE_CONSTANTS
    from phonopy.interface.calculator import read_crystal_structure

    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported MLFF request version.")
    backend = get_backend(request["backend"])
    options = request["options"]
    device = request.get("device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64 if backend.environment == "mace" else torch.float32)
    model = request.get("model")
    print(f"INFO: Loading {backend.label} on {device}", flush=True)
    calculator = create_calculator(backend.id, model, device)
    atoms = read("input.extxyz")
    atoms.calc = calculator
    print("INFO: Optimizing structure", flush=True)
    optimizer = FIRE(ExpCellFilter(atoms) if options["relax_cell"] else atoms)
    if not optimizer.run(fmax=options["fmax"], steps=options["nmax"]):
        raise RuntimeError("Structural optimization did not converge. Increase the maximum steps or inspect the structure.")
    # Use the original Atoms object, including when the optimizer uses a cell filter.
    write("POSCAR-unitcell", atoms, direct=True, format="vasp")
    size = options["lmin"]
    dimensions = size if isinstance(size, list) else [max(1, math.ceil(size / length)) for length in atoms.cell.lengths()]
    unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode="vasp")
    phonon = Phonopy(unitcell, supercell_matrix=dimensions, primitive_matrix=np.eye(3))
    phonon.generate_displacements(distance=options["delta"])
    forces = []
    cells = phonon.supercells_with_displacements
    for index, cell in enumerate(cells, 1):
        displaced = Atoms(symbols=cell.symbols, scaled_positions=cell.scaled_positions, cell=cell.cell, pbc=True)
        displaced.calc = calculator
        forces.append(displaced.get_forces())
        print(f"PROGRESS {index}/{len(cells)} displacements", flush=True)
    phonon.forces = forces
    phonon.produce_force_constants()
    if Path("BORN").is_file():
        nac = parse_BORN(phonon.primitive, filename="BORN")
        nac["factor"] = 14.4
        phonon.nac_params = nac
    write_FORCE_CONSTANTS(phonon.force_constants, p2s_map=phonon.primitive.p2s_map)
    phonon.save("phonopy.yaml", settings={"force_constants": True})
    print("INFO: Computing band structure and density of states", flush=True)
    phonon.auto_band_structure(npoints=11, with_eigenvectors=True, write_yaml=True)
    phonon.auto_total_dos()
    phonon.write_total_dos()
    checkpoint = Path(model) if model else None
    digest = None
    if checkpoint and checkpoint.is_file():
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json("result.json", {
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "backend": backend.id,
        "model": model or backend.default_model,
        "checkpoint_sha256": digest,
        "device": device,
        "dtype": str(torch.get_default_dtype()),
        "supercell": dimensions,
        "options": options,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in (backend.distribution, "torch", "ase", "phonopy")
        }},
    })
    print("INFO: MLFF calculation completed", flush=True)


def main():
    """Execute a job or probe the selected environment without loading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe", choices=("mattersim", "sevennet", "orb", "mace", "mace-off"))
    args = parser.parse_args()
    try:
        if args.probe:
            import importlib
            modules = {"mattersim": "mattersim.forcefield.potential", "sevennet": "sevenn.calculator",
                       "orb": "orb_models.forcefield.calculator", "mace": "mace.calculators", "mace-off": "mace.calculators"}
            importlib.import_module(modules[args.probe])
            print(f"{args.probe}: available", flush=True)
        elif args.request:
            run_job(json.loads(args.request.read_text(encoding="utf-8")))
        else:
            parser.error("Supply --request or --probe.")
        return 0
    except Exception as error:
        traceback.print_exc()
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
