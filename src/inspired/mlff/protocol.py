"""Small, versioned job contract shared by the GUI and isolated workers."""

import json
import math
from pathlib import Path

PROTOCOL_VERSION = 1
ARTIFACTS = ("POSCAR-unitcell", "FORCE_CONSTANTS", "phonopy.yaml", "band.yaml", "total_dos.dat")


def calculation_options(lmin=12, fmax=0.01, nmax=100, delta=0.03, relax_cell=False):
    """Validate GUI inputs before starting an expensive model process."""
    dimensions = str(lmin).split()
    if len(dimensions) == 1:
        size = float(dimensions[0])
        if not math.isfinite(size) or size <= 0:
            raise ValueError("Lmin must be a positive number.")
    elif len(dimensions) == 3:
        size = [int(value) for value in dimensions]
        if min(size) < 1:
            raise ValueError("Supercell dimensions must be positive integers.")
    else:
        raise ValueError("Lmin/Dim must be one positive length or three positive integers.")
    force = float(fmax)
    displacement = float(delta)
    steps = int(nmax)
    if not all(math.isfinite(value) and value > 0 for value in (force, displacement)) or steps < 1:
        raise ValueError("Force tolerance, displacement, and optimization steps must be positive.")
    return dict(lmin=size, fmax=force, nmax=steps, delta=displacement, relax_cell=bool(relax_cell))


def write_json(path, data):
    """Publish JSON atomically so readers never see a partial manifest."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_result(directory):
    """Require a completed, compatible result with all expected artifacts."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result.get("protocol_version") != PROTOCOL_VERSION or result.get("status") != "completed":
        raise ValueError("The MLFF worker did not return a compatible completed result.")
    for filename in ARTIFACTS:
        if not (directory / filename).is_file():
            raise ValueError(f"The MLFF worker did not produce {filename}.")
    return result
