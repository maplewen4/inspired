"""Phonopy steps of the OCLIMAX pipeline, run through the phonopy Python API.

These replace the ``phonopy -c POSCAR-unitcell <conf>`` command-line calls. They
write the same ``mesh.yaml``, ``band.yaml`` and ``qpoints.yaml`` files consumed by
the bundled ``oclimax_convert`` binary, but always use the phonopy installed with
INSPIRED (not whichever ``phonopy`` is first on PATH) and load the force
constants once per run instead of once per single-crystal cycle.

Semantics follow phonopy 2.28's command line for the tags INSPIRED's
``mesh.conf`` files use. Any other tag is rejected rather than silently ignored.
One deliberate difference: a BORN file in the working directory enables the
non-analytical term correction (NAC) for every step, unless ``mesh.conf`` sets
``NAC = .FALSE.`` (the command line needed ``NAC = .TRUE.`` for the mesh step).
"""

import os
from dataclasses import dataclass

import numpy as np
import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.phonon.band_structure import get_band_qpoints

from inspired.gui.oclimax_params import conf_key, conf_value

UNITCELL = "POSCAR-unitcell"
MESH_TAGS = {"MP", "MESH", "MESH_NUMBERS"}
SUPPORTED_TAGS = MESH_TAGS | {
    "DIM",
    "GAMMA_CENTER",
    "EIGENVECTORS",
    "FC_SYMMETRY",
    "FORCE_CONSTANTS",
    "NAC",
    "PRIMITIVE_AXES",
    "PRIMITIVE_AXIS",
}


class PhonopyConfError(ValueError):
    """mesh.conf cannot be reproduced faithfully through the API."""


def _flag(value):
    # Phonopy treats ".TRUE.", "TRUE", "T", ".true." ... as true.
    return value.strip().lstrip(".").lower().startswith("t")


@dataclass(frozen=True)
class MeshConf:
    dim: tuple
    mesh: object  # three integers, or a float sampling length
    gamma_center: bool = False
    eigenvectors: bool = False
    fc_symmetry: bool = False
    force_constants: str = ""  # "", "READ" or "WRITE"
    nac: object = None  # True/False when mesh.conf sets NAC, else None (decided by BORN)

    @property
    def use_nac(self):
        """NAC whenever BORN exists, unless mesh.conf explicitly disables it."""
        return os.path.isfile("BORN") if self.nac is None else self.nac

    @property
    def supercell_matrix(self):
        return np.reshape(self.dim, (3, 3)) if len(self.dim) == 9 else list(self.dim)


def read_mesh_conf(path="mesh.conf"):
    """Parse the phonopy configuration INSPIRED uses for OCLIMAX runs."""
    values = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            key = conf_key(line)
            if key is not None:
                values[key] = conf_value(line)
    unsupported = sorted(set(values) - SUPPORTED_TAGS)
    if unsupported:
        raise PhonopyConfError(
            f"{path} contains settings INSPIRED does not apply: {', '.join(unsupported)}. "
            "Remove them or run phonopy manually."
        )
    for key in ("PRIMITIVE_AXES", "PRIMITIVE_AXIS"):
        if key in values and values[key].upper() not in ("P", "1 0 0 0 1 0 0 0 1"):
            try:
                identity = np.allclose(np.array(values[key].split(), dtype=float), np.eye(3).ravel())
            except ValueError:
                identity = False
            if not identity:
                raise PhonopyConfError(f"{path}: INSPIRED requires the unit cell as primitive cell ({key} = identity).")
    try:
        dim = tuple(int(value) for value in values["DIM"].split())
    except KeyError:
        raise PhonopyConfError(f"{path} must define the supercell (DIM).") from None
    except ValueError:
        raise PhonopyConfError(f"{path}: DIM must be integers.") from None
    if len(dim) not in (3, 9):
        raise PhonopyConfError(f"{path}: DIM needs 3 or 9 integers.")
    mesh_values = [values[key] for key in ("MP", "MESH", "MESH_NUMBERS") if key in values]
    if not mesh_values:
        raise PhonopyConfError(f"{path} must define the sampling mesh (MP or MESH).")
    try:
        numbers = mesh_values[0].split()
        mesh = float(numbers[0]) if len(numbers) == 1 else [int(value) for value in numbers]
    except (ValueError, IndexError):
        raise PhonopyConfError(f"{path}: the sampling mesh must be three integers or one length.") from None
    if isinstance(mesh, list) and len(mesh) != 3:
        raise PhonopyConfError(f"{path}: the sampling mesh needs three integers.")
    force_constants = values.get("FORCE_CONSTANTS", "").strip().upper()
    if force_constants not in ("", "READ", "WRITE"):
        raise PhonopyConfError(f"{path}: FORCE_CONSTANTS must be READ or WRITE.")
    return MeshConf(
        dim=dim,
        mesh=mesh,
        gamma_center=_flag(values.get("GAMMA_CENTER", "F")),
        eigenvectors=_flag(values.get("EIGENVECTORS", "F")),
        fc_symmetry=_flag(values.get("FC_SYMMETRY", "F")),
        force_constants=force_constants,
        nac=_flag(values["NAC"]) if "NAC" in values else None,
    )


def load_phonon(conf, *, read_force_constants, symmetrize_fc, nac):
    """Load the harmonic model as ``phonopy -c POSCAR-unitcell`` would."""
    source = "FORCE_CONSTANTS" if read_force_constants else "FORCE_SETS"
    if not os.path.isfile(source):
        raise FileNotFoundError(f"{source} is required by the phonopy settings but was not found.")
    if nac and not os.path.isfile("BORN"):
        raise FileNotFoundError("NAC is enabled but the BORN file was not found.")
    phonon = phonopy.load(
        supercell_matrix=conf.supercell_matrix,
        primitive_matrix=np.eye(3),
        unitcell_filename=UNITCELL,
        force_constants_filename=source if read_force_constants else None,
        force_sets_filename=None if read_force_constants else source,
        is_nac=nac,
        born_filename="BORN" if nac else None,
        symmetrize_fc=symmetrize_fc,
        log_level=0,
    )
    if conf.force_constants == "WRITE" and not read_force_constants:
        write_FORCE_CONSTANTS(phonon.force_constants, p2s_map=phonon.primitive.p2s_map)
    return phonon


def mesh_phonon(conf):
    """The model for the mesh step, honouring mesh.conf exactly."""
    return load_phonon(
        conf,
        read_force_constants=conf.force_constants == "READ",
        symmetrize_fc=conf.fc_symmetry,
        nac=conf.use_nac,
    )


def single_crystal_phonon(conf):
    """The model for single-crystal cuts: prefer FORCE_CONSTANTS, symmetrize, NAC as for the mesh."""
    return load_phonon(
        conf,
        read_force_constants=os.path.isfile("FORCE_CONSTANTS"),
        symmetrize_fc=True,
        nac=conf.use_nac,
    )


def write_mesh_yaml(phonon, conf):
    """Write mesh.yaml (phonopy's MESH mode)."""
    phonon.run_mesh(conf.mesh, with_eigenvectors=conf.eigenvectors, is_gamma_center=conf.gamma_center)
    phonon.write_yaml_mesh()


def write_band_yaml(phonon, path, npoints):
    """Write band.yaml for one connected path of q-points (phonopy's BAND mode)."""
    bands = get_band_qpoints([np.asarray(path, dtype=float)], npoints=npoints)
    connections = [True] * (len(path) - 2) + [False]
    phonon.run_band_structure(bands, with_eigenvectors=True, path_connections=connections)
    phonon.write_yaml_band_structure()


def write_qpoints_yaml(phonon, qpoints):
    """Write qpoints.yaml for explicit q-points (phonopy's QPOINTS mode)."""
    phonon.run_qpoints(np.asarray(qpoints, dtype=float), with_eigenvectors=True)
    phonon.write_yaml_qpoints_phonon()
