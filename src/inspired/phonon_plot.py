"""Phonon dispersion and DOS plots shared by the DFT and MLFF workflows.

INSPIRED keeps the input unit cell as phonopy's primitive cell, because OCLIMAX
inputs, single-crystal Q in r.l.u. and BORN files are defined on that cell. For
a conventional cell this folds the dispersion (e.g. 12 branches instead of 3 for
fcc Cu). For plotting only, the model is re-expressed on the standard primitive
cell, whose seekpath band path has the usual high-symmetry labels.
"""

import contextlib
import inspect
import io

import numpy as np
from phonopy import Phonopy
from phonopy.harmonic import force_constants as fc_tools
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath


def _full_force_constants(phonon):
    force_constants = phonon.force_constants
    if force_constants.shape[0] == force_constants.shape[1]:
        return force_constants
    # The first parameter is the Phonopy object in phonopy 2.x and its primitive cell in 3+.
    first = next(iter(inspect.signature(fc_tools.compact_fc_to_full_fc).parameters))
    owner = phonon if first == "phonon" else phonon.primitive
    return fc_tools.compact_fc_to_full_fc(owner, force_constants)


def _primitive_index(phonon, supercell_indices):
    """Map supercell atom indices to indices in *phonon*'s primitive cell."""
    primitive = phonon.primitive
    return [primitive.p2p_map[primitive.s2p_map[index]] for index in supercell_indices]


def standard_primitive_phonon(phonon):
    """Return the same harmonic model on the standard primitive cell.

    Force constants and Born charges are mapped atom-by-atom through the shared
    supercell, so no file needs to be re-read and compact force constants work.
    Returns *phonon* unchanged when it is already primitive.
    """
    primitive = Phonopy(phonon.unitcell, supercell_matrix=phonon.supercell_matrix, primitive_matrix="auto")
    if len(primitive.primitive) == len(phonon.primitive):
        return phonon
    primitive.force_constants = _full_force_constants(phonon)
    nac_params = phonon.nac_params
    if nac_params is not None:
        nac_params = dict(nac_params)
        nac_params["born"] = np.asarray(nac_params["born"])[_primitive_index(phonon, primitive.primitive.p2s_map)]
        primitive.nac_params = nac_params
    return primitive


def plot_band_structure_and_dos(phonon):
    """Plot a labelled dispersion with the DOS on the standard primitive cell."""
    try:
        plot_phonon = standard_primitive_phonon(phonon)
    except Exception as error:
        print(f"WARNING: Could not reduce to the standard primitive cell ({error}); plotting the input cell.")
        plot_phonon = phonon
    if plot_phonon is not phonon:
        print(
            f"INFO: Dispersion plotted on the standard primitive cell ({len(plot_phonon.primitive)} atoms) "
            f"instead of the input cell ({len(phonon.primitive)} atoms), so branches are not folded."
        )
    _print_special_points(plot_phonon)
    plot_phonon.auto_band_structure()
    plot_phonon.auto_total_dos()
    return plot_phonon.plot_band_structure_and_dos()


def _print_special_points(phonon):
    """List the path labels with coordinates in the plotted cell's reciprocal basis."""
    with contextlib.redirect_stderr(io.StringIO()):
        bands, labels, _ = get_band_qpoints_by_seekpath(phonon.primitive, 1, is_const_interval=True)
    points = []
    for i in range(len(bands)):
        if i == 0 or (bands[i - 1][1] != bands[i][0]).any():
            points.append(bands[i][0])
        points.append(bands[i][1])
    print("INFO: Labels for the special points in phonon dispersion:")
    for label, point in zip(labels, points):
        print(label, point)
