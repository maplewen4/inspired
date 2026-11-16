"""The dispersion plot model must be the same physics on the standard primitive cell."""

from pathlib import Path

import matplotlib
import numpy as np
import phonopy
import pytest
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from inspired.phonon_plot import plot_band_structure_and_dos, standard_primitive_phonon

matplotlib.use("Agg")
CU = Path(__file__).parents[1] / "examples" / "Cu"


def load_cu(**kwargs):
    """Conventional 4-atom fcc Cu with DFT forces, as stored by INSPIRED."""
    return phonopy.load(
        supercell_matrix=[4, 4, 4],
        primitive_matrix=np.eye(3),
        unitcell_filename=str(CU / "POSCAR-unitcell"),
        force_sets_filename=str(CU / "FORCE_SETS"),
        **kwargs,
    )


def frequencies(phonon, qpoints):
    """Return sorted frequencies at fractional q-points of *phonon*'s primitive cell."""
    phonon.run_qpoints(qpoints)
    return np.sort(phonon.get_qpoints_dict()["frequencies"], axis=1)


def to_cell_basis(qpoints, source, target):
    """Convert fractional q-points between the reciprocal bases of two cells."""
    cartesian = np.asarray(qpoints) @ np.linalg.inv(source.primitive.cell).T
    return cartesian @ target.primitive.cell.T


QPOINTS = [[0.1, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.1, 0.4], [0.5, 0.25, 0.75]]


def test_conventional_cell_is_unfolded_without_changing_frequencies():
    """Each primitive-cell branch also appears in the folded spectrum at the same physical q."""
    conventional = load_cu()
    primitive = standard_primitive_phonon(conventional)
    assert len(conventional.primitive) == 4
    assert len(primitive.primitive) == 1
    unfolded = frequencies(primitive, QPOINTS)
    folded = frequencies(conventional, to_cell_basis(QPOINTS, primitive, conventional))
    assert unfolded.shape[1] == 3
    for branches, candidates in zip(unfolded, folded):
        for frequency in branches:
            assert np.min(np.abs(candidates - frequency)) < 1e-6


def test_compact_force_constants_give_identical_plot_model():
    """Compact force constants (as in some DFT databases) are expanded correctly."""
    full = standard_primitive_phonon(load_cu())
    compact = standard_primitive_phonon(load_cu(is_compact_fc=True))
    np.testing.assert_allclose(frequencies(compact, QPOINTS), frequencies(full, QPOINTS), atol=1e-8)


def test_primitive_input_is_returned_unchanged():
    """No copy is made when the input is already the standard primitive cell."""
    primitive = standard_primitive_phonon(load_cu())
    assert standard_primitive_phonon(primitive) is primitive


def test_born_charges_follow_their_atoms():
    """Born charges are re-indexed for the primitive cell, not truncated."""
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:], positions=atoms.positions)
    phonon = Phonopy(unitcell, supercell_matrix=[2, 2, 2], primitive_matrix=np.eye(3))
    natoms = len(phonon.supercell)
    phonon.force_constants = np.zeros((natoms, natoms, 3, 3))
    charge = {"Na": 1.1, "Cl": -1.1}
    born = np.array([charge[symbol] * np.eye(3) for symbol in phonon.primitive.symbols])
    phonon.nac_params = {"born": born, "dielectric": 2.4 * np.eye(3), "factor": 14.4}

    primitive = standard_primitive_phonon(phonon)
    assert len(primitive.primitive) == 2
    expected = [charge[symbol] * np.eye(3) for symbol in primitive.primitive.symbols]
    np.testing.assert_allclose(primitive.nac_params["born"], expected)


def test_plot_has_high_symmetry_labels(capsys):
    """The plot uses the labelled seekpath path of the standard primitive cell."""
    plot_band_structure_and_dos(load_cu())
    output = capsys.readouterr().out
    assert "standard primitive cell (1 atoms)" in output
    assert "Labels for the special points" in output
    matplotlib.pyplot.close("all")


@pytest.mark.parametrize("module", ["inspired.gui.dft_worker", "inspired.gui.mlff_worker"])
def test_workers_share_the_plot(module):
    """DFT and MLFF dispersions are drawn by the same function."""
    imported = __import__(module, fromlist=["plot_band_structure_and_dos"])
    assert imported.plot_band_structure_and_dos is plot_band_structure_and_dos
