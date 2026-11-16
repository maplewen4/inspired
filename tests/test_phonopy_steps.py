"""The phonopy API steps must reproduce the phonopy 2.28 command line used before."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from inspired.gui import phonopy_steps
from inspired.gui.phonopy_steps import PhonopyConfError, read_mesh_conf

CU = Path(__file__).parents[1] / "examples" / "Cu"
# A coarser mesh than the example keeps the comparison quick.
MESH_CONF = "DIM = 4 4 4\nFC_SYMMETRY = .TRUE.\nMP = 4 4 4\nGAMMA_CENTER = .TRUE.\nEIGENVECTORS = .TRUE.\n"
PATH = [[0.95, -0.5, -0.5], [0.95, 0.0, 0.0], [0.95, 0.5, 0.5]]


@pytest.fixture
def cu(tmp_path, monkeypatch):
    """A working directory with the Cu DFT example, as INSPIRED prepares it."""
    for name in ("FORCE_SETS", "POSCAR-unitcell"):
        shutil.copy(CU / name, tmp_path)
    (tmp_path / "mesh.conf").write_text(MESH_CONF)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def phonons(path, key):
    """Frequencies and eigenvectors from a phonopy YAML file."""
    data = yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)
    points = data[key]
    frequencies = np.array([[band["frequency"] for band in point["band"]] for point in points])
    vectors = np.array([[np.array(band["eigenvector"])[..., 0] for band in point["band"]] for point in points])
    return frequencies, np.abs(vectors)  # eigenvector phases are arbitrary for degenerate modes


def phonopy_cli(cu, conf, output):
    """Run the command line the pipeline used before, keeping its output for comparison."""
    (cu / "cli.conf").write_text(conf)
    subprocess.run(["phonopy", "-c", "POSCAR-unitcell", "cli.conf"], check=True, capture_output=True)
    return Path(cu / output).rename(cu / f"cli-{output}")


def assert_same_phonons(api_file, cli_file, key):
    """Compare the two outputs, allowing for eigenvector mixing within degenerate modes."""
    api_frequencies, api_vectors = phonons(api_file, key)
    cli_frequencies, cli_vectors = phonons(cli_file, key)
    np.testing.assert_allclose(api_frequencies, cli_frequencies, atol=1e-6)
    nondegenerate = np.all(np.abs(np.diff(cli_frequencies, axis=1)) > 1e-4, axis=0)
    columns = np.concatenate([[nondegenerate[0]], nondegenerate]) & np.concatenate([nondegenerate, [nondegenerate[-1]]])
    np.testing.assert_allclose(api_vectors[:, columns], cli_vectors[:, columns], atol=1e-6)


def test_mesh_step_matches_phonopy_command_line(cu):
    """mesh.yaml written through the API equals the CLI's for the same mesh.conf."""
    cli = phonopy_cli(cu, MESH_CONF, "mesh.yaml")
    conf = read_mesh_conf()
    phonopy_steps.write_mesh_yaml(phonopy_steps.mesh_phonon(conf), conf)
    assert_same_phonons("mesh.yaml", cli, "phonon")


def test_band_step_matches_phonopy_command_line(cu):
    """band.yaml for a single-crystal cut equals the CLI's BAND mode (with FC_SYMMETRY spelled correctly)."""
    band = " ".join(str(value) for point in PATH for value in point)
    cli = phonopy_cli(
        cu, f"DIM = 4 4 4\nBAND = {band}\nBAND_POINTS = 21\nFC_SYMMETRY = .TRUE.\nEIGENVECTORS = .TRUE.\n", "band.yaml"
    )
    phonopy_steps.write_band_yaml(phonopy_steps.single_crystal_phonon(read_mesh_conf()), PATH, 21)
    assert_same_phonons("band.yaml", cli, "phonon")


def test_qpoints_step_matches_phonopy_command_line(cu):
    """qpoints.yaml equals the CLI's QPOINTS mode."""
    qpoints = [[0.1, 0.2, 0.3], [0.5, 0.0, 0.5], [1.0, 0.25, 0.25]]
    (cu / "QPOINTS").write_text(f"{len(qpoints)}\n" + "".join(" ".join(map(str, q)) + "\n" for q in qpoints))
    cli = phonopy_cli(
        cu, "DIM = 4 4 4\nQPOINTS = .TRUE.\nFC_SYMMETRY = .TRUE.\nEIGENVECTORS = .TRUE.\n", "qpoints.yaml"
    )
    phonopy_steps.write_qpoints_yaml(phonopy_steps.single_crystal_phonon(read_mesh_conf()), qpoints)
    assert_same_phonons("qpoints.yaml", cli, "phonon")


@pytest.mark.usefixtures("cu")
def test_single_crystal_force_constants_obey_acoustic_sum_rule():
    """Cuts use symmetrized force constants, so acoustic modes vanish at Gamma.

    The former band.conf misspelled FC_SYMMETRY, leaving a spurious ~0.004 THz
    acoustic frequency that OCLIMAX's 1/omega weighting turned into a large
    zero-energy peak at Bragg positions.
    """
    phonon = phonopy_steps.single_crystal_phonon(read_mesh_conf())
    phonon.run_qpoints([[0, 0, 0]])
    assert np.abs(phonon.get_qpoints_dict()["frequencies"][0][:3]).max() < 1e-6


def test_mesh_conf_flags_and_nine_value_dim(tmp_path):
    """Phonopy's boolean spellings and a full supercell matrix are understood."""
    path = tmp_path / "mesh.conf"
    path.write_text(
        "DIM = 2 0 0 0 2 0 0 0 3\nMP = 6 6 4\nGAMMA_CENTER = T\nFC_SYMMETRY = .true.\nFORCE_CONSTANTS = READ\n"
    )
    conf = read_mesh_conf(path)
    assert conf.supercell_matrix.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert conf.mesh == [6, 6, 4]
    assert conf.gamma_center and conf.fc_symmetry and not conf.eigenvectors and not conf.nac
    assert conf.force_constants == "READ"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("DIM = 2 2 2\nMP = 4 4 4\nTMAX = 1000\n", "TMAX"),
        ("DIM = 2 2 2\nMP = 4 4 4\nPRIMITIVE_AXES = 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0\n", "unit cell as primitive"),
        ("MP = 4 4 4\n", "DIM"),
        ("DIM = 2 2 2\n", "sampling mesh"),
    ],
)
def test_mesh_conf_that_cannot_be_reproduced_is_rejected(tmp_path, content, message):
    """Settings the API steps do not apply are reported instead of silently ignored."""
    path = tmp_path / "mesh.conf"
    path.write_text(content)
    with pytest.raises(PhonopyConfError, match=message):
        read_mesh_conf(path)


BORN = "14.400\n2.0 0 0 0 2.0 0 0 0 2.0\n1.0 0 0 0 1.0 0 0 0 1.0\n"


@pytest.mark.parametrize(
    ("born", "nac_line", "expected"),
    [(True, "", True), (False, "", False), (True, "NAC = .FALSE.\n", False), (True, "NAC = .TRUE.\n", True)],
)
def test_born_file_enables_nac_unless_disabled(cu, born, nac_line, expected):
    """Powder and single-crystal runs use BORN automatically; NAC = .FALSE. still opts out."""
    if born:
        (cu / "BORN").write_text(BORN)
    (cu / "mesh.conf").write_text(MESH_CONF + nac_line)
    conf = read_mesh_conf()
    assert conf.use_nac is expected
    for loader in (phonopy_steps.mesh_phonon, phonopy_steps.single_crystal_phonon):
        assert (loader(conf).nac_params is not None) is expected
