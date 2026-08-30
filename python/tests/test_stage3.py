from pathlib import Path

import numpy as np

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
COORDINATES = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
WEIGHTS = np.array([1.0], dtype=np.float64)
REGIONS = np.array([[1, -1, -1]], dtype=np.int64)


def _nonempty(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def test_stage3_spinor_chain_and_mldump(tmp_path: Path) -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "spinor_hydrogen_checkpoint.toml")
    physics = mt.CheckpointPhysics(checkpoint)
    product = physics.spinor_product_input(
        FIXTURES / "spinor_hydrogen_input.toml", q=[0.0, 0.0, 0.0]
    )
    assert product.export_orbitals()["energies"].shape[0] == 1
    basis = product.export_basis(0)
    assert basis["pauli_rows"].shape[1] == 3
    assert basis["local_orbital_rows"].shape[1] == 6
    radials = product.export_radials()
    assert radials["radial_labels"].shape[1] == 4
    assert radials["p"].shape == radials["q"].shape

    product_slice = physics.spinor_q_slice(FIXTURES / "spinor_hydrogen_input.toml")
    mpb = mt.build_spinor_mpb(
        product,
        np.array([[0, 0, 0]], dtype=np.int64),
        product_l_max=1,
        product_g_max=1.5,
        overlap_tolerance=1.0e-10,
    )
    thc = mt.build_spinor_thc(
        product_slice,
        COORDINATES,
        WEIGHTS,
        REGIONS,
        rank=1,
        engine="qrcp",
    )
    coulomb = mt.build_spinor_coulomb(
        product_slice,
        thc,
        lexp=2,
        interpolation_pw_cutoff=1.5,
        interpolation_l_max=1,
        comparisons=[(0, mpb, 0)],
    )
    matrix = coulomb.export_matrix(0)
    assert matrix["q_umklapp_index"].dtype == np.int32
    assert matrix["matrix"].shape == (1, 1)
    assert matrix["spin"] is None

    path = tmp_path / "spinor.mldump.h5"
    mt.write_spinor_mldump(
        path,
        product_slice,
        thc,
        coulomb,
        producer_name="libmuffintin-python-test",
        producer_version="0.2.0",
        source_revision="stage3-fixture",
        site_species=["H"],
        site_labels=["H-1"],
    )
    _nonempty(path)


def test_stage3_scalar_writers_reuse_stage2_handles(tmp_path: Path) -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    physics = mt.CheckpointPhysics(checkpoint)
    product = physics.scalar_product_input(
        FIXTURES / "hydrogen_input.toml", q=[0.0, 0.0, 0.0]
    )
    product_slice = physics.scalar_q_slice(FIXTURES / "hydrogen_input.toml")
    mpb = mt.build_scalar_mpb(
        product,
        np.array([[0, 0, 0, 0]], dtype=np.int64),
        product_l_max=1,
        product_g_max=1.5,
        overlap_tolerance=1.0e-10,
    )
    thc = mt.build_scalar_thc(
        product_slice,
        COORDINATES,
        WEIGHTS,
        REGIONS,
        spin=0,
        rank=1,
        engine="qrcp",
    )
    coulomb = mt.build_scalar_coulomb(
        product_slice,
        thc,
        lexp=2,
        interpolation_pw_cutoff=1.5,
        interpolation_l_max=1,
        comparisons=[(0, mpb, 0)],
    )

    mldump = tmp_path / "scalar.mldump.h5"
    mt.write_scalar_mldump(
        mldump,
        product_slice,
        thc,
        coulomb,
        producer_name="libmuffintin-python-test",
        producer_version="0.2.0",
        source_revision="stage3-fixture",
        site_species=["H"],
        site_labels=["H-1"],
    )
    _nonempty(mldump)

    coqui = tmp_path / "scalar.coqui.h5"
    mt.write_scalar_coqui_cholesky(
        coqui, product_slice, thc, coulomb, tolerance=1.0e-12
    )
    _nonempty(coqui)
