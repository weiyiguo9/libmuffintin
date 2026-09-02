from pathlib import Path

import numpy as np
import pytest

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
V2_FIXTURE = FIXTURES / "mldump_exchange_v2.h5"
V1_FIXTURE = FIXTURES / "mldump_spinor_v1.h5"


def test_mldump_v2_exchange_export_has_frozen_schema_and_shapes() -> None:
    assert "read_mldump_v2" in mt.__all__
    export = mt.read_mldump_v2(V2_FIXTURE)

    assert set(export) == {
        "schema",
        "version",
        "kind",
        "producer_name",
        "producer_version",
        "source_revision",
        "feature_representation",
        "exchange_vv_hartree",
        "exchange_cv_hartree",
        "exchange_cc_hartree",
        "exchange_total_hartree",
        "exchange_total_relation",
        "cross_trace_average_hartree",
        "cross_trace_mismatch_hartree",
        "sectors",
        "provenance",
    }
    assert export["schema"] == "libmuffintin.pyexport"
    assert export["version"] == 2
    assert export["kind"] == "exchange"
    assert export["feature_representation"] == "spinor_full_first_variation"
    assert export["exchange_total_relation"] == (
        "exchange_total=exchange_vv+exchange_cv+exchange_cc;"
        "exchange_cv=(trace_cv+trace_vc)/2"
    )
    assert all(
        isinstance(export[key], float)
        for key in (
            "exchange_vv_hartree",
            "exchange_cv_hartree",
            "exchange_cc_hartree",
            "exchange_total_hartree",
            "cross_trace_average_hartree",
            "cross_trace_mismatch_hartree",
        )
    )

    sectors = export["sectors"]
    assert set(sectors) == {"vv", "cv", "vc", "cc"}
    spaces = {
        "vv": ("valence", "valence"),
        "cv": ("core", "valence"),
        "vc": ("valence", "core"),
        "cc": ("core", "core"),
    }
    for name, expected_spaces in spaces.items():
        sector = sectors[name]
        assert set(sector) == {
            "layout",
            "trace_hartree",
            "maximum_antihermitian_residual",
            "fit_frobenius",
            "fit_column_max",
            "mpb_quadratic_maximum_absolute",
            "mpb_quadratic_maximum_relative",
            "mpb_quadratic_worst_absolute_q_index",
            "mpb_quadratic_worst_absolute_column",
            "mpb_quadratic_worst_relative_q_index",
            "mpb_quadratic_worst_relative_column",
        }
        layout = sector["layout"]
        assert set(layout) == {
            "occupied_space",
            "target_space",
            "n_k",
            "n_occupied",
            "n_target",
        }
        assert (layout["occupied_space"], layout["target_space"]) == expected_spaces
        assert all(
            isinstance(sector[key], float)
            for key in (
                "trace_hartree",
                "maximum_antihermitian_residual",
                "fit_frobenius",
                "fit_column_max",
                "mpb_quadratic_maximum_absolute",
                "mpb_quadratic_maximum_relative",
            )
        )
        assert all(
            isinstance(sector[key], int)
            for key in (
                "mpb_quadratic_worst_absolute_q_index",
                "mpb_quadratic_worst_absolute_column",
                "mpb_quadratic_worst_relative_q_index",
                "mpb_quadratic_worst_relative_column",
            )
        )

    provenance = export["provenance"]
    assert set(provenance) == {
        "source_frame",
        "backend",
        "gamma_policy",
        "product_l_max",
        "product_g_max_inv_bohr",
        "overlap_tolerance",
        "coulomb_lexp",
        "interpolation_l_max",
        "interpolation_pw_cutoff_inv_bohr",
        "selector_strategy",
        "selector_engine",
        "requested_rank_policy",
        "requested_rank_n_mu",
        "requested_rank_threshold",
        "requested_rank_n_max",
        "rank_scaling",
        "k_weights",
        "valence_occupations",
        "core_identity",
        "core_occupations",
    }
    scaling = provenance["rank_scaling"]
    assert set(scaling) == {
        "n_k",
        "n_valence",
        "n_core",
        "n_candidates",
        "effective_rank",
        "vv_columns_per_q",
        "cv_columns_per_q",
        "vc_columns_per_q",
        "cc_columns_per_q",
        "pooled_columns_per_q",
        "selector_rows",
    }
    assert provenance["source_frame"] == "relaxed_core_hf_final_rebuilt_frame"
    assert provenance["backend"] == "core_aware_thc_with_exact_mpb_oracle"
    assert provenance["gamma_policy"] in {"finite_body", "reject"}
    assert provenance["selector_strategy"] == "allq_l2"
    assert (scaling["n_k"], scaling["n_valence"], scaling["n_core"]) == (2, 2, 1)
    assert provenance["k_weights"].dtype == np.float64
    assert provenance["k_weights"].shape == (scaling["n_k"],)
    assert provenance["valence_occupations"].dtype == np.float64
    assert provenance["valence_occupations"].shape == (
        scaling["n_k"],
        scaling["n_valence"],
    )
    assert provenance["core_identity"].dtype == np.int64
    assert provenance["core_identity"].shape == (scaling["n_core"], 4)
    assert provenance["core_occupations"].dtype == np.float64
    assert provenance["core_occupations"].shape == (scaling["n_core"],)

    assert sectors["vv"]["layout"] == {
        "occupied_space": "valence",
        "target_space": "valence",
        "n_k": scaling["n_k"],
        "n_occupied": scaling["n_valence"],
        "n_target": scaling["n_valence"],
    }
    assert sectors["cv"]["layout"]["n_occupied"] == scaling["n_core"]
    assert sectors["cv"]["layout"]["n_target"] == scaling["n_valence"]
    assert sectors["vc"]["layout"]["n_occupied"] == scaling["n_valence"]
    assert sectors["vc"]["layout"]["n_target"] == scaling["n_core"]
    assert sectors["cc"]["layout"]["n_occupied"] == scaling["n_core"]
    assert sectors["cc"]["layout"]["n_target"] == scaling["n_core"]


def test_mldump_v2_reader_rejects_v1() -> None:
    with pytest.raises(ValueError):
        mt.read_mldump_v2(V1_FIXTURE)
