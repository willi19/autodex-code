from pathlib import Path

import pytest

from src.demo.p2.protocol import (
    P2_COLLECTION_STATE,
    P2_CLASS_ROUTES,
    P2_GRASP_LOOKUP_FIELDS,
    P2_OBJECTS,
    basket_for,
    candidate_lookup_path,
    collection_result_root,
    coverage_json_path,
    foundpose_repre_path,
    get_p2_object,
)
from autodex.utils.path import get_candidate_path, project_dir
from src.demo.p2.run_demo import _inference_argv, _parse_p2_args
from src.demo.p2.run_auto import _parse_args as _parse_p2_auto_args


def test_p2_semantic_routes_are_fixed():
    assert [item.name for item in P2_OBJECTS] == [
        "apple", "banana", "pringles", "spam_can",
    ]
    assert basket_for("apple") == "left"
    assert basket_for("banana") == "left"
    assert basket_for("pringles") == "right"
    assert basket_for("spam_can") == "right"
    assert get_p2_object("pringles").semantic_class == "nonfruit"
    assert P2_CLASS_ROUTES == {
        "FRUIT": {"semantic_class": "fruit", "basket": "left", "bearing_deg": 50.0},
        "NON_FRUIT": {"semantic_class": "nonfruit", "basket": "right", "bearing_deg": -30.0},
    }


def test_p2_grasp_lookup_is_arm_independent():
    assert P2_COLLECTION_STATE == "collecting_v8_inspire"
    assert P2_GRASP_LOOKUP_FIELDS == ("object", "hand", "grasp_version")
    assert str(candidate_lookup_path("apple", candidate_root="/candidates")) == (
        "/candidates/inspire/v8/apple"
    )


def test_p2_paths_match_the_existing_v8_hierarchy():
    assert candidate_lookup_path("apple") == Path(get_candidate_path("inspire")) / "v8" / "apple"
    assert collection_result_root("apple") == (
        Path(project_dir) / "experiment" / "v8" / "inspire" / "apple"
    )
    assert str(collection_result_root("apple", project_root="/AutoDex")) == (
        "/AutoDex/experiment/v8/inspire/apple"
    )
    assert str(coverage_json_path("apple", project_root="/AutoDex")) == (
        "/AutoDex/experiment/v8/coverage/cov_v8_cand_apple.json"
    )
    assert str(foundpose_repre_path("apple", foundpose_root="/assets")) == (
        "/assets/apple/object_repre/v1/apple/1/repre.pth"
    )


def test_p2_rejects_both_joint0_bearing_flag_spellings():
    class _P2Args:
        obj = "apple"

    for argv in (
        ["--joint0-drop-bearing-deg", "-30"],
        ["--joint0-drop-bearing-deg=-30"],
    ):
        with pytest.raises(SystemExit, match="sets the J0 release bearing"):
            _inference_argv(_P2Args(), argv)


def test_p2_accepts_generic_asset_names_but_only_scores_benchmark_truth():
    direct, _ = _parse_p2_args(["--obj", "french_mustard"])
    continuous, _ = _parse_p2_auto_args(["--obj", "pepsi_light"])
    assert direct.obj == "french_mustard"
    assert continuous.obj == "pepsi_light"
