from pathlib import Path

from src.demo.p2.recording import (
    autodex_session_relative,
    resolve_signal_generator_params,
)
from src.demo.p2.run_auto import P2_LOCATIONS, P2_OUTCOMES, _make_episode_dir


def test_p2_operator_outcomes_are_cumulative():
    assert P2_OUTCOMES["f"] == {
        "name": "fail", "G": False, "P": False, "C": False,
    }
    assert P2_OUTCOMES["g"]["G"] is True
    assert P2_OUTCOMES["g"]["P"] is False
    assert P2_OUTCOMES["p"]["G"] is True
    assert P2_OUTCOMES["p"]["P"] is True
    assert P2_OUTCOMES["p"]["C"] is False
    assert P2_OUTCOMES["c"]["G"] is True
    assert P2_OUTCOMES["c"]["P"] is True
    assert P2_OUTCOMES["c"]["C"] is True
    assert P2_OUTCOMES["a"] == {
        "name": "aborted", "G": None, "P": None, "C": None,
        "scored": False,
    }
    assert P2_LOCATIONS["0"]["name"] == "upper_right"
    assert P2_LOCATIONS["2"]["name"] == "lower_left"


def test_p2_episode_and_capture_paths_match_existing_demo_hierarchy(tmp_path: Path):
    root = tmp_path / "AutoDex"
    episode = _make_episode_dir(root, "apple", "v8_demo")
    assert episode.parent == root / "experiment" / "v8_demo" / "inspire" / "apple"
    assert autodex_session_relative(root, episode) == (
        Path("AutoDex") / "experiment" / "v8_demo" / "inspire" / "apple" / episode.name
    )


def test_recording_uses_only_a_single_discovered_usbtmc_node(tmp_path: Path):
    only = tmp_path / "usbtmc5"
    only.touch()
    params, note = resolve_signal_generator_params(
        {"addr": "/dev/usbtmc0", "baud": 9600}, device_root=tmp_path)
    assert params["addr"] == str(only)
    assert "usbtmc5" in note

    (tmp_path / "usbtmc6").touch()
    params, note = resolve_signal_generator_params(
        {"addr": "/dev/usbtmc0"}, device_root=tmp_path)
    assert params["addr"] == "/dev/usbtmc0"
    assert note is None
