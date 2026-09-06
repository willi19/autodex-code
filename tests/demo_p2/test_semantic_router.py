import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.demo.p2.semantic_crops import make_semantic_crop
from src.demo.p2.semantic_router import (
    P2_PROMPT,
    P2SemanticRouter,
    SemanticCrop,
    _Session,
)


class _FakeClassifier:
    model_id = "test/fake-qwen"

    def __init__(self, label: str):
        self.label = label
        self.calls: list[list[np.ndarray]] = []

    def preload(self):
        pass

    def classify(self, images_rgb):
        assert len(images_rgb) == 3
        assert all(image.shape == (448, 448, 3) for image in images_rgb)
        self.calls.append(images_rgb)
        return self.label, self.label, 0.012


def test_p2_prompt_is_the_requested_binary_three_view_prompt():
    assert P2_PROMPT == (
        "The same foreground object is shown in three views.\n"
        "Classify it as either a fruit or a non-fruit.\n\n"
        "Return exactly one label:\nFRUIT\nor\nNON_FRUIT"
    )


def test_semantic_crop_has_one_edge_rule_and_neutral_background():
    image = np.full((120, 160, 3), 15, dtype=np.uint8)
    image[35:65, 55:95] = (200, 40, 10)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[35:65, 55:95] = True

    built = make_semantic_crop(image, mask)
    assert built is not None
    crop, info = built
    assert crop.shape == (448, 448, 3)
    assert info.bbox_xywh == (55, 35, 40, 30)
    assert info.border_margin_px == 35
    # Foreground remains real RGB while the only context is the protocol's
    # neutral gray, not checkerboard/table pixels.
    assert np.any(np.all(crop == (200, 40, 10), axis=-1))
    assert np.any(np.all(crop == (127, 127, 127), axis=-1))

    # No quality heuristic: one foreground pixel inside the 16px edge strip is
    # enough to reject the entire camera view.
    edge = mask.copy()
    edge[10, 80] = True
    assert make_semantic_crop(image, edge) is None


@pytest.mark.parametrize(
    ("object_name", "label", "basket", "bearing_deg"),
    [
        ("apple", "FRUIT", "left", 50.0),
        ("banana", "FRUIT", "left", 50.0),
        ("pringles", "NON_FRUIT", "right", -30.0),
        ("spam_can", "NON_FRUIT", "right", -30.0),
    ],
)
def test_first_three_distinct_pc_crops_are_classified_and_recorded(
    tmp_path: Path, object_name: str, label: str, basket: str, bearing_deg: float,
):
    """Exercise P2's arrival-order rule without sockets or a real VLM."""
    router = object.__new__(P2SemanticRouter)
    router.serial_to_pc = {
        "cam1": "capture1", "cam2": "capture2", "cam3": "capture3",
        "cam5": "capture5",
    }
    router.classifier = _FakeClassifier(label)
    router.evaluation_object = object_name
    router._sessions = {}
    router._lock = threading.RLock()

    request_id = 7
    out = tmp_path / "semantic"
    P2SemanticRouter.begin(router, request_id, out)
    source = np.full((448, 448, 3), (1, 2, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    assert ok
    metadata = {"req_id": request_id, "encoding": "jpeg"}

    # No PC/camera scoring: the first crop from each distinct capture PC wins.
    for serial in ("cam2", "cam1", "cam3", "cam5"):
        P2SemanticRouter._accept_message(router, {**metadata, "serial": serial},
                                          encoded.tobytes())
    session = router._sessions[request_id]
    assert session.done.wait(timeout=3.0)
    result = P2SemanticRouter.wait(router, request_id, timeout_s=0.1)

    assert result["status"] == "ok"
    assert result["prediction"] == label
    assert result["basket"] == basket
    assert result["bearing_deg"] == bearing_deg
    assert [entry["pc"] for entry in result["selected_crops"]] == [
        "capture2", "capture1", "capture3",
    ]
    assert result["semantic_evaluation"] == {
        "object": object_name,
        "expected_class": "fruit" if label == "FRUIT" else "nonfruit",
        "C": True,
    }
    # Each P2 object receives one VLM call with its three images together;
    # routing never invokes the classifier per image or votes afterwards.
    assert len(router.classifier.calls) == 1
    assert len(router.classifier.calls[0]) == 3
    assert (out / "crop_1_capture2_cam2.jpg").is_file()
    assert json.loads((out / "semantic_result.json").read_text())["prediction"] == label
