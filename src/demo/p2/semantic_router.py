"""P2-only VLM semantic route: three SAM3 crops -> FRUIT/NON_FRUIT.

The router is deliberately additive to AutoDex.  Capture PCs publish one
already-masked JPEG on a dedicated P2 port; existing mask and FoundPose pose
channels are neither changed nor consumed here.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.demo.p2.protocol import (
    P2_CLASS_ROUTES,
    P2_OBJECT_BY_NAME,
    P2_REQUIRED_SEMANTIC_CROPS,
)


logger = logging.getLogger(__name__)

P2_SEMANTIC_PORT = 5010
# Use the official instruct checkpoint and quantize it at load time. The
# published AWQ checkpoint fails to compile its lm_head on this machine's
# AutoAWQ/Triton stack; bitsandbytes NF4 is a supported 4-bit path here.
P2_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
P2_PROMPT = (
    "The same foreground object is shown in three views.\n"
    "Classify it as either a fruit or a non-fruit.\n\n"
    "Return exactly one label:\nFRUIT\nor\nNON_FRUIT"
)
_VALID_OUTPUT = re.compile(r"^(FRUIT|NON_FRUIT)$")


@dataclass(frozen=True)
class SemanticCrop:
    request_id: int
    pc: str
    serial: str
    rgb: np.ndarray
    metadata: dict[str, Any]
    arrived_monotonic: float


@dataclass
class _Session:
    output_dir: Path
    # Snapshot this only for post-hoc P2 semantic scoring.  It is never sent
    # to Qwen, and keeping it on the session lets a continuous runner begin a
    # later object even if a previous timeout's worker is still unwinding.
    evaluation_object: str | None = None
    selected_by_pc: dict[str, SemanticCrop] = field(default_factory=dict)
    started: bool = False
    result: dict[str, Any] | None = None
    thread: threading.Thread | None = None
    done: threading.Event = field(default_factory=threading.Event)


class QwenFruitClassifier:
    """Lazy wrapper around the pinned 4-bit Qwen2.5-VL model."""

    def __init__(self, model_id: str = P2_QWEN_MODEL, device: str = "cuda:0"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def preload(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        t0 = time.perf_counter()
        # This remains a general 4-bit VLM, rather than a four-object
        # classifier. NF4 is explicit so it fits the robot GPU without the
        # incompatible AWQ Triton kernel.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=448 * 448,
            max_pixels=448 * 448,
            # Preserve the processor stored with this checkpoint instead of
            # silently changing image pre-processing when Transformers flips
            # its future default to a fast implementation.
            use_fast=False,
        )
        self.model.eval()
        logger.info("[p2-vlm] loaded %s in %.1fs", self.model_id,
                    time.perf_counter() - t0)

    def classify(self, images_rgb: Sequence[np.ndarray]) -> tuple[str, str, float]:
        """Return exact label, raw text, and inference time for exactly 3 crops."""
        if len(images_rgb) != P2_REQUIRED_SEMANTIC_CROPS:
            raise ValueError("P2 needs exactly "
                             f"{P2_REQUIRED_SEMANTIC_CROPS} crops, received {len(images_rgb)}")
        self.preload()
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        images = [Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB")
                  for image in images_rgb]
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image, "resized_height": 448,
             "resized_width": 448}
            for image in images
        ]
        content.append({"type": "text", "text": P2_PROMPT})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=4,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        match = _VALID_OUTPUT.fullmatch(raw.upper())
        if match is None:
            raise RuntimeError(f"P2 VLM returned non-protocol output: {raw!r}")
        return match.group(1), raw, time.perf_counter() - t0


class P2SemanticRouter:
    """Receive the first crop from three PCs and classify it asynchronously."""

    def __init__(
        self,
        *,
        capture_ips: Sequence[str],
        pc_serials: Mapping[str, Sequence[str]],
        port: int = P2_SEMANTIC_PORT,
        model_id: str = P2_QWEN_MODEL,
        classifier: QwenFruitClassifier | None = None,
        timeout_s: float = 20.0,
        evaluation_object: str | None = None,
    ):
        if len(capture_ips) != len(pc_serials):
            raise ValueError("capture_ips must contain one address for every capture PC")
        self.port = int(port)
        if self.port <= 0 or self.port > 65535:
            raise ValueError("semantic port must be in 1..65535")
        if timeout_s <= 0:
            raise ValueError("semantic timeout must be positive")
        self.timeout_s = float(timeout_s)
        # Evaluation metadata is never fed to Qwen.  It exists only so the P2
        # episode can separately score semantic correctness after the route
        # when this happens to be one of the four benchmark objects.
        self.evaluation_object = evaluation_object
        self.serial_to_pc = {
            str(serial): pc for pc, serials in pc_serials.items() for serial in serials
        }
        self.classifier = classifier or QwenFruitClassifier(model_id=model_id)
        self._sessions: dict[int, _Session] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()

        import zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        for ip in capture_ips:
            self._sock.connect(f"tcp://{ip}:{self.port}")
        self._thread = threading.Thread(target=self._receive_loop,
                                        name="p2_semantic_crop", daemon=True)
        self._thread.start()
        # Give every SUB socket time to subscribe before the caller dispatches
        # the first capture.  This matches InitOrchestrator's own 0.3s settle.
        time.sleep(0.3)

    def preload(self) -> None:
        self.classifier.preload()

    def set_evaluation_object(self, name: str) -> None:
        """Set the object name used for optional P2 semantic scoring.

        The name is deliberately never part of the Qwen prompt or image input.
        A continuous P2 session keeps the loaded VLM and crop subscriber alive
        while the operator changes physical objects between episodes.  The four
        benchmark names receive automatic ground-truth C; any other valid
        asset name is explicitly recorded as unscored rather than rejected.
        """
        with self._lock:
            self.evaluation_object = name

    def begin(self, request_id: int, output_dir: Path) -> None:
        with self._lock:
            if request_id in self._sessions:
                raise RuntimeError(f"P2 semantic request {request_id} is already armed")
            output_dir.mkdir(parents=True, exist_ok=False)
            self._sessions[request_id] = _Session(
                output_dir=output_dir,
                evaluation_object=self.evaluation_object,
            )

    @staticmethod
    def run_info() -> dict[str, Any]:
        """The only P2 extension to the otherwise unchanged daemon run command."""
        return {"p2_semantic_enabled": True}

    def wait(self, request_id: int, timeout_s: float) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(request_id)
        if session is None:
            raise RuntimeError(f"P2 semantic request {request_id} was never armed")
        if not session.done.wait(timeout=max(0.0, float(timeout_s))):
            with self._lock:
                received = sorted(session.selected_by_pc)
            return {
                "status": "semantic_timeout",
                "request_id": request_id,
                "received_pcs": received,
                "required_pcs": 3,
            }
        assert session.result is not None
        return dict(session.result)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.setsockopt(17, 0)  # ZMQ_LINGER; keep import-free teardown.
            self._sock.close()
        except Exception:
            pass
        self._thread.join(timeout=3.0)

    def _receive_loop(self) -> None:
        from autodex.perception.init_orchestrator import _parse_multipart

        while not self._stop.is_set():
            try:
                if not self._sock.poll(timeout=100):
                    continue
                parts = self._sock.recv_multipart(flags=1)  # NOBLOCK
                metadata, blobs = _parse_multipart(parts)
                if isinstance(metadata, list):
                    entries = zip(metadata, blobs)
                elif isinstance(metadata, dict) and "items" in metadata:
                    entries = zip(metadata["items"], blobs)
                elif isinstance(metadata, dict):
                    entries = [(metadata, blobs[0] if blobs else b"")]
                else:
                    continue
                for item, blob in entries:
                    self._accept_message(item, blob)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.warning("[p2-vlm] crop subscriber: %s", exc)

    def _accept_message(self, item: Mapping[str, Any], blob: bytes) -> None:
        try:
            request_id = int(item["req_id"])
            serial = str(item["serial"])
        except (KeyError, TypeError, ValueError):
            return
        pc = self.serial_to_pc.get(serial)
        if pc is None:
            logger.warning("[p2-vlm] ignoring unassigned camera serial %s", serial)
            return
        encoded = np.frombuffer(blob, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning("[p2-vlm] unreadable crop from %s", serial)
            return
        crop = SemanticCrop(
            request_id=request_id, pc=pc, serial=serial,
            rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            metadata=dict(item), arrived_monotonic=time.monotonic(),
        )
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None or session.started or pc in session.selected_by_pc:
                return
            session.selected_by_pc[pc] = crop
            if len(session.selected_by_pc) != P2_REQUIRED_SEMANTIC_CROPS:
                return
            # Dict insertion order is the receipt order: these are exactly the
            # first three different capture PCs, not a later quality ranking.
            selected = tuple(session.selected_by_pc.values())
            session.started = True
            session.thread = threading.Thread(
                target=self._classify_session, args=(request_id, selected),
                name=f"p2_vlm_{request_id}", daemon=True,
            )
            session.thread.start()

    def _classify_session(self, request_id: int, selected: Sequence[SemanticCrop]) -> None:
        t0 = time.perf_counter()
        with self._lock:
            session = self._sessions.get(request_id)
        if session is None:
            return
        try:
            records = []
            for index, crop in enumerate(selected, start=1):
                filename = f"crop_{index}_{crop.pc}_{crop.serial}.jpg"
                output = session.output_dir / filename
                ok = cv2.imwrite(str(output), cv2.cvtColor(crop.rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise OSError(f"failed to write {output}")
                records.append({
                    "pc": crop.pc, "serial": crop.serial,
                    "crop_path": str(output), "crop_metadata": crop.metadata,
                })
            label, raw, model_s = self.classifier.classify([crop.rgb for crop in selected])
            route = P2_CLASS_ROUTES[label]
            outcome = {
                "status": "ok", "request_id": request_id,
                "model_id": self.classifier.model_id,
                "prompt": P2_PROMPT,
                "raw_output": raw,
                "prediction": label,
                **route,
                "selected_crops": records,
                "model_inference_s": round(model_s, 3),
                "semantic_total_s": round(time.perf_counter() - t0, 3),
            }
            if session.evaluation_object in P2_OBJECT_BY_NAME:
                expected = P2_OBJECT_BY_NAME[session.evaluation_object]
                outcome["semantic_evaluation"] = {
                    "object": expected.name,
                    "expected_class": expected.semantic_class,
                    "C": route["semantic_class"] == expected.semantic_class,
                }
            elif session.evaluation_object is not None:
                # Generic P2 operation intentionally supports objects outside
                # the four-item benchmark.  The VLM still routes them, but no
                # protocol ground truth is available for automatic C scoring.
                outcome["semantic_evaluation"] = {
                    "object": session.evaluation_object,
                    "expected_class": None,
                    "C": None,
                    "scored": False,
                }
            # ``print`` rather than only logging: this is the operator-facing
            # decision that determines the release direction, and must remain
            # visible even when application logging is filtered.
            print("[p2-vlm] "
                  f"prediction={label}  raw={raw!r}  "
                  f"basket={route['basket']}  "
                  f"release_bearing={route['bearing_deg']:+.1f} deg")
            with open(session.output_dir / "semantic_result.json", "w") as handle:
                json.dump(outcome, handle, indent=2)
        except Exception as exc:
            outcome = {
                "status": "semantic_vlm_error", "request_id": request_id,
                "model_id": self.classifier.model_id,
                "exception": repr(exc),
                "semantic_total_s": round(time.perf_counter() - t0, 3),
            }
            print(f"[p2-vlm] classification failed: {exc!r}")
            try:
                with open(session.output_dir / "semantic_result.json", "w") as handle:
                    json.dump(outcome, handle, indent=2)
            except OSError:
                pass
        finally:
            with self._lock:
                current = self._sessions.get(request_id)
                if current is not None:
                    current.result = outcome
                    current.done.set()


def route_for_prediction(label: str) -> dict[str, Any]:
    """Return the immutable P2 route for a protocol label."""
    try:
        return dict(P2_CLASS_ROUTES[label])
    except KeyError as exc:
        raise ValueError(f"unknown P2 semantic label {label!r}") from exc


def validate_p2_object(name: str) -> None:
    if name not in P2_OBJECT_BY_NAME:
        raise ValueError(f"{name!r} is not a configured P2 object")
