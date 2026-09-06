#!/usr/bin/env python3
"""Run P2: normal AutoDex inference plus a generic FRUIT/NON_FRUIT route.

The object name is still required by FoundPose and the v8 Inspire grasp pool.
It is *not* given to Qwen: the semantic prompt is always the same binary
fruit/non-fruit question.  The selected class changes only the existing
J0-only release bearing:

    FRUIT     -> +50 deg (left basket)
    NON_FRUIT -> -30 deg (right basket)

Example (after starting semantic-capable daemons):

    bash scripts/init_daemons.sh start --p2-semantic
    python src/demo/p2/run_demo.py --obj apple --arm franka --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.p2.protocol import P2_OBJECT_BY_NAME, P2_PROTOCOL_ID
from src.demo.p2.semantic_router import P2_QWEN_MODEL, P2SemanticRouter


def _parse_p2_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--obj", help="object name with FoundPose/v8 assets")
    parser.add_argument("--semantic-port", type=int, default=5010,
                        help="P2 crop PUB port (must match init_daemon --port-semantic)")
    parser.add_argument("--semantic-timeout-s", type=float, default=20.0,
                        help="wait after FoundPose collection for the first three-PC Qwen route")
    parser.add_argument("--vlm-model", default=P2_QWEN_MODEL,
                        help="official Qwen2.5-VL-3B-Instruct, loaded in NF4 4-bit mode")
    parser.add_argument("--no-video", action="store_true",
                        help="skip per-capture-PC execution AVI recording")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="execution AVI FPS on each capture PC (default: 30)")
    parser.add_argument("-h", "--help", action="store_true")
    p2_args, inference_argv = parser.parse_known_args(argv)
    if p2_args.help:
        parser.print_help()
        print("\nAll remaining options are those of src/demo/inference/run_demo.py.")
        raise SystemExit(0)
    if p2_args.obj is None:
        parser.error("--obj is required")
    if p2_args.semantic_timeout_s <= 0:
        parser.error("--semantic-timeout-s must be positive")
    if not 1 <= p2_args.semantic_port <= 65535:
        parser.error("--semantic-port must be in 1..65535")
    if p2_args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    return p2_args, inference_argv


def _inference_argv(p2_args: argparse.Namespace,
                   remaining: list[str]) -> list[str]:
    """Validate the unchanged inference runner's P2-compatible modes."""
    from src.demo.inference import run_demo as inference

    forwarded = ["--obj", p2_args.obj, *remaining]
    try:
        parsed = inference.parse_args(forwarded)
    except SystemExit:
        raise
    if parsed.perception != "foundpose":
        raise SystemExit("P2 requires --perception foundpose: SAM3 crops come from "
                         "the existing init_daemon")
    if parsed.transfer_mode != "joint0-arc":
        raise SystemExit("P2 fixes --transfer-mode joint0-arc; Cartesian placement "
                         "would change the protocol's only motion variable")
    if parsed.drop_target != "fixed-box":
        raise SystemExit("P2 fixes --drop-target fixed-box; marker capture is not part "
                         "of semantic routing")
    if parsed.lift_height is not None and abs(parsed.lift_height - 0.15) > 1e-9:
        raise SystemExit("P2 fixes the lift to 0.15 m")
    if any(token == "--joint0-drop-bearing-deg"
           or token.startswith("--joint0-drop-bearing-deg=")
           for token in remaining):
        raise SystemExit("P2 sets the J0 release bearing only from the generic "
                         "Qwen FRUIT/NON_FRUIT result; do not pass "
                         "--joint0-drop-bearing-deg")
    # The inference default is already exactly these values.  Do not inject or
    # override options: its normal parser and all existing planning behaviour
    # stay the source of truth.
    return forwarded


def main(argv: list[str] | None = None) -> None:
    p2_args, remaining = _parse_p2_args(argv)
    forwarded = _inference_argv(p2_args, remaining)
    from src.demo.inference import run_demo as inference

    if p2_args.obj in P2_OBJECT_BY_NAME:
        print(f"[p2] protocol={P2_PROTOCOL_ID} benchmark object={p2_args.obj}; "
              "Qwen sees only three neutral-background SAM3 crops, never this name")
    else:
        print(f"[p2] protocol={P2_PROTOCOL_ID} generic object={p2_args.obj}; "
              "Qwen sees only three neutral-background SAM3 crops, never this name; "
              "automatic semantic ground-truth C is unavailable")
    print("[p2] FRUIT -> left/+50.0 deg; NON_FRUIT -> right/-30.0 deg")

    recording_runtime = None

    def make_router(*, args, capture_ips, pc_serials, run_dir):
        # ``args`` and ``run_dir`` are deliberately unused: P2's semantic
        # classification has no object-specific prompt, crop-ranking, or
        # output-location override.  The parent inference runner owns episode
        # output under the normal v8_demo hierarchy.
        del args, run_dir
        return P2SemanticRouter(
            capture_ips=capture_ips, pc_serials=pc_serials,
            port=p2_args.semantic_port, model_id=p2_args.vlm_model,
            timeout_s=p2_args.semantic_timeout_s,
            evaluation_object=p2_args.obj,
        )

    def make_execution_recorder(*, args, rcc, executor, run_dir, pc_list,
                                pc_serials, project_root):
        """Build the P2-only video recorder after the robot is connected."""
        nonlocal recording_runtime
        if p2_args.no_video:
            return None
        from src.demo.p2.recording import P2RecordingRuntime

        if recording_runtime is None:
            serials = [serial for pc in pc_list for serial in pc_serials[pc]]
            recording_runtime = P2RecordingRuntime(
                rcc=rcc, pc_list=pc_list, serials=serials,
                video_fps=p2_args.video_fps,
            )
        return recording_runtime.for_episode(
            run_dir=run_dir, project_root=project_root, executor=executor)

    inference.main(
        argv=forwarded,
        semantic_router_factory=make_router,
        execution_recorder_factory=make_execution_recorder,
    )


if __name__ == "__main__":
    main()
