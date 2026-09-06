"""Small hardware-independent helpers for continuous demo recording."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_path_component(value: str) -> str:
    """Return a stable, readable directory component without path traversal."""
    cleaned = _SAFE_COMPONENT.sub("_", str(value).strip()).strip("._")
    if not cleaned:
        raise ValueError(f"path component is empty after sanitising {value!r}")
    return cleaned


def catalogue_path_component(object_names: Iterable[str]) -> str:
    """Readable fixed-catalogue directory, shortened deterministically if needed."""
    names = sorted({safe_path_component(name) for name in object_names})
    if not names:
        raise ValueError("at least one catalogue object is required")
    joined = "__".join(names)
    if len(joined) <= 96:
        return joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"catalogue_{len(names)}_{digest}"


def timestamp_session_name(*, now: Optional[dt.datetime] = None) -> str:
    """Return the timestamp-only episode name used by existing demos.

    Microseconds retain the legacy date/time convention while preventing two
    takes started within one second from colliding.  Human labels deliberately
    do not enter the directory tree: the timestamp is the episode identity.
    """
    now = now or dt.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")


def create_session_dir(
    project_root: Path,
    *,
    experiment_name: str,
    arm: str,
    hand: str,
    object_names: Iterable[str],
    now: Optional[dt.datetime] = None,
) -> tuple[Path, str]:
    """Create the canonical non-overwriting NAS session directory.

    The layout deliberately follows the existing banana-demo convention while
    adding a catalogue component for multi-object runs::

        AutoDex/experiment/continuous_basket/franka_inspire/banana/<timestamp>/
        AutoDex/experiment/continuous_basket/franka_inspire/apple__banana/<timestamp>/
    """
    parent = (
        Path(project_root)
        / "experiment"
        / safe_path_component(experiment_name)
        / f"{safe_path_component(arm)}_{safe_path_component(hand)}"
        / catalogue_path_component(object_names)
    )
    parent.mkdir(parents=True, exist_ok=True)
    base = timestamp_session_name(now=now)
    candidate = parent / base
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate, candidate.name
        except FileExistsError:
            # A frozen/mock clock or a hand-supplied suffix can still collide.
            candidate = parent / f"{base}_{suffix:02d}"
            suffix += 1


def autodex_session_relative(project_root: Path, session_dir: Path) -> Path:
    """Return the capture/NAS-relative form accepted by ParaDex camera sinks."""
    root = Path(project_root).resolve()
    session = Path(session_dir).resolve()
    try:
        return Path("AutoDex") / session.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"session {session} is outside AutoDex NAS root {root}") from exc


def parse_autodex_session_relative(value: str | Path) -> Path:
    """Validate an ``AutoDex/.../<timestamp>`` relative session path."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "AutoDex":
        raise ValueError("session must be a relative path beginning with 'AutoDex/'")
    return path


def resolve_signal_generator_params(
    configured: Mapping[str, Any], *, device_root: Path = Path("/dev")
) -> tuple[dict[str, Any], Optional[str]]:
    """Resolve a stale USBTMC device node without changing ParaDex config.

    Linux may enumerate the same UTG900E as ``/dev/usbtmc5`` rather than the
    historic ``/dev/usbtmc0`` after other USBTMC devices have appeared.  Use a
    sole visible USBTMC device only when the configured node is absent; leave
    ambiguous multi-device setups to the explicit ParaDex configuration.
    """
    params = dict(configured)
    configured_addr = params.get("addr")
    if configured_addr and Path(str(configured_addr)).exists():
        return params, None

    candidates = sorted(path for path in device_root.glob("usbtmc*") if path.exists())
    if len(candidates) != 1:
        return params, None

    resolved = str(candidates[0])
    params["addr"] = resolved
    return params, (
        f"configured trigger {configured_addr!r} is unavailable; "
        f"using discovered {resolved}"
    )


def should_auto_upload(
    *,
    camera_recording: bool,
    uploads_deferred: bool,
    normal_exit: bool,
    robot_motion_started: bool,
) -> bool:
    """Return whether a completed take is worth processing automatically.

    The capture PCs record from before perception so a real demonstration is
    one uncut take.  A run can nevertheless finish without ever commanding a
    robot motion (for example, when every grasp is rejected by collision
    checking).  Converting and transferring five camera streams in that case
    blocks the operator while providing no demonstration footage.  Preserve
    the raw AVI files for diagnosis, but require at least one physical motion
    before starting the expensive automatic NAS upload.
    """
    return (
        camera_recording
        and not uploads_deferred
        and normal_exit
        and robot_motion_started
    )
