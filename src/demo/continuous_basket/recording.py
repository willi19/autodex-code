"""Small hardware-independent helpers for continuous demo recording."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


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
