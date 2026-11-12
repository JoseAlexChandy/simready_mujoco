"""Baseline backend: the upstream LightwheelAI/usd2mjcf converter (git submodule).

Upstream has no installable package, so we add its folder to sys.path. Set
SRMJ_USD2MJCF_PATH to point at a different checkout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = REPO_ROOT / "third_party" / "usd2mjcf"


class UpstreamNotFound(RuntimeError):
    pass


def upstream_path() -> Path:
    p = Path(os.environ.get("SRMJ_USD2MJCF_PATH", DEFAULT_PATH))
    if not (p / "lightwheel").is_dir():
        raise UpstreamNotFound(
            f"usd2mjcf not found at {p}. Run `git submodule update --init` "
            "or set SRMJ_USD2MJCF_PATH."
        )
    return p


def _ensure_importable() -> Path:
    p = upstream_path()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


_UTILS_PKG = "_usd2mjcf_utils"


def _load_upstream_utils(root: Path):
    """Import upstream's `utils/` package under a private name.

    Its name (`utils`) is too generic to put on sys.path, and it uses relative imports,
    so it must be loaded as a package rather than file by file.
    """
    if _UTILS_PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _UTILS_PKG,
            root / "utils" / "__init__.py",
            submodule_search_locations=[str(root / "utils")],
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_UTILS_PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{_UTILS_PKG}.add_collision") and sys.modules[_UTILS_PKG]


def convert(
    usd_file: Path,
    out_dir: Path,
    generate_collision: bool = False,
    preprocess_resolution: int = 20,
    resolution: int = 2000,
    stage=None,
) -> Path:
    """Convert with upstream usd2mjcf. Returns the path of the written MJCF file.

    Pass `stage` to convert an already-opened stage (e.g. with a physics variant selected);
    otherwise `usd_file` is opened as authored.
    """
    root = _ensure_importable()
    from lightwheel.srl.from_usd.to_mjcf import UsdToMjcf  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    converter = UsdToMjcf(stage) if stage is not None else UsdToMjcf.init_from_file(str(usd_file))
    mjcf_file = Path(converter.save_to_file(str(out_dir)))

    if generate_collision:
        mod = _load_upstream_utils(root).add_collision
        mod.add_collision_to_only_visual_mjcf(
            str(mjcf_file),
            preprocess_resolution=preprocess_resolution,
            resolution=resolution,
        )
    return mjcf_file
