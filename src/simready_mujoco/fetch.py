"""Resolve asset entries from a batch config to local files (downloading from HF if needed).

An entry can be:
  path:     a local USD file
  simready: a package URL copied from the SimReady Central index
            (github.com/NVIDIA-Omniverse/simready-central, files under submissions/)
  hf:       { repo, file, revision } for any USD on a Hugging Face dataset
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")
PACKAGING_FILE = "com.nvidia.simready.packaging.json"

_PACKAGE_URL = re.compile(
    r"https://huggingface\.co/datasets/(?P<repo>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/"
    r"(?P<asset>.+)/" + re.escape(PACKAGING_FILE) + r"$"
)


@dataclass
class PackageRef:
    repo: str
    revision: str
    asset: str  # asset folder inside the dataset


@dataclass
class ResolvedAsset:
    usd: Path
    license: str | None = None
    package_id: str | None = None


def parse_package_url(url: str) -> PackageRef:
    m = _PACKAGE_URL.match(url.strip())
    if not m:
        raise ValueError(
            f"Not a SimReady package URL: {url}\n"
            f"Expected https://huggingface.co/datasets/<org>/<repo>/resolve/<commit>/<asset>/{PACKAGING_FILE}"
        )
    return PackageRef(**m.groupdict())


def find_interface_usd(asset_dir: Path, hint: str | None = None) -> Path:
    """Find the asset's root USD: <asset>/<intermediate folder>/<interface>.usd (SimReady layout)."""
    if hint:
        p = asset_dir / hint
        if not p.exists():
            raise FileNotFoundError(f"interface '{hint}' not found in {asset_dir}")
        return p
    candidates = sorted(
        p
        for p in asset_dir.glob("*/*.usd*")
        if p.suffix in {".usd", ".usda", ".usdc", ".usdz"} and not p.parent.name.startswith(".")
    )
    if not candidates:
        raise FileNotFoundError(f"No interface USD found under {asset_dir}/*/")
    if len(candidates) > 1:
        # Prefer the one named after the asset folder; otherwise the first, and say so.
        named = [p for p in candidates if asset_dir.name.lower() in p.stem.lower()]
        if len(named) == 1:
            return named[0]
        raise ValueError(
            f"Several root USDs in {asset_dir}: {[str(p.relative_to(asset_dir)) for p in candidates]}. "
            "Set `interface:` in the config to pick one."
        )
    return candidates[0]


def read_package_info(asset_dir: Path) -> dict[str, Any]:
    f = asset_dir / PACKAGING_FILE
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    return {k: data.get(k) for k in ("package_id", "license", "format_version")}


def _download(repo: str, revision: str | None, patterns: list[str]) -> Path:
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep batch output readable
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Install the 'hf' extra: pip install -e '.[hf]'") from e
    return Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=DATA_DIR / repo / (revision or "main"),
        )
    )


def resolve_asset(entry: dict[str, Any], base_dir: Path = Path(".")) -> ResolvedAsset:
    """Return the local root USD (downloading if needed) plus licence info for one config entry."""
    if "path" in entry:
        p = Path(entry["path"])
        return ResolvedAsset(usd=p if p.is_absolute() else base_dir / p)

    if "simready" in entry:
        ref = parse_package_url(entry["simready"])
        local = _download(ref.repo, ref.revision, [f"{ref.asset}/**"])
        asset_dir = local / ref.asset
        info = read_package_info(asset_dir)
        return ResolvedAsset(
            usd=find_interface_usd(asset_dir, entry.get("interface")),
            license=info.get("license"),
            package_id=info.get("package_id"),
        )

    if "hf" in entry:
        hf = entry["hf"]
        folder = str(Path(hf["file"]).parent)
        # Download the whole folder: USD files reference textures/payloads next to them.
        pattern = f"{folder}/**" if folder != "." else hf["file"]
        local = _download(hf["repo"], hf.get("revision"), [pattern])
        return ResolvedAsset(usd=local / hf["file"])

    raise ValueError(f"Asset '{entry.get('id')}' needs one of: path, simready, hf.")
