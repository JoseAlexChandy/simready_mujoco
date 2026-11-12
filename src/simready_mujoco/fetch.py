"""Resolve asset entries from a batch config to local files (downloading from HF if needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATA_DIR = Path("data")


def resolve_asset(entry: dict[str, Any], base_dir: Path = Path(".")) -> Path:
    """Return a local path for an asset entry with either `path` or `hf` set."""
    if "path" in entry:
        p = Path(entry["path"])
        return p if p.is_absolute() else (base_dir / p)

    if "hf" in entry:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("Install the 'hf' extra: pip install -e '.[hf]'") from e
        hf = entry["hf"]
        file = hf["file"]
        # Download the asset's whole folder: USD files reference textures/payloads next to them.
        folder = str(Path(file).parent)
        pattern = f"{folder}/**" if folder != "." else file
        local = snapshot_download(
            repo_id=hf["repo"],
            repo_type="dataset",
            revision=hf.get("revision"),
            allow_patterns=[pattern],
            local_dir=DATA_DIR / hf["repo"] / (hf.get("revision") or "main"),
        )
        return Path(local) / file

    raise ValueError(f"Asset '{entry.get('id')}' needs either 'path' or 'hf'.")
