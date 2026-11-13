"""Find USD assets on disk and add them to a batch config (`srmj add`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from simready_mujoco.fetch import PACKAGING_FILE, find_interface_usd

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}


@dataclass
class Found:
    id: str
    path: Path
    note: str = ""


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return s or "asset"


def find_assets(path: Path) -> tuple[list[Found], list[str]]:
    """USD assets under `path`. Returns (found, problems).

    - a .usd/.usda/.usdc/.usdz file             -> that file
    - a SimReady package folder (has packaging)  -> its root USD
    - any folder                                 -> every SimReady package inside it; if there
      are none, each USD at the top level, and for each subfolder the USD named like it
      (or its only top-level USD)
    """
    path = path.expanduser().resolve()
    found: list[Found] = []
    problems: list[str] = []
    if path.is_file():
        if path.suffix.lower() in USD_SUFFIXES:
            found.append(Found(_slug(path.stem), path))
        else:
            problems.append(f"{path}: not a USD file")
        return found, problems
    if not path.is_dir():
        return found, [f"{path}: not found"]

    packages = sorted(p.parent for p in path.rglob(PACKAGING_FILE))
    if packages:
        for pkg in packages:
            try:
                found.append(Found(_slug(pkg.name), find_interface_usd(pkg), "SimReady package"))
            except (FileNotFoundError, ValueError) as e:
                problems.append(f"{pkg}: {e}")
        return found, problems

    for f in sorted(path.iterdir()):
        if f.is_file() and f.suffix.lower() in USD_SUFFIXES:
            found.append(Found(_slug(f.stem), f))
        elif f.is_dir() and not f.name.startswith("."):
            usds = [u for u in sorted(f.iterdir()) if u.suffix.lower() in USD_SUFFIXES]
            named = [u for u in usds if u.stem.lower() == f.name.lower()]
            pick = named or (usds if len(usds) == 1 else [])
            if pick:
                found.append(Found(_slug(f.name), pick[0]))
            elif usds:
                problems.append(
                    f"{f}: several USD files {[u.name for u in usds]}; add the main one directly"
                )
    return found, problems


def add_to_config(config: Path, items: list[Found], category: str = "object") -> list[str]:
    """Append items to the config's `assets:` list (text append keeps comments).
    Returns the ids added; existing ids and paths are skipped."""
    cfg = yaml.safe_load(config.read_text()) or {}
    existing = cfg.get("assets") or []
    ids = {e.get("id") for e in existing}
    paths = {str(Path(e["path"]).expanduser().resolve()) for e in existing if "path" in e}
    lines, added = [], []
    for it in items:
        if str(it.path) in paths:
            continue
        aid, n = it.id, 2
        while aid in ids:
            aid, n = f"{it.id}_{n}", n + 1
        ids.add(aid)
        added.append(aid)
        lines += [f"  - id: {aid}", f"    category: {category}", f"    path: {it.path}"]
    if lines:
        text = config.read_text()
        if "assets:" not in text:
            text += "\nassets:\n"
        if not text.endswith("\n"):
            text += "\n"
        config.write_text(text + "\n".join(lines) + "\n")
    return added
