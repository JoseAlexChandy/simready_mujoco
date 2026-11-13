"""Write ATTRIBUTION.md for the assets in a batch config (CC-BY requires it when sharing).

    python scripts/attribution.py [configs/subset.yaml] > ATTRIBUTION.md

For each asset: its id, title/package id, licence, source (SimReady URL or local path),
and the note that it was converted to MuJoCo MJCF by this project. Licence and package
details come from the SimReady packaging file next to the asset's USD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PACKAGING = "com.nvidia.simready.packaging.json"
SOURCE_HOME = "https://huggingface.co/spaces/nvidia/simready-central"


def packaging_for(usd: Path) -> dict:
    for folder in [usd.parent, *usd.parents]:
        f = folder / PACKAGING
        if f.exists():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                return {}
        if folder.name in ("data", "") or folder == folder.parent:
            break
    return {}


def main(config: str = "configs/subset.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    rows, licences = [], {}
    for e in cfg.get("assets", []):
        if "path" in e and str(e["path"]).startswith("tests/"):
            continue  # synthetic fixtures made by this project
        info: dict = {}
        source = e.get("simready") or e.get("path", "")
        if "path" in e:
            info = packaging_for(Path(e["path"]).expanduser())
        elif "simready" in e:
            parts = e["simready"].split("/resolve/")[-1].split("/")
            local = (
                list(Path("data").glob(f"*/*/*/{parts[1]}/{PACKAGING}")) if len(parts) > 1 else []
            )
            info = json.loads(local[0].read_text()) if local else {}
        lic = info.get("license") or info.get("licence") or "see source"
        title = info.get("name") or info.get("title") or info.get("package_id") or e["id"]
        by = info.get("author") or info.get("authors") or info.get("publisher") or ""
        if isinstance(by, list):
            by = ", ".join(str(b) for b in by)
        licences[lic.upper()] = licences.get(lic.upper(), 0) + 1
        src = source if str(source).startswith("http") else "SimReady Central download"
        rows.append(f"| `{e['id']}` | {title} | {by} | {lic} | {src} |")

    out = sys.stdout
    out.write("# Asset attribution\n\n")
    out.write(
        "The 3D assets in this release come from [NVIDIA SimReady Central]"
        f"({SOURCE_HOME}) and its publishers. They were **converted from OpenUSD to MuJoCo MJCF** "
        "by [simready-mujoco](https://github.com/JoseAlexChandy/simready_mujoco): geometry, "
        "materials and textures re-exported (textures resized to at most 2048 px), physics "
        "translated, and in some cases mass properties corrected (see each asset's "
        "`conversion_report.json`). Each asset keeps the licence set by its publisher.\n\n"
    )
    out.write("Licences: " + ", ".join(f"{k} ({v})" for k, v in sorted(licences.items())) + "\n\n")
    out.write(
        "CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/ — share and adapt with "
        "attribution, indicating changes.\n\n"
    )
    out.write("| id | title / package | author | licence | source |\n|---|---|---|---|---|\n")
    out.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
