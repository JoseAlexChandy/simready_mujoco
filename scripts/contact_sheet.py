"""Render every converted asset into one labelled picture, to see what each one is.

    python scripts/contact_sheet.py                       # all non-robot assets
    python scripts/contact_sheet.py --only D00080 D00081  # just these
    python scripts/contact_sheet.py --new                 # assets added since the last sheet

Each tile shows the asset from a three-quarter view with its id, size (cm) and mass.
Writes outputs/contact_sheet.png (and remembers which ids it showed, for --new).
Rendering is offscreen: uses EGL by default (set MUJOCO_GL=glfw or osmesa to change).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from simready_mujoco.sim.scene import ROBOT_CATEGORIES, _bbox  # noqa: E402

TILE_W, TILE_H, LABEL_H, COLS = 320, 240, 44, 6


def _bounds(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of the visible geometry: mesh vertices where there are meshes, the geom's
    own box otherwise."""
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 and m.ngeom > 1:  # collision copies
            continue
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        else:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            v = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        pts.append(p + v @ R.T)
    if not pts:
        return _bbox(m)
    allp = np.vstack(pts)
    return allp.min(0), allp.max(0)


def render_tile(xml: Path) -> tuple[Image.Image, np.ndarray, float]:
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, TILE_W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, TILE_H)
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.6
    m.vis.headlight.specular[:] = 0.1
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = _bounds(m, d)
    size = hi - lo
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.6 * float(np.linalg.norm(size)) + 0.03
    cam.azimuth, cam.elevation = 135.0, -25.0
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # hide collision geoms
    with mujoco.Renderer(m, TILE_H, TILE_W) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        depth = r.render()
    background = depth >= depth.max() - 1e-6  # nothing drawn there
    rgb[background] = 255
    return Image.fromarray(rgb), size, float(m.body_mass.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/subset.yaml")
    ap.add_argument("--only", nargs="*", help="asset ids to show")
    ap.add_argument("--new", action="store_true", help="only assets not on the last sheet")
    ap.add_argument("--out", default=None, help="output PNG (default outputs/contact_sheet.png)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(cfg.get("output_dir", "outputs"))
    out = Path(args.out) if args.out else out_root / "contact_sheet.png"
    seen_file = out_root / "contact_sheet_ids.json"
    seen = set(json.loads(seen_file.read_text())) if args.new and seen_file.exists() else set()

    ids = [
        e["id"]
        for e in cfg.get("assets", [])
        if e.get("category") not in ROBOT_CATEGORIES
        and (not args.only or e["id"] in args.only)
        and e["id"] not in seen
    ]
    tiles = []
    for aid in ids:
        xmls = sorted((out_root / aid).glob("*.xml"))
        if not xmls:
            print(f"  {aid}: not converted yet (run srmj batch), skipped")
            continue
        try:
            img, size, mass = render_tile(xmls[0])
        except Exception as e:  # noqa: BLE001
            print(f"  {aid}: could not render ({e})")
            continue
        tiles.append((aid, img, size, mass))
        print(f"  {aid}: {' x '.join(f'{s * 100:.0f}' for s in size)} cm, {mass:.2f} kg")
    if not tiles:
        sys.exit("Nothing to render.")

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE_W, rows * (TILE_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    for i, (aid, img, size, mass) in enumerate(tiles):
        x, y = (i % COLS) * TILE_W, (i // COLS) * (TILE_H + LABEL_H)
        sheet.paste(img, (x, y))
        draw.text((x + 6, y + TILE_H + 3), aid, fill="black", font=font)
        dims = " x ".join(f"{s * 100:.0f}" for s in size)
        draw.text((x + 6, y + TILE_H + 23), f"{dims} cm   {mass:.2f} kg", fill="#444", font=small)
        draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H + LABEL_H - 1], outline="#ccc")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    seen_file.write_text(json.dumps(sorted(seen | {t[0] for t in tiles})))
    print(f"Wrote {out} ({len(tiles)} assets)")


if __name__ == "__main__":
    main()
