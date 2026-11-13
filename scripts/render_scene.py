"""Render pictures of a scene for the README (offscreen, no window).

    python scripts/render_scene.py                                  # overview + default close-ups
    python scripts/render_scene.py --focus d00326 "d00249 d00270" microwave_oven
    python scripts/render_scene.py --open 0.5 --scene outputs/scene/scene.xml

The scene settles for a moment first; with --open, every door, drawer and lid that has a
limited range is opened to that fraction of it (motors are moved with them), to show the
articulation. Each --focus is one picture framed around those assets (space-separated ids
in one argument share a picture). Writes docs/images/scene_*.jpg.
Uses EGL by default (MUJOCO_GL=glfw or osmesa to change).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_FOCUS = [
    "d00326 d00081 d00323 d00395",  # display shelf with baskets
    "d00249 d00270 d00268 d00264 d00288",  # caddy, bowl, plate, board, eggs
    "microwave_oven laptop_16z90tl air_fryer toaster",  # appliances, doors open
    "refrigerator_a01 cabinet_file_a01",  # fridge and file cabinet
]


def settle(m: mujoco.MjModel, d: mujoco.MjData, seconds: float) -> None:
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def open_parts(m: mujoco.MjModel, d: mujoco.MjData, fraction: float) -> int:
    """Move every limited hinge/slide joint to `fraction` of its range (motors follow)."""
    n = 0
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not m.jnt_limited[j]:
            continue
        lo, hi = m.jnt_range[j]
        span = hi - lo
        if span < 0.02:  # buttons, triggers: leave alone
            continue
        rest = d.qpos[m.jnt_qposadr[j]]
        target = lo + fraction * span if abs(rest - lo) <= abs(rest - hi) else hi - fraction * span
        d.qpos[m.jnt_qposadr[j]] = target
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j:
                d.ctrl[a] = target
        n += 1
    mujoco.mj_forward(m, d)
    return n


def bounds(m: mujoco.MjModel, d: mujoco.MjData, prefixes: list[str] | None):
    """World bounds of the visible geoms of assets whose names start with a prefix."""
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 or int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        name = m.body(m.geom_bodyid[g]).name
        if prefixes is not None and not any(name.startswith(p + "/") for p in prefixes):
            continue
        if prefixes is None and "/" not in name:
            continue  # floor, sheet
        r = float(m.geom_rbound[g]) or 0.05
        pts += [d.geom_xpos[g] - r, d.geom_xpos[g] + r]
    if not pts:
        return None
    p = np.array(pts)
    return p.min(0), p.max(0)


def render(m, d, lo, hi, out: Path, width: int, height: int, azimuth: float, elevation: float):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (lo + hi) / 2
    size = hi - lo
    cam.distance = 0.75 * float(np.linalg.norm(size)) + 0.3
    cam.azimuth, cam.elevation = azimuth, elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # hide collision geoms
    with mujoco.Renderer(m, height, width) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        Image.fromarray(r.render()).save(out, quality=88, optimize=True)
    print(f"  {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="outputs/scene/scene.xml")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--focus", nargs="*", default=None, help="asset ids per close-up")
    ap.add_argument("--open", type=float, default=0.55, help="open parts to this fraction")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to settle first")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
    m.vis.quality.shadowsize = max(m.vis.quality.shadowsize, 4096)
    d = mujoco.MjData(m)
    settle(m, d, args.settle)
    if args.open > 0:
        print(f"Opened {open_parts(m, d, args.open)} doors, drawers and lids")
        settle(m, d, 0.3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    whole = bounds(m, d, None)
    if whole is not None:
        render(m, d, *whole, out / "scene_overview.jpg", args.width, args.height, 125, -30)
    focus = DEFAULT_FOCUS if args.focus is None else args.focus
    present = {m.body(b).name.split("/")[0] for b in range(m.nbody)}
    for i, group in enumerate(focus, 1):
        ids = [x for x in group.split() if x in present]
        if not ids:
            print(f"  skip close-up {i}: none of {group} in the scene")
            continue
        b = bounds(m, d, ids)
        if b is not None:
            render(m, d, *b, out / f"scene_closeup_{i}.jpg", args.width, args.height, 135, -25)


if __name__ == "__main__":
    main()
