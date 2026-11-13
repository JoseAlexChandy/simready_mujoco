"""Compose converted assets into one MuJoCo scene.

Default layout ("sheet"): a light-brown sheet on the floor with every object laid out on it
in rows, big items at the back and small ones in front. Optional "table" layout: big items on
the floor behind a table, small items on the tabletop, fixed-base robots on its left end.
Robots and grippers are left out unless `include_robots` is set.

Placement uses each asset's collision bounding box at its rest pose, and every item starts a
few millimetres above its support, so it settles instead of dropping. Unactuated doors,
drawers and other moving parts get light damping and friction (scene only, assets unchanged)
so they stay where you drag them in the viewer.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from simready_mujoco._quiet import redirect_output
from simready_mujoco.sim.profiles import get_profile

CLEARANCE = 0.003  # start just above the support surface
FLOOR_HEIGHT = 0.5  # taller than this -> floor
FLOOR_FOOTPRINT = 0.6  # wider/deeper than this -> floor
GAP = 0.06  # spacing between items on the table
FLOOR_GAP = 0.35


@dataclass
class TableSpec:
    enabled: bool = False
    length: float = 1.4  # along x; grows to fit the items
    width: float = 0.8  # along y
    height: float = 0.75
    top: float = 0.04  # tabletop thickness


@dataclass
class SheetSpec:
    """Floor sheet the objects stand on (used when the table is disabled)."""

    rgba: tuple = (0.80, 0.68, 0.50, 1.0)  # light brown
    margin: float = 0.4  # border around the objects, metres
    thickness: float = 0.004
    gap: float = 0.3  # spacing between objects
    length: float = 0.0  # filled in by layout
    width: float = 0.0


@dataclass
class ArticulationSpec:
    """Damping/friction for unactuated hinge and slide joints, so dragged parts stay put."""

    damping: float = 0.5  # N·m·s/rad (hinges) or N·s/m (slides)
    frictionloss: float = 0.05
    enabled: bool = True


ROBOT_CATEGORIES = ("robot", "gripper")


@dataclass
class Item:
    id: str
    mjcf: str
    place: str = "auto"  # auto | floor | table | robot | skip | [x, y, z]
    category: str | None = None
    yaw_deg: float = 0.0
    # filled in by layout
    bbox_min: tuple = (0.0, 0.0, 0.0)
    bbox_max: tuple = (0.0, 0.0, 0.0)
    pos: tuple = (0.0, 0.0, 0.0)
    surface: str = ""


@dataclass
class SceneReport:
    scene: str
    table: dict
    items: list[dict] = field(default_factory=list)
    settle: list[dict] = field(default_factory=list)
    sheet: dict | None = None
    skipped_robots: list[str] = field(default_factory=list)
    softened_joints: int = 0


def _bbox(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """World AABB of collision geoms (all geoms if none collide) at qpos0."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    geoms = geoms or list(range(m.ngeom))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for g in geoms:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = c + signs * h
        R = d.geom_xmat[g].reshape(3, 3)
        world = corners @ R.T + d.geom_xpos[g]
        lo, hi = np.minimum(lo, world.min(0)), np.maximum(hi, world.max(0))
    if not np.isfinite(lo).all():
        lo, hi = np.zeros(3), np.zeros(3)
    return lo, hi


def _has_free_root(m: mujoco.MjModel) -> bool:
    return any(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(m.njnt))


def _classify(item: Item, size: np.ndarray, free: bool) -> str:
    if item.place != "auto":
        return item.place if isinstance(item.place, str) else "fixed"
    if item.category == "robot" and not free:
        return "robot"
    if size[2] > FLOOR_HEIGHT or max(size[0], size[1]) > FLOOR_FOOTPRINT:
        return "floor"
    return "table"


def layout(items: list[Item], table: TableSpec) -> list[Item]:
    """Assign positions. Returns the items that will be placed (skips removed)."""
    placed: list[Item] = []
    table_items, floor_items, robots = [], [], []
    for it in items:
        if it.place == "skip":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mujoco.MjModel.from_xml_path(it.mjcf)
        lo, hi = _bbox(m)
        it.bbox_min, it.bbox_max = tuple(lo), tuple(hi)
        size = hi - lo
        kind = _classify(it, size, _has_free_root(m))
        it.surface = kind
        if kind == "table":
            table_items.append((it, size))
        elif kind == "floor":
            floor_items.append((it, size))
        elif kind == "robot":
            robots.append((it, size))
        else:  # explicit [x, y, z]
            it.pos = tuple(float(v) for v in it.place)
            it.surface = "fixed"
        placed.append(it)

    # Table: robots take the left end; objects fill columns (along y) from left to right.
    robot_zone = 0.45 if robots else 0.0
    usable_w = table.width - 2 * GAP
    columns: list[list] = [[]]
    used_w = 0.0
    for it, size in sorted(table_items, key=lambda p: -p[1][1]):
        if columns[-1] and used_w + size[1] > usable_w:
            columns.append([])
            used_w = 0.0
        columns[-1].append((it, size))
        used_w += size[1] + GAP
    widths = [max((s[0] for _, s in col), default=0.0) for col in columns]
    needed = sum(w + GAP for w in widths)
    table.length = max(table.length, robot_zone + needed + 2 * GAP)
    x = -table.length / 2 + robot_zone + GAP
    for col, cw in zip(columns, widths, strict=False):
        y = -usable_w / 2
        for it, size in col:
            it.pos = _rest_pos(it, x + cw / 2, y + size[1] / 2, table.height)
            y += size[1] + GAP
        x += cw + GAP

    for it, _size in robots:
        cx = -table.length / 2 + robot_zone / 2
        it.pos = _rest_pos(it, cx, 0.0, table.height)

    # Floor: a row behind the table (+y), left to right.
    y_floor = table.width / 2 + FLOOR_GAP
    total = sum(s[0] for _, s in floor_items) + FLOOR_GAP * max(len(floor_items) - 1, 0)
    x = -total / 2
    for it, size in floor_items:
        it.pos = _rest_pos(it, x + size[0] / 2, y_floor + size[1] / 2, 0.0)
        x += size[0] + FLOOR_GAP
    return placed


def layout_sheet(items: list[Item], sheet: SheetSpec) -> list[Item]:
    """Lay every item out on a floor sheet in rows: big items at the back (+y), small in front."""
    sized = []
    for it in items:
        if it.place == "skip":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mujoco.MjModel.from_xml_path(it.mjcf)
        lo, hi = _bbox(m)
        it.bbox_min, it.bbox_max = tuple(lo), tuple(hi)
        if isinstance(it.place, (list, tuple)):
            it.pos, it.surface = tuple(float(v) for v in it.place), "fixed"
        else:
            sized.append((it, hi - lo))
    big = [p for p in sized if p[1][2] > FLOOR_HEIGHT or max(p[1][:2]) > FLOOR_FOOTPRINT]
    small = [p for p in sized if p not in big]
    total_area = sum((s[0] + sheet.gap) * (s[1] + sheet.gap) for _, s in sized) or 1.0
    row_limit = max(2.0, 1.6 * np.sqrt(total_area))

    rows: list[list] = []
    for group in (sorted(big, key=lambda p: -p[1][2]), sorted(small, key=lambda p: -p[1][1])):
        row: list = []
        for it, size in group:
            if row and sum(s[0] + sheet.gap for _, s in row) + size[0] > row_limit:
                rows.append(row)
                row = []
            row.append((it, size))
        if row:
            rows.append(row)

    widths = [sum(s[0] for _, s in r) + sheet.gap * (len(r) - 1) for r in rows]
    depths = [max(s[1] for _, s in r) for r in rows]
    sheet.length = (max(widths) if widths else 1.0) + 2 * sheet.margin
    sheet.width = sum(depths) + sheet.gap * max(len(rows) - 1, 0) + 2 * sheet.margin
    y = sheet.width / 2 - sheet.margin  # back edge; rows go towards the viewer (-y)
    surface = sheet.thickness
    for row, w, dep in zip(rows, widths, depths, strict=False):
        x = -w / 2
        for it, size in row:
            it.pos = _rest_pos(it, x + size[0] / 2, y - dep / 2, surface)
            it.surface = "sheet"
            x += size[0] + sheet.gap
        y -= dep + sheet.gap
    return [it for it in items if it.place != "skip"]


def _rest_pos(it: Item, cx: float, cy: float, surface_z: float) -> tuple[float, float, float]:
    lo, hi = np.array(it.bbox_min), np.array(it.bbox_max)
    centre = (lo + hi) / 2
    return (cx - centre[0], cy - centre[1], surface_z - lo[2] + CLEARANCE)


def _shell(profile: dict[str, Any], table: TableSpec, sheet: SheetSpec | None) -> mujoco.MjSpec:
    opt = " ".join(f'{k}="{v}"' for k, v in profile.items() if k != "flags")
    flags = " ".join(f'{k}="{v}"' for k, v in (profile.get("flags") or {}).items())
    support = ""
    if table.enabled:
        L, W, H, T = table.length, table.width, table.height, table.top
        leg = 0.03
        legs = "\n".join(
            f'      <geom name="table_leg{i}" type="box" size="{leg} {leg} {(H - T) / 2}" '
            f'pos="{sx * (L / 2 - 2 * leg)} {sy * (W / 2 - 2 * leg)} {(H - T) / 2}" material="table"/>'
            for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        )
        support = f"""    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="{L / 2} {W / 2} {T / 2}" pos="0 0 {H - T / 2}" material="table" friction="0.8 0.005 0.0001"/>
{legs}
    </body>"""
    if sheet is not None:
        support += f"""
    <geom name="sheet" type="box" size="{sheet.length / 2} {sheet.width / 2} {sheet.thickness / 2}"
      pos="0 0 {sheet.thickness / 2}" rgba="{" ".join(str(c) for c in sheet.rgba)}" friction="0.9 0.005 0.0001"/>"""
    xml = f"""<mujoco model="simready_scene">
  <compiler angle="radian" autolimits="true"/>
  <option {opt}>{f"<flag {flags}/>" if flags else ""}</option>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="120" elevation="-25" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.25 0.3 0.35"
      rgb2="0.18 0.22 0.27" markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="4 4" reflectance="0.15"/>
    <material name="table" rgba="0.55 0.4 0.28 1"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
{support}
  </worldbody>
</mujoco>"""
    return mujoco.MjSpec.from_string(xml)


def _soften_articulation(child: mujoco.MjSpec, art: ArticulationSpec) -> int:
    """Give unactuated hinge/slide joints light damping + friction so dragged parts stay put.
    Joints that already have damping, or are driven by an actuator, are left alone."""
    if not art.enabled:
        return 0
    driven = {a.target for a in child.actuators}
    n = 0
    for j in child.joints:
        if j.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if j.name in driven or np.any(np.asarray(j.damping) > 0):
            continue
        try:
            j.damping = art.damping
        except TypeError:  # MuJoCo >= 4: damping is an array
            j.damping = np.array([art.damping, 0.0, 0.0])
        if j.frictionloss == 0:
            j.frictionloss = art.frictionloss
        n += 1
    return n


def _absolute_meshes(child: mujoco.MjSpec, child_xml: Path) -> None:
    """Point mesh and texture files at their real location (absolute) before attaching."""
    mesh_dir = child_xml.parent / (child.meshdir or "")
    for mesh in child.meshes:
        if mesh.file:
            mesh.file = str((mesh_dir / mesh.file).resolve())
    child.meshdir = ""
    tex_dir = child_xml.parent / (child.texturedir or "")
    for tex in child.textures:
        if tex.file:
            tex.file = str((tex_dir / tex.file).resolve())
    child.texturedir = ""


def _relative_mesh_paths(xml: str, scene_dir: Path) -> str:
    """Rewrite absolute mesh paths in exported XML relative to the scene file, so the outputs
    folder can be moved. (MjSpec.to_xml recompiles, so this is done on the text.)"""
    import re

    base = scene_dir.resolve()

    def rel(match: re.Match) -> str:
        path = match.group(1)
        return f'file="{os.path.relpath(path, base)}"' if os.path.isabs(path) else match.group(0)

    return re.sub(r'file="([^"]+)"', rel, xml)


def build_scene(
    items: list[Item],
    out_file: str | Path,
    table: TableSpec | None = None,
    sim_profile: str = "manipulation",
    settle_seconds: float = 2.0,
    sheet: SheetSpec | None = None,
    include_robots: bool = False,
    articulation: ArticulationSpec | None = None,
) -> SceneReport:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = table or TableSpec()
    articulation = articulation or ArticulationSpec()
    skipped_robots = []
    if not include_robots:
        for it in items:
            if it.category in ROBOT_CATEGORIES and it.place == "auto":
                it.place = "skip"
                skipped_robots.append(it.id)
    if table.enabled:
        placed, sheet = layout(items, table), None
    else:
        sheet = sheet or SheetSpec()
        placed = layout_sheet(items, sheet)
    profile = get_profile(sim_profile)
    scene = _shell(profile, table, sheet)

    softened = 0
    with warnings.catch_warnings(), redirect_output(out_file.parent / "scene_build.log"):
        warnings.simplefilter("ignore")
        for it in placed:
            child = mujoco.MjSpec.from_file(it.mjcf)
            _absolute_meshes(child, Path(it.mjcf))
            softened += _soften_articulation(child, articulation)
            yaw = np.radians(it.yaw_deg)
            frame = scene.worldbody.add_frame(
                pos=list(it.pos), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            )
            scene.attach(child, prefix=f"{it.id}/", frame=frame)
        model = scene.compile()
        out_file.write_text(_relative_mesh_paths(scene.to_xml(), out_file.parent))

    report = SceneReport(str(out_file), asdict(table), [asdict(i) for i in placed])
    report.sheet = asdict(sheet) if sheet else None
    report.skipped_robots = skipped_robots
    report.softened_joints = softened
    report.settle = settle_check(model, placed, settle_seconds)
    return report


def settle_check(m: mujoco.MjModel, items: list[Item], seconds: float) -> list[dict]:
    """Simulate and report how far each free-floating item moved (did it rest or fall off?)."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    roots = {}
    for b in range(1, m.nbody):
        name = m.body(b).name
        if (
            m.body_parentid[b] == 0
            and m.body_jntnum[b]
            and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            roots[name] = (b, d.xpos[b].copy())
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    out = []
    for name, (b, start) in roots.items():
        moved = d.xpos[b] - start
        out.append(
            {
                "body": name,
                "drop_m": float(-moved[2]),
                "slide_m": float(np.linalg.norm(moved[:2])),
                "final_z": float(d.xpos[b][2]),
                "ok": bool(abs(moved[2]) < 0.05 and np.linalg.norm(moved[:2]) < 0.1),
            }
        )
    return out
