"""Put objects on or in other objects: an apple in a bowl, knives in a cutlery caddy, a
basket on the middle tier of a shelf.

In configs/scene.yaml, under `place:`:

    d00322: {in: d00270}                 # apple in the bowl
    d00253: {in: d00249, upright: true}  # knife standing in the caddy
    d00323: {on: d00326, level: 2}       # basket on the shelf's second surface from the top
    d00080: {on: d00264, at: [0.3, 0]}   # loaf on the cutting board, right of centre

Surfaces come from where a vertical line through the spot is inside the target's collision
shape: the top of each solid span is a surface. `on` takes the n-th from the top
(`level: n`, default 1): a board, a stove, a shelf tier. `in` takes the lowest, the
container's floor, so a basket's handle or a caddy's dividers don't count.
`at: [x, y]` is a spot across the target's footprint (-1..1 each way, 0 = centre). Without
it, items sharing a target are spread over a grid. `upright: true` stands an item on end,
long axis vertical, heavier end down. `yaw` turns it (degrees).

Targets can themselves sit on other targets (fruit in a basket on a shelf). The item is
placed just above the surface, and the settle check confirms it comes to rest.

For a container to hold things inside it, its collider must be hollow. Convert containers
with `decompose: concave` (see `srmj set`); a solid convex hull has no inside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

CLEARANCE = 0.004  # drop from just above the surface
SPREAD = 0.55  # items spread over this fraction of the target's footprint


@dataclass
class NestSpec:
    target: str
    level: int = 1
    at: tuple[float, float] | None = None
    upright: bool = False
    yaw: float = 0.0
    inside: bool = (
        False  # `in:` -> the container's floor (lowest surface); `on:` -> counts from the top
    )

    @classmethod
    def parse(cls, value) -> NestSpec | None:
        if not isinstance(value, dict):
            return None
        # YAML 1.1 reads a bare `on` key as the boolean true: {on: bowl} -> {True: 'bowl'}
        value = {("on" if k is True else k): v for k, v in value.items()}
        keys = set(value) - {"on", "in", "level", "at", "upright", "yaw"}
        if keys:
            raise ValueError(f"place: unknown keys {sorted(keys)} in {value}")
        inside = "in" in value and "on" not in value
        target = value.get("on", value.get("in"))
        if not target:
            raise ValueError(f"place: {value} needs `on:` or `in:` naming another asset")
        at = value.get("at")
        return cls(
            target=str(target),
            level=int(value.get("level", 1)),
            at=(float(at[0]), float(at[1])) if at is not None else None,
            upright=bool(value.get("upright", False)),
            yaw=float(value.get("yaw", 0.0)),
            inside=inside and "level" not in value,
        )


@dataclass
class Shape:
    """A converted asset at its rest pose, in its own frame."""

    model: mujoco.MjModel
    data: mujoco.MjData
    points: np.ndarray  # collision geometry points (vertices / box corners)
    com: np.ndarray
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.lo, self.hi = self.points.min(0), self.points.max(0)


def load_shape(mjcf: str) -> Shape:
    from simready_mujoco.sim.scene import geometry_model  # noqa: PLC0415

    m = geometry_model(mjcf)  # no textures: only the geometry is needed here
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    geoms = geoms or list(range(m.ngeom))
    pts = []
    box = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    for g in geoms:
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            v = m.geom_aabb[g, :3] + box * m.geom_aabb[g, 3:]
        pts.append(p + np.asarray(v, float) @ R.T)
    points = np.vstack(pts) if pts else np.zeros((1, 3))
    roots = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    mass = sum(m.body_subtreemass[b] for b in roots) or 1.0
    com = sum(m.body_subtreemass[b] * d.subtree_com[b] for b in roots) / mass
    return Shape(m, d, points, np.asarray(com, float))


def _geom_ray(m, d, g: int, start: np.ndarray, vec: np.ndarray) -> float:
    """Distance along `vec` from `start` to collision geom g, or -1."""
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        return float(mujoco.mj_rayMesh(m, d, g, start, vec))
    return float(
        mujoco.mju_rayGeom(
            d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], start, vec, int(m.geom_type[g])
        )
    )


def solid_spans(shape: Shape, xy: np.ndarray) -> list[tuple[float, float]]:
    """Where the vertical line through (x, y) is inside the collision geometry: merged
    (bottom, top) spans, highest first. Each collider is convex, so the line enters it once
    from above and once from below; that gives its span exactly, however thick it is."""
    m, d = shape.model, shape.data
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    geoms = geoms or list(range(m.ngeom))
    top, bottom = shape.hi[2] + 0.05, shape.lo[2] - 0.05
    down, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    spans = []
    for g in geoms:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        a = _geom_ray(m, d, g, np.array([xy[0], xy[1], top]), down)
        b = _geom_ray(m, d, g, np.array([xy[0], xy[1], bottom]), up)
        if a >= 0 and b >= 0:
            spans.append((bottom + b, top - a))
    spans.sort(key=lambda s: -s[1])
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and hi >= merged[-1][0] - 0.001:  # touches/overlaps the span above
            merged[-1][0] = min(merged[-1][0], lo)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def surfaces(shape: Shape, xy: np.ndarray) -> list[float]:
    """Heights of the upward-facing surfaces on the vertical line through (x, y), top first
    (the top of each solid span: a shelf board, a basket's handle, a bowl's floor)."""
    return [hi for _, hi in solid_spans(shape, xy)]


SEARCH_STEP, SEARCH_RADIUS = 0.01, 0.05  # m


def _nearest_surface(shape: Shape, xy: np.ndarray) -> tuple[np.ndarray, list[float], float]:
    """Look along nearby vertical lines (1 cm grid, up to 5 cm away, nearest first) for one
    that meets the geometry. Returns (xy used, surfaces, distance moved)."""
    n = int(SEARCH_RADIUS / SEARCH_STEP)
    offsets = [
        np.array([i, j]) * SEARCH_STEP
        for i in range(-n, n + 1)
        for j in range(-n, n + 1)
        if (i or j) and math.hypot(i, j) * SEARCH_STEP <= SEARCH_RADIUS + 1e-9
    ]
    offsets.sort(key=lambda o: float(np.hypot(*o)))
    for o in offsets:
        found = surfaces(shape, xy + o)
        if found:
            return xy + o, found, float(np.hypot(*o))
    return xy, [], 0.0


def rotation(shape: Shape, spec: NestSpec) -> np.ndarray:
    """Quaternion (w, x, y, z): optionally stand the item on end, then turn it by yaw."""
    q = np.array([1.0, 0, 0, 0])
    if spec.upright:
        size = shape.hi - shape.lo
        k = int(np.argmax(size))
        centre = (shape.hi + shape.lo) / 2
        h = np.zeros(3)
        h[k] = 1.0 if shape.com[k] >= centre[k] else -1.0  # towards the heavier end
        target = np.array([0.0, 0.0, -1.0])  # heavier end down
        axis = np.cross(h, target)
        if np.linalg.norm(axis) < 1e-9:  # already along z
            axis, angle = np.array([1.0, 0, 0]), (0.0 if h[2] < 0 else math.pi)
        else:
            angle = math.acos(float(np.clip(np.dot(h, target), -1, 1)))
            axis = axis / np.linalg.norm(axis)
        mujoco.mju_axisAngle2Quat(q, axis, angle)
    if spec.yaw:
        qy = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qy, np.array([0.0, 0, 1]), math.radians(spec.yaw))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qy, q)
        q = out
    return q


def quat_matrix(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def spread(n: int) -> list[tuple[float, float]]:
    """n spots over a grid across -1..1, centre first for one item."""
    if n <= 1:
        return [(0.0, 0.0)]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    xs = np.linspace(-1, 1, cols) if cols > 1 else [0.0]
    ys = np.linspace(1, -1, rows) if rows > 1 else [0.0]
    return [(float(x), float(y)) for y in ys for x in xs][:n]


def place_on(
    item: Shape,
    spec: NestSpec,
    target: Shape,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    at: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Frame (pos, quat) for `item` resting on `target`, plus a note on what was found."""
    centre = (target.lo + target.hi) / 2
    half = (target.hi - target.lo) / 2 * SPREAD
    xy = centre[:2] + np.array(at) * half[:2]
    found, moved = surfaces(target, xy), 0.0
    if not found:  # open structures (a wire rack): the line can pass between the wires
        xy, found, moved = _nearest_surface(target, xy)
    heights = "/".join(f"{z:.2f}" for z in found) + (
        f", {moved * 100:.0f} cm from the spot" if moved else ""
    )
    if not found:
        z_local, note = target.hi[2], "no surface found; on top"
    elif spec.inside:
        z_local, note = found[-1], f"floor ({heights} m)"
    elif len(found) >= spec.level:
        z_local, note = found[spec.level - 1], f"surface {spec.level} ({heights} m)"
    else:
        z_local, note = found[-1], f"only {len(found)} surface(s) ({heights} m); lowest"
    spot = np.asarray(target_pos) + quat_matrix(target_quat) @ np.array([xy[0], xy[1], z_local])
    q_rel = rotation(item, spec)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(target_quat, float), q_rel)  # turn with the target
    pts = item.points @ quat_matrix(q).T
    lo, hi = pts.min(0), pts.max(0)
    pos = np.array(
        [spot[0] - (lo[0] + hi[0]) / 2, spot[1] - (lo[1] + hi[1]) / 2, spot[2] - lo[2] + CLEARANCE]
    )
    return pos, q, note
