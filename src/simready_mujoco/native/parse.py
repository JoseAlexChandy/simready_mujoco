"""USD stage (physics variant already selected) -> IR Model.

Order of work:
  1. rigid bodies and their world frames
  2. joints -> kinematic tree (loops and excluded joints become equality constraints)
  3. collision geoms (authored colliders) and visual geoms (renderable meshes)
  4. mass properties via UsdPhysics.RigidBodyAPI.ComputeMassProperties (USD's own precedence
     rules), with per-collider volume/inertia computed from the collision geometry
  5. actuators: MjcActuator prims (1:1) first, UsdPhysics drives as fallback
  6. mimic joints (Newton or PhysX authoring) -> equality/joint
  7. scene: gravity, mjc:option / mjc:flag
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from simready_mujoco.ir import (
    Actuator,
    Body,
    Equality,
    Geom,
    Inertial,
    Joint,
    JointType,
    MeshData,
    Model,
    Pose,
    ShapeType,
    Tendon,
    VisualMaterial,
)
from simready_mujoco.native.geometry import (
    AXIS_VEC,
    IDENTITY_QUAT,
    bake_scale,
    decompose,
    mesh_arrays,
    pose_matrix,
    quat_tuple,
    relative,
    rigid_pose,
    rotate,
)
from simready_mujoco.usd.physics import parse_physics
from simready_mujoco.usd.stage import applied_schemas

DEG = math.pi / 180.0
SHAPE_KINDS = (
    "CubeShape",
    "SphereShape",
    "CapsuleShape",
    "Capsule1Shape",
    "CylinderShape",
    "Cylinder1Shape",
    "ConeShape",
    "MeshShape",
    "PlaneShape",
    "CustomShape",
)
JOINT_KINDS = ("RevoluteJoint", "PrismaticJoint", "SphericalJoint", "FixedJoint", "D6Joint")

# mjc:* attribute -> MJCF attribute, per prim kind. Range pairs are handled separately.
MJC_ACTUATOR_ATTRS = {
    "mjc:gainType": "gaintype",
    "mjc:gainPrm": "gainprm",
    "mjc:biasType": "biastype",
    "mjc:biasPrm": "biasprm",
    "mjc:dynType": "dyntype",
    "mjc:dynPrm": "dynprm",
    "mjc:gear": "gear",
    "mjc:ctrlLimited": "ctrllimited",
    "mjc:forceLimited": "forcelimited",
    "mjc:actLimited": "actlimited",
    "mjc:actDim": "actdim",
    "mjc:actEarly": "actearly",
    "mjc:group": "group",
    "mjc:inheritRange": "inheritrange",
}
MJC_ACTUATOR_RANGES = {
    "mjc:ctrlRange": "ctrlrange",
    "mjc:forceRange": "forcerange",
    "mjc:actRange": "actrange",
    "mjc:lengthRange": "lengthrange",
}
MJC_JOINT_ATTRS = {
    "mjc:armature": "armature",
    "mjc:damping": "damping",
    "mjc:frictionloss": "frictionloss",
    "mjc:stiffness": "stiffness",
    "mjc:springref": "springref",
    "mjc:margin": "margin",
    "mjc:solreflimit": "solreflimit",
    "mjc:solimplimit": "solimplimit",
    "mjc:solreffriction": "solreffriction",
    "mjc:solimpfriction": "solimpfriction",
    "mjc:springdamper": "springdamper",
    "mjc:actuatorgravcomp": "actuatorgravcomp",
    "mjc:actuatorfrclimited": "actuatorfrclimited",
}
MJC_JOINT_RANGES = {"mjc:actuatorfrcrange": "actuatorfrcrange"}
# Newton joint attributes (used by NVIDIA's MJCF->USD converter). mjc:* wins when both exist.
# newton:damping is per degree for revolute joints, like UsdPhysics drive gains.
NEWTON_JOINT_ATTRS = {
    "newton:armature": ("armature", False),
    "newton:damping": ("damping", True),
    "newton:friction": ("frictionloss", False),
}
MJC_TENDON_ATTRS = {
    "mjc:limited": "limited",
    "mjc:stiffness": "stiffness",
    "mjc:damping": "damping",
    "mjc:frictionloss": "frictionloss",
    "mjc:armature": "armature",
    "mjc:springlength": "springlength",
    "mjc:margin": "margin",
    "mjc:solreflimit": "solreflimit",
    "mjc:solimplimit": "solimplimit",
    "mjc:solreffriction": "solreffriction",
    "mjc:solimpfriction": "solimpfriction",
    "mjc:actuatorfrclimited": "actuatorfrclimited",
}
MJC_TENDON_RANGES = {"mjc:range": "range", "mjc:actuatorfrcrange": "actuatorfrcrange"}
MJC_COLLISION_ATTRS = {
    "mjc:condim": "condim",
    "mjc:gap": "gap",
    "mjc:margin": "margin",
    "mjc:priority": "priority",
    "mjc:solimp": "solimp",
    "mjc:solmix": "solmix",
    "mjc:solref": "solref",
}

DEFAULT_RGBA = (0.7, 0.7, 0.7, 1.0)
COLLISION_RGBA = (0.9, 0.4, 0.2, 0.4)
MIN_MASS = 1e-4
MIN_INERTIA = 1e-8


# ----------------------------------------------------------------------------- helpers


def _authored(prim: Usd.Prim, name: str):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def _plain(v):
    """USD value -> plain Python (lists of floats, str for tokens)."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return [float(x) for x in v]
    except TypeError:
        return str(v)


def _passthrough(prim: Usd.Prim, attrs: dict, ranges: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src, dst in attrs.items():
        v = _authored(prim, src)
        if v is not None:
            out[dst] = _plain(v)
    for src, dst in ranges.items():
        lo, hi = _authored(prim, f"{src}:min"), _authored(prim, f"{src}:max")
        if lo is not None or hi is not None:
            out[dst] = [float(lo or 0.0), float(hi or 0.0)]
    return out


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _finite_limit(limit) -> tuple[float, float] | None:
    if limit is None or not getattr(limit, "enabled", True):
        return None
    lo, hi = float(limit.lower), float(limit.upper)
    if not (math.isfinite(lo) and math.isfinite(hi)) or abs(lo) > 1e30 or abs(hi) > 1e30:
        return None
    if lo > hi:  # USD convention for a locked axis
        return None
    return lo, hi


class _Names:
    """Unique MJCF names per namespace."""

    def __init__(self) -> None:
        self.used: dict[str, set[str]] = {}

    def __call__(self, ns: str, base: str) -> str:
        used = self.used.setdefault(ns, set())
        base = base or ns
        name, i = base, 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        return name


@dataclass
class _Edge:
    path: str
    kind: str
    desc: Any
    b0: str  # "" = world
    b1: str
    f0: Gf.Matrix4d  # joint frame on body0 side, world
    f1: Gf.Matrix4d  # joint frame on body1 side, world
    local0: tuple  # (pos in rigid body0 frame, quat)
    local1: tuple


@dataclass
class _Ctx:
    stage: Usd.Stage
    mpu: float
    kpu: float
    xcache: UsdGeom.XformCache
    names: _Names = field(default_factory=_Names)
    model: Model = None  # type: ignore[assignment]
    body_rigid: dict[str, Gf.Matrix4d] = field(default_factory=dict)
    body_scale: dict[str, np.ndarray] = field(default_factory=dict)
    body_name: dict[str, str] = field(default_factory=dict)
    joint_name: dict[str, str] = field(default_factory=dict)  # USD joint path -> MJCF name
    joint_type: dict[str, JointType] = field(default_factory=dict)  # MJCF name -> type
    joint_range: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    joint_ref: dict[str, float] = field(default_factory=dict)
    mass_shapes: dict[str, trimesh.Trimesh] = field(default_factory=dict)  # collider -> body frame
    mesh_keys: dict[tuple, str] = field(default_factory=dict)
    decompose: str = "off"
    decompose_only: dict[str, int] | None = None  # mesh -> refinement level (None = all, level 0)
    decomposed: dict[str, list[str]] = field(default_factory=dict)  # mesh -> convex part meshes
    visual_materials: dict[str, str] = field(default_factory=dict)  # USD key -> material name

    def world(self, prim: Usd.Prim) -> Gf.Matrix4d:
        return self.xcache.GetLocalToWorldTransform(prim)


# ----------------------------------------------------------------------------- main entry


def parse_stage(
    stage: Usd.Stage,
    name: str = "model",
    actuators: str = "auto",
    decompose: str = "off",
    decompose_only: dict[str, int] | None = None,
) -> Model:
    """decompose: "concave" splits concave collision meshes into convex parts (CoACD), "all"
    splits every collision mesh, "off" keeps one convex hull per mesh. decompose_only limits
    it to those mesh names. ("auto" is resolved by the caller: see api.convert.)"""
    if decompose not in ("concave", "all", "off"):
        raise ValueError("parse_stage decompose must be concave | all | off")
    ctx = _Ctx(
        stage=stage,
        mpu=float(UsdGeom.GetStageMetersPerUnit(stage)),
        kpu=float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
        xcache=UsdGeom.XformCache(),
        decompose=decompose,
        decompose_only=decompose_only,
    )
    ctx.model = Model(name=name)
    parsed = parse_physics(stage)

    bodies = _collect_bodies(ctx, parsed)
    edges = _collect_edges(ctx, parsed, bodies)
    _build_tree(ctx, bodies, edges)
    collider_owner = _collect_colliders(ctx, parsed, bodies)
    _collect_visuals(ctx, bodies, set(collider_owner))
    _compute_inertials(ctx, bodies)
    tendons = _collect_tendons(ctx)
    _collect_actuators(ctx, parsed, tendons, actuators)
    _collect_mimics(ctx)
    _collect_scene(ctx, parsed)
    _collect_excludes(ctx, parsed, edges)

    m = ctx.model
    m.notes.update(
        bodies=len(m.bodies),
        joints=sum(len(b.joints) for b in m.bodies),
        actuators=len(m.actuators),
        actuators_from_mjc=sum(a.source == "mjc" for a in m.actuators),
        actuators_from_drives=sum(a.source == "drive" for a in m.actuators),
        equalities=len(m.equalities),
        tendons=len(m.tendons),
        contact_excludes=len(m.excludes),
        meshes=len(m.meshes),
        decomposed_meshes={k: len(v) for k, v in ctx.decomposed.items() if v},
        inertial_sources={
            s: sum(b.inertial is not None and b.inertial.source == s for b in m.bodies)
            for s in ("authored", "computed", "default")
        },
    )
    return m


# ----------------------------------------------------------------------------- 1. bodies


def _collect_bodies(ctx: _Ctx, parsed) -> list[str]:
    paths = []
    for path, desc in parsed.get("RigidBody", []):
        if not getattr(desc, "rigidBodyEnabled", True):
            continue
        prim = ctx.stage.GetPrimAtPath(path)
        rigid, scale = decompose(ctx.world(prim))
        ctx.body_rigid[path] = rigid
        ctx.body_scale[path] = np.abs(scale)
        ctx.body_name[path] = ctx.names("body", prim.GetName())
        paths.append(path)
    return paths


def _owner_body(prim: Usd.Prim, bodies: set[str]) -> str:
    """Nearest ancestor (or self) that is a rigid body; "" if none (static/world)."""
    p = prim
    while p and not p.IsPseudoRoot():
        if str(p.GetPath()) in bodies:
            return str(p.GetPath())
        p = p.GetParent()
    return ""


# ----------------------------------------------------------------------------- 2. joints / tree


def _collect_edges(ctx: _Ctx, parsed, bodies: list[str]) -> list[_Edge]:
    body_set = set(bodies)
    edges = []
    for kind in JOINT_KINDS:
        for path, desc in parsed.get(kind, []):
            if not getattr(desc, "jointEnabled", True):
                continue
            b0, b1 = str(desc.body0 or ""), str(desc.body1 or "")
            b0 = b0 if b0 in body_set else ""
            b1 = b1 if b1 in body_set else ""
            if not b0 and not b1:
                ctx.model.warnings.append(f"{path}: joint connects no rigid bodies, skipped")
                continue

            def frame(body: str, pos, rot):
                scale = ctx.body_scale.get(body, np.ones(3))
                p = Gf.Vec3d(*(np.asarray(pos, dtype=float) * scale))
                local = pose_matrix(p, rot)
                world = local * ctx.body_rigid[body] if body else local
                return world, (tuple(float(c) * ctx.mpu for c in p), quat_tuple(rot))

            f0, l0 = frame(b0, desc.localPose0Position, desc.localPose0Orientation)
            f1, l1 = frame(b1, desc.localPose1Position, desc.localPose1Orientation)
            edges.append(_Edge(path, kind, desc, b0, b1, f0, f1, l0, l1))
    return edges


def _joint_value_at_rest(edge: _Edge, axis: tuple) -> float:
    """Joint coordinate at the authored pose (rad or m), i.e. MJCF `ref`."""
    q0 = quat_tuple(edge.f0.ExtractRotationQuat())
    q1 = quat_tuple(edge.f1.ExtractRotationQuat())
    if edge.kind == "PrismaticJoint":
        d = np.array(edge.f1.ExtractTranslation()) - np.array(edge.f0.ExtractTranslation())
        d0 = rotate(_qconj(q0), d)
        return float(np.dot(d0, axis))
    rel = _qmul(_qconj(q0), q1)
    w, v = rel[0], np.array(rel[1:])
    ang = 2.0 * math.atan2(float(np.dot(v, axis)), w)
    return (ang + math.pi) % (2 * math.pi) - math.pi


def _build_tree(ctx: _Ctx, bodies: list[str], edges: list[_Edge]) -> None:
    m = ctx.model
    adj: dict[str, list[_Edge]] = {b: [] for b in ["", *bodies]}
    loops: list[_Edge] = []
    for e in edges:
        if e.desc.excludeFromArticulation:
            loops.append(e)
            continue
        adj[e.b0].append(e)
        adj[e.b1].append(e)

    parent_of: dict[str, tuple[str, _Edge | None]] = {}
    order: list[str] = []
    used: set[int] = set()

    def bfs(root: str) -> None:
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for e in adj[u]:
                if id(e) in used:
                    continue
                v = e.b1 if e.b0 == u else e.b0
                used.add(id(e))
                if v == "" or v in parent_of or v == root:
                    loops.append(e)  # closes a cycle
                    continue
                parent_of[v] = (u, e)
                order.append(v)
                queue.append(v)

    bfs("")
    # Components not attached to the world: root them at their first body (free unless kinematic).
    for b in bodies:
        if b not in parent_of:
            parent_of[b] = ("", None)
            order.append(b)
            bfs(b)

    for b in order:
        parent, edge = parent_of[b]
        prim = ctx.stage.GetPrimAtPath(b)
        rel = relative(ctx.body_rigid[b], ctx.body_rigid[parent]) if parent else ctx.body_rigid[b]
        pos, quat = rigid_pose(rel, ctx.mpu)
        body = Body(
            name=ctx.body_name[b],
            parent=ctx.body_name.get(parent, ""),
            pose=Pose(pos, quat),
            source_path=b,
        )
        if edge is None:
            kinematic = bool(_authored(prim, "physics:kinematicEnabled"))
            if not kinematic:
                body.joints.append(Joint(ctx.names("joint", f"{body.name}_free"), JointType.FREE))
        elif edge.kind != "FixedJoint":
            j = _make_joint(ctx, edge, flipped=(edge.b1 == parent))
            if j is not None:
                body.joints.append(j)
        m.bodies.append(body)

    for e in loops:
        _loop_equality(ctx, e)


def _make_joint(ctx: _Ctx, e: _Edge, flipped: bool) -> Joint | None:
    prim = ctx.stage.GetPrimAtPath(e.path)
    name = ctx.names("joint", prim.GetName())
    child_local = e.local0 if flipped else e.local1
    extra = _passthrough(prim, MJC_JOINT_ATTRS, MJC_JOINT_RANGES)
    for src, (dst, per_degree) in NEWTON_JOINT_ATTRS.items():
        v = _authored(prim, src)
        if v is not None and dst not in extra:
            scale = 1.0 / DEG if (per_degree and e.kind == "RevoluteJoint") else 1.0
            extra[dst] = float(v) * scale

    if e.kind in ("RevoluteJoint", "PrismaticJoint"):
        axis_tok = str(e.desc.axis)
        axis_frame = AXIS_VEC.get(axis_tok, (0.0, 0.0, 1.0))
        axis = rotate(child_local[1], axis_frame)
        if flipped:
            axis = tuple(-c for c in axis)
        revolute = e.kind == "RevoluteJoint"
        unit = DEG if revolute else ctx.mpu
        lim = _finite_limit(e.desc.limit)
        rng = (lim[0] * unit, lim[1] * unit) if lim else None
        ref = _joint_value_at_rest(e, axis_frame) * (1.0 if revolute else ctx.mpu)
        jt = JointType.HINGE if revolute else JointType.SLIDE
        j = Joint(name, jt, child_local[0], axis, rng, ref, extra, e.path)
    elif e.kind == "SphericalJoint":
        lim = _finite_limit(e.desc.limit)
        rng = None
        if lim:
            cone = min(abs(lim[0]), abs(lim[1]))
            if abs(abs(lim[0]) - abs(lim[1])) > 1e-6:
                ctx.model.warnings.append(
                    f"{e.path}: asymmetric cone limits; MuJoCo ball uses min = {cone:g} deg"
                )
            rng = (0.0, cone * DEG)
        j = Joint(name, JointType.BALL, child_local[0], (0.0, 0.0, 1.0), rng, 0.0, extra, e.path)
    else:  # D6
        ctx.model.warnings.append(f"{e.path}: D6 joints not supported yet, treated as fixed")
        return None
    ctx.joint_name[e.path] = name
    ctx.joint_type[name] = j.type
    ctx.joint_range[name] = j.range
    ctx.joint_ref[name] = j.ref
    return j


MJC_EQUALITY_ATTRS = {"mjc:solref": "solref", "mjc:solimp": "solimp"}


def _loop_equality(ctx: _Ctx, e: _Edge) -> None:
    b0 = ctx.body_name.get(e.b0, "world")
    b1 = ctx.body_name.get(e.b1, "world")
    prim = ctx.stage.GetPrimAtPath(e.path)
    name = ctx.names("equality", prim.GetName())
    soft = _passthrough(prim, MJC_EQUALITY_ATTRS, {})
    if e.kind == "FixedJoint":
        ctx.model.equalities.append(Equality("weld", name, {"body1": b1, "body2": b0, **soft}))
    elif e.kind in ("RevoluteJoint", "SphericalJoint"):
        anchor = e.local1[0] if e.b1 else e.local0[0]
        ctx.model.equalities.append(
            Equality("connect", name, {"body1": b1, "body2": b0, "anchor": list(anchor), **soft})
        )
        if e.kind == "RevoluteJoint":
            ctx.model.warnings.append(
                f"{e.path}: loop-closing hinge approximated by a point (connect) constraint"
            )
    else:
        ctx.model.warnings.append(f"{e.path}: loop-closing {e.kind} not supported, skipped")


# ----------------------------------------------------------------------------- 3. geoms


def _attach(ctx: _Ctx, owner: str, geom: Geom) -> None:
    if owner:
        ctx.model.body(ctx.body_name[owner]).geoms.append(geom)
    else:
        ctx.model.world_geoms.append(geom)


def _owner_frame(ctx: _Ctx, owner: str) -> Gf.Matrix4d:
    return ctx.body_rigid[owner] if owner else Gf.Matrix4d(1.0)


def _mesh_name(ctx: _Ctx, prim: Usd.Prim, scale: np.ndarray, verts_faces) -> str:
    src = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
    key = (str(src.GetPath()), tuple(np.round(scale, 6)))
    if key in ctx.mesh_keys:
        return ctx.mesh_keys[key]
    v, f = bake_scale(verts_faces[0], verts_faces[1], scale, ctx.mpu)
    name = ctx.names("mesh", prim.GetName())
    ctx.model.meshes[name] = MeshData(name, v.astype(np.float32), f.astype(np.int32))
    ctx.mesh_keys[key] = name
    return name


def _geom_from_prim(
    ctx: _Ctx, prim: Usd.Prim, owner: str, collision: bool
) -> tuple[Geom, trimesh.Trimesh | None] | None:
    """Build a Geom for a gprim, in the owner body's frame. Also returns a trimesh in the body
    frame (for mass computation) when the shape is closed enough to have a volume."""
    rel = relative(ctx.world(prim), _owner_frame(ctx, owner))
    rigid, scale = decompose(rel)
    pos, quat = rigid_pose(rigid, ctx.mpu)
    s = np.abs(scale) * ctx.mpu
    ns = "geom"
    gname = ctx.names(ns, prim.GetName())
    T = np.array(rigid, dtype=float).T  # row-vector -> column-vector 4x4
    T[:3, 3] *= ctx.mpu

    def axis_quat(axis: str):
        # MJCF capsules/cylinders run along local z
        if axis == "X":
            return _qmul(quat, (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))
        if axis == "Y":
            return _qmul(quat, (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0))
        return quat

    tm = None
    if prim.IsA(UsdGeom.Cube):
        half = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * s
        g = Geom(gname, ShapeType.BOX, Pose(pos, quat), tuple(half))
        tm = trimesh.creation.box(extents=2 * half)
    elif prim.IsA(UsdGeom.Sphere):
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * float(s.max())
        g = Geom(gname, ShapeType.SPHERE, Pose(pos, quat), (r,))
        tm = trimesh.creation.icosphere(subdivisions=2, radius=r)
    elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
        is_cap = prim.IsA(UsdGeom.Capsule)
        api = UsdGeom.Capsule(prim) if is_cap else UsdGeom.Cylinder(prim)
        axis = str(api.GetAxisAttr().Get() or "Z")
        ai = "XYZ".index(axis)
        r = float(api.GetRadiusAttr().Get()) * float(np.delete(s, ai).max())
        h = float(api.GetHeightAttr().Get()) * float(s[ai])
        shape = ShapeType.CAPSULE if is_cap else ShapeType.CYLINDER
        g = Geom(gname, shape, Pose(pos, axis_quat(axis)), (r, h / 2))
        tm = (
            trimesh.creation.capsule(height=h, radius=r)
            if is_cap
            else trimesh.creation.cylinder(radius=r, height=h)
        )
        if axis != "Z":  # align trimesh's z-axis primitive with the USD axis (geom-local)
            tm.apply_transform(
                trimesh.transformations.quaternion_matrix(_qmul(_qconj(quat), axis_quat(axis)))
            )
    elif prim.IsA(UsdGeom.Mesh):
        arrays = mesh_arrays(UsdGeom.Mesh(prim))
        if arrays is None:
            ctx.model.warnings.append(f"{prim.GetPath()}: empty mesh, skipped")
            return None
        if len(np.unique(arrays[0], axis=0)) < 4:
            ctx.model.warnings.append(
                f"{prim.GetPath()}: mesh has fewer than 4 vertices (MuJoCo minimum), skipped"
            )
            return None
        mname = _mesh_name(ctx, prim, scale, arrays)
        g = Geom(gname, ShapeType.MESH, Pose(pos, quat), mesh=mname)
        md = ctx.model.meshes[mname]
        if collision:
            try:
                tm = trimesh.Trimesh(md.vertices, md.faces, process=False).convex_hull
            except Exception:  # noqa: BLE001  (degenerate / flat meshes)
                tm = None
    else:
        ctx.model.warnings.append(f"{prim.GetPath()}: {prim.GetTypeName()} not supported, skipped")
        return None

    g.collision = collision
    g.source_path = str(prim.GetPath())
    if tm is not None:  # geom frame -> body frame (positions already in metres)
        tm.apply_transform(T)
    return g, tm


def _collect_colliders(ctx: _Ctx, parsed, bodies: list[str]) -> dict[str, str]:
    """Create collision geoms. Returns {collider path: owner body path}."""
    body_set = set(bodies)
    materials = {p: d for p, d in parsed.get("RigidBodyMaterial", [])}
    owners: dict[str, str] = {}
    for kind in SHAPE_KINDS:
        for path, desc in parsed.get(kind, []):
            if not getattr(desc, "collisionEnabled", True):
                continue
            prim = ctx.stage.GetPrimAtPath(path)
            rb = str(getattr(desc, "rigidBody", "") or "")
            owner = rb if rb in body_set else _owner_body(prim, body_set)
            built = _geom_from_prim(ctx, prim, owner, collision=True)
            if built is None:
                continue
            g, tm = built
            g.rgba = COLLISION_RGBA
            g.extra.update(_passthrough(prim, MJC_COLLISION_ATTRS, {}))
            if g.shape == ShapeType.MESH:
                md = ctx.model.meshes[g.mesh]
                md.maxhullvert = _authored(prim, "mjc:maxhullvert")
            mats = [str(m) for m in getattr(desc, "materials", []) or [] if str(m)]
            if mats and mats[0] in materials:
                g.friction = _friction(ctx, mats[0], materials[mats[0]])
            geoms = [g]
            if g.shape == ShapeType.MESH and ctx.decompose != "off":
                parts = _convex_parts(ctx, g.mesh)
                if parts:
                    geoms, tm = _part_geoms(ctx, g, parts)
            if tm is not None and tm.volume > 1e-12:
                ctx.mass_shapes[path] = tm
            for geom in geoms:
                _attach(ctx, owner, geom)
            owners[path] = owner
    return owners


def _convex_parts(ctx: _Ctx, mesh_name: str) -> list[str]:
    """Names of convex part meshes for a concave collision mesh ([] = keep the hull)."""
    from simready_mujoco.native.decompose import decompose, is_concave

    if mesh_name in ctx.decomposed:
        return ctx.decomposed[mesh_name]
    if ctx.decompose_only is not None and mesh_name not in ctx.decompose_only:
        ctx.decomposed[mesh_name] = []
        return []
    md = ctx.model.meshes[mesh_name]
    names: list[str] = []
    if ctx.decompose == "all" or is_concave(md.vertices, md.faces):
        try:
            level = (ctx.decompose_only or {}).get(mesh_name, 0)
            parts = decompose(md.vertices, md.faces, level)
        except ImportError:
            ctx.model.warnings.append(
                f"{mesh_name}: concave collider kept as a solid hull (install coacd: pip install coacd)"
            )
            parts = []
        for i, (pv, pf) in enumerate(parts):
            pname = ctx.names("mesh", f"{mesh_name}_cvx{i}")
            ctx.model.meshes[pname] = MeshData(
                pname, pv.astype(np.float32), pf.astype(np.int32), source=mesh_name
            )
            names.append(pname)
    ctx.decomposed[mesh_name] = names
    return names


def _part_geoms(ctx: _Ctx, g: Geom, parts: list[str]) -> tuple[list[Geom], trimesh.Trimesh | None]:
    """One geom per convex part (same pose/material as the original), plus their combined
    shape in the body frame for mass computation."""
    T = np.eye(4)
    T[:3, :3] = trimesh.transformations.quaternion_matrix(g.pose.quat)[:3, :3]
    T[:3, 3] = g.pose.pos
    geoms, hulls = [], []
    for i, pname in enumerate(parts):
        pg = Geom(
            name=ctx.names("geom", f"{g.name}_cvx{i}"),
            shape=ShapeType.MESH,
            pose=g.pose,
            mesh=pname,
            collision=True,
            rgba=g.rgba,
            friction=g.friction,
            extra=dict(g.extra),
            source_path=g.source_path,
        )
        geoms.append(pg)
        md = ctx.model.meshes[pname]
        hull = trimesh.Trimesh(md.vertices.astype(np.float64), md.faces, process=False).convex_hull
        hull.apply_transform(T)
        hulls.append(hull)
    combined = trimesh.util.concatenate(hulls) if hulls else None
    return geoms, combined


def _friction(ctx: _Ctx, mat_path: str, desc) -> tuple[float, ...]:
    prim = ctx.stage.GetPrimAtPath(mat_path)
    slide = float(desc.dynamicFriction)
    torsion = _authored(prim, "mjc:torsionalfriction")
    roll = _authored(prim, "mjc:rollingfriction")
    return (slide, float(torsion if torsion is not None else 0.005), float(roll or 0.0001))


def _input_value(shader: UsdShade.Shader, name: str):
    """Value of a shader input, following connections (e.g. to material interface inputs)."""
    inp = shader.GetInput(name)
    if not inp:
        return None
    for attr in inp.GetValueProducingAttributes():
        v = attr.Get()
        if v is not None:
            return v
    return inp.Get()


def _rgba(prim: Usd.Prim) -> tuple[float, float, float, float]:
    """Visual colour: bound material's base colour first, then displayColor, then grey."""
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if mat:
        shader = mat.ComputeSurfaceSource()[0]
        if shader:
            for name in ("diffuseColor", "diffuse_color_constant", "base_color", "baseColor"):
                c = _input_value(shader, name)
                if c is not None and not isinstance(c, str):
                    a = _input_value(shader, "opacity")
                    return (
                        float(c[0]),
                        float(c[1]),
                        float(c[2]),
                        float(a) if a is not None else 1.0,
                    )
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    if pv and pv.HasAuthoredValue():
        vals = pv.Get()
        if vals is not None and len(vals) > 0:
            c = vals[0]
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return DEFAULT_RGBA


def _collect_visuals(ctx: _Ctx, bodies: list[str], colliders: set[str]) -> None:
    body_set = set(bodies)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in ctx.stage.Traverse(predicate):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if prim.IsA(UsdGeom.Cone) or prim.IsA(UsdGeom.Plane):
            continue
        owner = _owner_body(prim, body_set)
        if prim.IsA(UsdGeom.Mesh):
            for g in _visual_mesh_geoms(ctx, prim, owner):
                _attach(ctx, owner, g)
            continue
        built = _geom_from_prim(ctx, prim, owner, collision=False)
        if built is None:
            continue
        g, _ = built
        mat = _visual_material(ctx, prim, prim, textured=False)
        g.visual_material = mat
        g.rgba = None if mat else _rgba(prim)
        _attach(ctx, owner, g)


def _visual_material(
    ctx: _Ctx, target: Usd.Prim, mesh_prim: Usd.Prim, textured: bool
) -> str | None:
    """MJCF material name for the material bound to `target` (a mesh or GeomSubset).
    textured=False asks for an untextured variant (mesh without UVs): the texture's mean
    colour is used instead."""
    from simready_mujoco.native.appearance import display_color, resolve_material

    mat, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
    if not mat and target != mesh_prim:
        mat, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    if mat:
        key = str(mat.GetPath())
        spec = None
        if key not in ctx.visual_materials:
            spec = resolve_material(mat, ctx.names("material", mat.GetPrim().GetName()))
            ctx.visual_materials[key] = spec.name
            shin = None if spec.roughness is None else float(np.clip(1.0 - spec.roughness, 0, 1))
            spec_level = None if spec.specular is None else float(np.clip(spec.specular, 0, 1))
            ctx.model.visual_materials[spec.name] = VisualMaterial(
                spec.name, spec.rgba, spec.texture, spec_level, shin
            )
            ctx.visual_materials[key + "#uv"] = spec.uv_primvar
        name = ctx.visual_materials[key]
        vm = ctx.model.visual_materials[name]
        if vm.texture_src and not textured:
            flat_key = key + "#flat"
            if flat_key not in ctx.visual_materials:
                flat = ctx.names("material", f"{name}_flat")
                ctx.visual_materials[flat_key] = flat
                ctx.model.visual_materials[flat] = VisualMaterial(
                    flat, _mean_color(vm.texture_src, vm.rgba), None, vm.specular, vm.shininess
                )
            return ctx.visual_materials[flat_key]
        return name
    dc = display_color(mesh_prim)
    if dc is None:
        return None
    key = f"displayColor:{tuple(round(c, 4) for c in dc)}"
    if key not in ctx.visual_materials:
        name = ctx.names("material", "display_color")
        ctx.visual_materials[key] = name
        ctx.model.visual_materials[name] = VisualMaterial(name, dc)
    return ctx.visual_materials[key]


def _mean_color(path: str, fallback) -> tuple[float, float, float, float]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            c = np.asarray(im.convert("RGB").resize((16, 16))).reshape(-1, 3).mean(0) / 255.0
        return (float(c[0]), float(c[1]), float(c[2]), float(fallback[3]))
    except Exception:  # noqa: BLE001
        return tuple(fallback)


def _visual_mesh_geoms(ctx: _Ctx, prim: Usd.Prim, owner: str) -> list[Geom]:
    """One visual geom per material on the mesh (GeomSubsets), with UVs when textured."""
    from simready_mujoco.native.appearance import corner_uvs, material_groups, topology, weld

    mesh = UsdGeom.Mesh(prim)
    topo = topology(mesh)
    if topo is None:
        ctx.model.warnings.append(f"{prim.GetPath()}: empty mesh, skipped")
        return []
    rel = relative(ctx.world(prim), _owner_frame(ctx, owner))
    rigid, scale = decompose(rel)
    pos, quat = rigid_pose(rigid, ctx.mpu)
    src = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
    n_faces = len(mesh.GetFaceVertexCountsAttr().Get())
    groups = material_groups(mesh, n_faces)
    uv_cache: dict[str, np.ndarray | None] = {}
    geoms = []
    for gi, (bind_prim, face_ids) in enumerate(groups):
        mask = slice(None) if face_ids is None else np.isin(topo.tri_face, face_ids)
        tris = topo.tris[mask]
        if len(tris) == 0:
            continue
        mat = _visual_material(ctx, bind_prim, prim, textured=True)
        vm = ctx.model.visual_materials.get(mat) if mat else None
        uvs = None
        if vm is not None and vm.texture_src:
            key = next(k for k, v in ctx.visual_materials.items() if v == mat)
            uv_name = ctx.visual_materials.get(key + "#uv", "st")
            if uv_name not in uv_cache:
                uv_cache[uv_name] = corner_uvs(prim, uv_name, topo)
            if uv_cache[uv_name] is not None:
                uvs = uv_cache[uv_name][mask]
            else:
                mat = _visual_material(ctx, bind_prim, prim, textured=False)
        mesh_key = (str(src.GetPath()), tuple(np.round(scale, 6)), gi, mat, uvs is not None)
        if mesh_key not in ctx.mesh_keys:
            v, f, tc = weld(topo.points, tris, uvs)
            if len(np.unique(v, axis=0)) < 4:
                ctx.model.warnings.append(
                    f"{prim.GetPath()}: mesh part has fewer than 4 vertices (MuJoCo minimum), skipped"
                )
                continue
            v, f = bake_scale(v, f, scale, ctx.mpu)
            suffix = f"_{gi}" if len(groups) > 1 else ""
            mname = ctx.names("mesh", f"{prim.GetName()}{suffix}")
            ctx.model.meshes[mname] = MeshData(
                mname,
                v.astype(np.float32),
                f.astype(np.int32),
                texcoords=None if tc is None else tc.astype(np.float32),
            )
            ctx.mesh_keys[mesh_key] = mname
        g = Geom(
            ctx.names("geom", f"{prim.GetName()}{'_' + str(gi) if len(groups) > 1 else ''}"),
            ShapeType.MESH,
            Pose(pos, quat),
            mesh=ctx.mesh_keys[mesh_key],
            collision=False,
            visual_material=mat,
            rgba=None if mat else DEFAULT_RGBA,
            source_path=str(prim.GetPath()),
        )
        geoms.append(g)
    return geoms


# ----------------------------------------------------------------------------- 4. inertia


def _compute_inertials(ctx: _Ctx, bodies: list[str]) -> None:
    def mass_info(prim: Usd.Prim):
        mi = UsdPhysics.RigidBodyAPI.MassInformation()
        tm = ctx.mass_shapes.get(str(prim.GetPath()))
        mi.localPos = Gf.Vec3f(0.0)
        mi.localRot = Gf.Quatf(1.0)
        if tm is None:
            mi.volume = 0.0
            mi.inertia = Gf.Matrix3f(0.0)
            mi.centerOfMass = Gf.Vec3f(0.0)
            return mi
        # Values in stage units, unit density, inertia about the shape's own COM.
        inv = 1.0 / ctx.mpu
        tm.density = 1.0
        mi.volume = float(tm.volume) * inv**3
        mi.centerOfMass = Gf.Vec3f(*(tm.center_mass * inv))
        mi.inertia = Gf.Matrix3f(*(np.asarray(tm.moment_inertia) * inv**5).flatten().tolist())
        return mi

    for b in bodies:
        prim = ctx.stage.GetPrimAtPath(b)
        body = ctx.model.body(ctx.body_name[b])
        authored = prim.HasAPI(UsdPhysics.MassAPI) and (
            _authored(prim, "physics:mass") is not None
            and _authored(prim, "physics:diagonalInertia") is not None
        )
        try:
            mass, diag, com, axes = UsdPhysics.RigidBodyAPI(prim).ComputeMassProperties(mass_info)
        except Exception as e:  # noqa: BLE001
            ctx.model.warnings.append(f"{b}: mass computation failed ({e})")
            mass, diag, com, axes = 0.0, Gf.Vec3f(0), Gf.Vec3f(0), Gf.Quatf(1)
        mass = float(mass) * ctx.kpu
        diag_v = [float(x) * ctx.kpu * ctx.mpu**2 for x in diag]
        if not math.isfinite(mass) or mass <= 0:
            if not _moves(ctx.model, body.name):
                continue  # welded to the world: MuJoCo allows massless static bodies
            ctx.model.warnings.append(f"{body.name}: no mass information, using {MIN_MASS} kg")
            body.inertial = Inertial(MIN_MASS, diaginertia=(MIN_INERTIA,) * 3, source="default")
            continue
        diag_v = [d if math.isfinite(d) and d > 0 else MIN_INERTIA for d in diag_v]
        body.inertial = Inertial(
            mass=mass,
            pos=tuple(float(c) * ctx.mpu for c in com),
            diaginertia=tuple(diag_v),
            quat=quat_tuple(axes),
            source="authored" if authored else "computed",
        )


def _moves(model: Model, body_name: str) -> bool:
    """True if the body or any ancestor has a joint (i.e. it is not welded to the world)."""
    name = body_name
    while name:
        b = model.body(name)
        if b.joints:
            return True
        name = b.parent
    return False


# ----------------------------------------------------------------------------- 5. actuators


MJC_TARGET = "mjc:target"


def _collect_tendons(ctx: _Ctx) -> dict[str, str]:
    """MjcTendon prims -> fixed tendons. Returns {USD path: MJCF tendon name}."""
    names: dict[str, str] = {}
    for prim in ctx.stage.Traverse():
        if prim.GetTypeName() != "MjcTendon":
            continue
        kind = str(_authored(prim, "mjc:type") or "spatial")
        if kind != "fixed":
            ctx.model.warnings.append(
                f"{prim.GetPath()}: {kind} tendons not supported yet, skipped"
            )
            continue
        rel = prim.GetRelationship("mjc:path")
        targets = [str(p) for p in rel.GetTargets()] if rel else []
        coefs = list(_authored(prim, "mjc:path:coef") or [1.0] * len(targets))
        indices = list(_authored(prim, "mjc:path:indices") or range(len(coefs)))
        joints = []
        for k, idx in enumerate(indices):
            jname = ctx.joint_name.get(targets[idx]) if idx < len(targets) else None
            if not jname:
                ctx.model.warnings.append(f"{prim.GetPath()}: tendon joint {idx} not converted")
                continue
            joints.append((jname, float(coefs[k]) if k < len(coefs) else 1.0))
        if not joints:
            continue
        name = ctx.names("tendon", prim.GetName())
        attrs = _passthrough(prim, MJC_TENDON_ATTRS, MJC_TENDON_RANGES)
        ctx.model.tendons.append(Tendon(name, joints, attrs))
        names[str(prim.GetPath())] = name
    return names


ACTUATOR_SOURCES = ("auto", "mjc", "drive")


def _drive_actuators(ctx: _Ctx, parsed) -> dict[str, tuple[dict[str, Any], str, float]]:
    """UsdPhysics drives as PD actuators: {joint name: (MJCF attrs, USD joint path, kp)}."""
    out: dict[str, tuple[dict[str, Any], str, float]] = {}
    for kind in ("RevoluteJoint", "PrismaticJoint"):
        for path, desc in parsed.get(kind, []):
            jname = ctx.joint_name.get(path)
            drive = getattr(desc, "drive", None)
            if not jname or drive is None or not drive.enabled:
                continue
            kp, kv = float(drive.stiffness), float(drive.damping)
            if kp <= 0 and kv <= 0:
                continue
            if kind == "RevoluteJoint":  # USD angular gains are per degree
                kp, kv = kp / DEG, kv / DEG
            else:
                kp, kv = kp / ctx.mpu, kv / ctx.mpu
            if drive.acceleration:
                ctx.model.warnings.append(
                    f"{path}: acceleration-type drive converted as force-type (gains not scaled)"
                )
            attrs: dict[str, Any] = {"gaintype": "fixed", "biastype": "affine"}
            if kp > 0:
                attrs["gainprm"] = [kp]
                attrs["biasprm"] = [0.0, -kp, -kv]
                rng = ctx.joint_range.get(jname)
                if rng:
                    attrs["ctrlrange"] = list(rng)
            else:  # velocity drive
                attrs["gainprm"] = [kv]
                attrs["biasprm"] = [0.0, 0.0, -kv]
            fl = float(drive.forceLimit)
            if math.isfinite(fl) and 0 < fl < 1e30:
                attrs["forcerange"] = [-fl, fl]
            out[jname] = (attrs, path, kp)
    return out


def _lacks_position_feedback(attrs: dict[str, Any]) -> bool:
    """True if an MjcActuator cannot hold a position (no affine -kp*q bias term)."""
    if attrs.get("biastype") != "affine":
        return True
    bias = attrs.get("biasprm") or [0.0, 0.0, 0.0]
    return len(bias) < 2 or abs(float(bias[1])) < 1e-9


def _collect_actuators(ctx: _Ctx, parsed, tendons: dict[str, str], source: str = "auto") -> None:
    """Actuators from MjcActuator prims and/or UsdPhysics drives.

    source: "mjc"   MjcActuator wins wherever it exists (drives only fill the gaps)
            "drive" UsdPhysics drives win wherever they exist
            "auto"  like "mjc", except when an MjcActuator has no position feedback but the
                    joint has a stiff position drive (seen in SimReady FR3: MjcActuator gains of
                    0.003, drive gains matching Menagerie); then the drive is used and we warn.
    """
    if source not in ACTUATOR_SOURCES:
        raise ValueError(f"actuator source must be one of {ACTUATOR_SOURCES}")
    drives = _drive_actuators(ctx, parsed)
    driven: set[str] = set()
    for prim in ctx.stage.Traverse():
        if prim.GetTypeName() != "MjcActuator":
            continue
        rel = prim.GetRelationship(MJC_TARGET)
        targets = rel.GetTargets() if rel else []
        tpath = str(targets[0]) if targets else ""
        attrs = _passthrough(prim, MJC_ACTUATOR_ATTRS, MJC_ACTUATOR_RANGES)
        if tpath in ctx.joint_name:
            jname = ctx.joint_name[tpath]
            drive = drives.get(jname)
            use_drive = drive is not None and (
                source == "drive"
                or (source == "auto" and drive[2] > 0 and _lacks_position_feedback(attrs))
            )
            if use_drive:
                if source == "auto":
                    ctx.model.warnings.append(
                        f"{prim.GetPath()}: MjcActuator has no position feedback "
                        f"(gainprm={attrs.get('gainprm', [1])[:1]}, biasprm={attrs.get('biasprm', [])[:3]}) "
                        f"but the joint has a position drive (kp={drive[2]:.4g}); using the drive"
                    )
                continue  # the drive is added below
            aname = ctx.names("actuator", prim.GetName())
            ctx.model.actuators.append(Actuator(aname, jname, attrs, "mjc"))
            driven.add(jname)
        elif tpath in tendons:
            aname = ctx.names("actuator", prim.GetName())
            ctx.model.actuators.append(Actuator(aname, tendons[tpath], attrs, "mjc", "tendon"))
        else:
            ctx.model.warnings.append(
                f"{prim.GetPath()}: actuator target '{tpath}' is not a converted joint or tendon, skipped"
            )

    for jname, (attrs, path, _) in drives.items():
        if jname in driven:
            continue
        prim = ctx.stage.GetPrimAtPath(path)
        ctx.model.actuators.append(
            Actuator(ctx.names("actuator", f"{prim.GetName()}_drive"), jname, attrs, "drive")
        )


# ----------------------------------------------------------------------------- 6. mimic


def _collect_mimics(ctx: _Ctx) -> None:
    for jpath, follower in ctx.joint_name.items():
        prim = ctx.stage.GetPrimAtPath(jpath)
        schemas = applied_schemas(prim)
        rot = ctx.joint_type.get(follower) == JointType.HINGE
        unit = DEG if rot else ctx.mpu

        rel = prim.GetRelationship("newton:mimicJoint")
        if rel and rel.GetTargets():
            leader = ctx.joint_name.get(str(rel.GetTargets()[0]))
            c0 = _authored(prim, "newton:mimicCoef0")
            c0 = c0 if c0 is not None else _authored(prim, "mjc:coef0") or 0.0
            c1 = _authored(prim, "newton:mimicCoef1")
            c1 = c1 if c1 is not None else _authored(prim, "mjc:coef1")
            c1 = 1.0 if c1 is None else c1
            higher = [float(_authored(prim, f"mjc:coef{i}") or 0.0) for i in (2, 3, 4)]
            _add_mimic(ctx, follower, leader, [float(c0) * unit, float(c1), *higher], jpath)
            continue

        for s in schemas:
            if not s.startswith("PhysxMimicJointAPI:"):
                continue
            inst = s.split(":", 1)[1]
            ref_rel = prim.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
            targets = ref_rel.GetTargets() if ref_rel else []
            leader = ctx.joint_name.get(str(targets[0])) if targets else None
            g = float(_authored(prim, f"physxMimicJoint:{inst}:gearing") or 0.0)
            o = float(_authored(prim, f"physxMimicJoint:{inst}:offset") or 0.0) * unit
            coefs = _physx_mimic_coefs(ctx, follower, leader, g, o, jpath)
            _add_mimic(ctx, follower, leader, [*coefs, 0.0, 0.0, 0.0], jpath)


def _physx_mimic_coefs(ctx, follower, leader, gearing, offset, path) -> list[float]:
    """PhysX: q + gearing*q_ref + offset = 0. Its sign convention is not consistently followed
    by assets, so pick the convention whose mapped leader range fits the follower's limits."""
    physx = [-offset, -gearing]
    alt = [offset, gearing]
    lr, fr = ctx.joint_range.get(leader), ctx.joint_range.get(follower)
    if not lr or not fr:
        return physx

    def fits(c):
        lo, hi = sorted((c[0] + c[1] * lr[0], c[0] + c[1] * lr[1]))
        tol = 1e-3 + 0.02 * (fr[1] - fr[0])
        return lo >= fr[0] - tol and hi <= fr[1] + tol

    if fits(physx):
        return physx
    if fits(alt):
        ctx.model.warnings.append(
            f"{path}: mimic gearing sign flipped to fit joint limits (authored for the other convention)"
        )
        return alt
    ctx.model.warnings.append(
        f"{path}: mimic coupling exceeds follower limits under both conventions"
    )
    return physx


def _add_mimic(ctx: _Ctx, follower: str, leader: str | None, polycoef: list[float], path) -> None:
    """polycoef is in absolute joint coordinates (f = c0 + c1*l + ...). MuJoCo's joint equality
    is relative to each joint's reference: f - f0 = a0 + a1*(l - l0) + ..., so re-base it."""
    if not leader:
        ctx.model.warnings.append(f"{path}: mimic reference joint not found, skipped")
        return
    f0, l0 = ctx.joint_ref.get(follower, 0.0), ctx.joint_ref.get(leader, 0.0)
    if f0 or l0:
        if any(abs(c) > 0 for c in polycoef[2:]):
            ctx.model.warnings.append(
                f"{path}: non-linear mimic with non-zero joint reference; coupling not re-based"
            )
        else:
            polycoef = [polycoef[0] - f0 + polycoef[1] * l0, polycoef[1], 0.0, 0.0, 0.0]
    name = ctx.names("equality", f"{follower}_mimic")
    soft = _passthrough(ctx.stage.GetPrimAtPath(path), MJC_EQUALITY_ATTRS, {})
    ctx.model.equalities.append(
        Equality(
            "joint", name, {"joint1": follower, "joint2": leader, "polycoef": polycoef, **soft}
        )
    )


# ----------------------------------------------------------------------------- collision filtering


def _collect_excludes(ctx: _Ctx, parsed, edges: list[_Edge]) -> None:
    """USD joints don't let their two bodies collide unless physics:collisionEnabled is true,
    and FilteredPairsAPI disables pairs explicitly. MuJoCo only filters parent/child pairs
    (and not when the parent is welded to the world), so emit explicit excludes."""
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        na, nb = ctx.body_name.get(a), ctx.body_name.get(b)
        if na and nb and na != nb:
            pairs.add(tuple(sorted((na, nb))))

    for e in edges:
        if not getattr(e.desc, "collisionEnabled", False):
            add(e.b0, e.b1)
    for path, desc in parsed.get("RigidBody", []):
        for other in getattr(desc, "filteredCollisions", []) or []:
            add(path, str(other))
    ctx.model.excludes = sorted(pairs)


# ----------------------------------------------------------------------------- 7. scene


def _collect_scene(ctx: _Ctx, parsed) -> None:
    scenes = parsed.get("Scene", [])
    if not scenes:
        return
    path, desc = scenes[0]
    mag = float(desc.gravityMagnitude)
    d = np.array(desc.gravityDirection, dtype=float)
    if math.isfinite(mag) and mag > 0 and np.linalg.norm(d) > 0:
        ctx.model.gravity = tuple(float(c) for c in d / np.linalg.norm(d) * mag * ctx.mpu)
    prim = ctx.stage.GetPrimAtPath(path)
    for a in prim.GetAuthoredAttributes():
        n = a.GetName()
        if n.startswith("mjc:option:"):
            ctx.model.option[n.split(":", 2)[2]] = _plain(a.Get())
        elif n.startswith("mjc:flag:"):
            ctx.model.flags[n.split(":", 2)[2]] = "enable" if a.Get() else "disable"


__all__ = ["parse_stage", "IDENTITY_QUAT"]
