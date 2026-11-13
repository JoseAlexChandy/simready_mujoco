"""Add joints to assets that ship as one rigid piece (a microwave whose door is just a mesh).

Many SimReady props model their moving parts as separate meshes but author no joints, so the
whole object is one rigid body. An `articulation:` list in the asset config names those parts
and how they move; `apply_articulation` turns each into a rigid body with a revolute or
prismatic UsdPhysics joint to the object, authored on the session layer. The files on disk are
never touched, and the rest of the pipeline sees an ordinary articulated asset.

Every position and axis is given in the asset's own frame (its default prim), which is what
`srmj parts` prints, so a spec reads the same whatever the asset's up axis or placement:

    articulation:
      - part: door              # prim name or path (relative to the default prim)
        joint: hinge            # hinge | slide
        axis: -z                # hinge: rotation axis (right-hand rule, + opens)
                                # slide: direction the part moves when opening
        at: [-x, +y]            # hinge line: the part's bounding-box edge at min x, max y
        range: [0, 110]         # degrees for a hinge, metres for a slide
        spring: 0.5             # optional: springs back to closed (N·m/rad or N/m)
        damping: 0.2            # optional
        mass: 0.6               # optional, kg (default: share of the object's mass by volume)
        colliders: [door_col]   # optional: extra prims that move with the part
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

OWNER_ATTR = "srmj:body"  # forces a prim (and its children) onto a body: see native.parse
AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class PartJoint:
    part: str
    joint: str = "hinge"
    axis: str = "z"
    at: list[str] = field(default_factory=list)
    range: tuple[float, float] | None = None
    spring: float | None = None
    damping: float | None = None
    mass: float | None = None
    colliders: list[str] = field(default_factory=list)
    parent: str | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PartJoint:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"articulation: unknown keys {sorted(unknown)} (allowed: {sorted(known)})"
            )
        d = dict(d)
        if isinstance(d.get("at"), str):
            d["at"] = [d["at"]]
        if isinstance(d.get("colliders"), str):
            d["colliders"] = [d["colliders"]]
        if d.get("range") is not None:
            lo, hi = d["range"]
            d["range"] = (float(lo), float(hi))
        spec = cls(**d)
        if spec.joint not in ("hinge", "slide"):
            raise ValueError(f"articulation {spec.part}: joint must be hinge or slide")
        return spec


def parse_axis(token: str) -> np.ndarray:
    """'x' | '+y' | '-z' -> unit vector."""
    t = str(token).strip().lower()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    if t not in AXES:
        raise ValueError(f"axis must be x, y or z with an optional sign, got '{token}'")
    v = np.zeros(3)
    v[AXES[t]] = sign
    return v


def resolve_prim(stage: Usd.Stage, root: Usd.Prim, ref: str) -> Usd.Prim:
    """A prim given by absolute path, path relative to the default prim, or unique name."""
    for path in (ref, f"{root.GetPath()}/{ref.lstrip('/')}"):
        if Sdf.Path.IsValidPathString(path) and path.startswith("/"):
            p = stage.GetPrimAtPath(path)
            if p:
                return p
    matches = [p for p in Usd.PrimRange(root) if p.GetName() == ref]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"articulation: no prim named '{ref}' (run `srmj parts` to list them)")
    raise ValueError(
        f"articulation: '{ref}' matches {len(matches)} prims, use a path: "
        + ", ".join(str(m.GetPath()) for m in matches[:6])
    )


def _bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True
    )


def local_bbox(prim: Usd.Prim, root: Usd.Prim, cache=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned bounds of `prim` in `root`'s frame (stage units), or None if empty."""
    cache = cache or _bbox_cache()
    rng = cache.ComputeRelativeBound(prim, root).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return np.array(rng.GetMin()), np.array(rng.GetMax())


def _parent_body(prim: Usd.Prim) -> Usd.Prim | None:
    p = prim.GetParent()
    while p and not p.IsPseudoRoot():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def _rotation_x_to(v: np.ndarray) -> Gf.Quatd:
    """Quaternion turning +X onto v."""
    return Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*v.tolist())).GetQuat()


def _rigid_quat(m: Gf.Matrix4d) -> Gf.Quatd:
    return Gf.Matrix4d(m).RemoveScaleShear().ExtractRotationQuat()


def _hull_volume(prims: list[Usd.Prim], root: Usd.Prim) -> float:
    """Convex-hull volume of the mesh points under `prims`, in root-frame units."""
    import trimesh  # noqa: PLC0415

    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
    pts = []
    for top in prims:
        for p in Usd.PrimRange(top):
            if not p.IsA(UsdGeom.Mesh):
                continue
            raw = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not raw:
                continue
            m = cache.GetLocalToWorldTransform(p) * root_inv
            a = np.asarray(raw, dtype=float)
            r = np.array([[m[i][j] for j in range(3)] for i in range(3)])
            pts.append(a @ r + np.array(m.ExtractTranslation()))
    if not pts:
        return 0.0
    try:
        return float(trimesh.convex.convex_hull(np.vstack(pts)).volume)
    except Exception:  # noqa: BLE001  (degenerate / flat point sets)
        return 0.0


def apply_articulation(stage: Usd.Stage, specs: list[dict | PartJoint]) -> list[dict]:
    """Author the joints described by `specs` on the session layer. Returns one summary per
    joint (part, joint path, anchor and axis in the asset frame, mass given)."""
    if not specs:
        return []
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("articulation needs an asset with a default prim")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    kpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    stage.SetEditTarget(stage.GetSessionLayer())
    xc = UsdGeom.XformCache()
    bbc = _bbox_cache()
    root_world = xc.GetLocalToWorldTransform(root)
    parsed = [s if isinstance(s, PartJoint) else PartJoint.from_dict(s) for s in specs]

    # Bounds and volumes are measured before anything is changed.
    plan = []
    for spec in parsed:
        part = resolve_prim(stage, root, spec.part)
        if part.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"articulation: {part.GetPath()} is already a rigid body")
        parent = resolve_prim(stage, root, spec.parent) if spec.parent else _parent_body(part)
        if parent is None:
            raise ValueError(
                f"articulation: {part.GetPath()} has no rigid-body ancestor to hinge on; "
                "give `parent:`"
            )
        extra = [resolve_prim(stage, root, c) for c in spec.colliders]
        box = local_bbox(part, root, bbc)
        if box is None:
            raise ValueError(f"articulation: {part.GetPath()} has no geometry")
        plan.append((spec, part, parent, extra, box))

    total_mass = {}
    for _, _, parent, _, _ in plan:
        m = parent.GetAttribute("physics:mass")
        if m and m.HasAuthoredValue() and float(m.Get()) > 0:
            total_mass[str(parent.GetPath())] = float(m.Get())
    parent_volume = {p: _hull_volume([stage.GetPrimAtPath(p)], root) for p in total_mass}

    out = []
    names: set[str] = set()
    for spec, part, parent, extra, (lo, hi) in plan:
        axis = parse_axis(spec.axis)
        k = int(np.argmax(np.abs(axis)))
        anchor = (lo + hi) / 2.0
        if spec.joint == "hinge":
            for tok in spec.at:
                t = str(tok).strip().lower()
                i = AXES.get(t.lstrip("+-"))
                if i is None:
                    raise ValueError(f"articulation {spec.part}: `at` items look like -x, +y, z")
                if i == k:
                    raise ValueError(f"articulation {spec.part}: `at` can't name the hinge axis")
                if t.startswith("-"):
                    anchor[i] = lo[i]
                elif t.startswith("+"):
                    anchor[i] = hi[i]

        # asset frame -> world
        p_world = root_world.Transform(Gf.Vec3d(*anchor.tolist()))
        a_world = np.array(root_world.TransformDir(Gf.Vec3d(*axis.tolist())))
        a_world /= np.linalg.norm(a_world)
        q_joint = _rotation_x_to(a_world)

        for body in (part,):
            UsdPhysics.RigidBodyAPI.Apply(body)
        for c in extra:
            c.CreateAttribute(OWNER_ATTR, Sdf.ValueTypeNames.String, custom=True).Set(
                str(part.GetPath())
            )

        base = spec.name or f"{part.GetName()}_{spec.joint}"
        jname = base
        n = 1
        while jname in names or part.GetChild(jname):
            n += 1
            jname = f"{base}_{n}"
        names.add(jname)
        jpath = part.GetPath().AppendChild(jname)
        if spec.joint == "hinge":
            joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        else:
            joint = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([part.GetPath()])
        joint.CreateCollisionEnabledAttr(False)  # the part sits inside its parent's hull
        for side, body in ((0, parent), (1, part)):
            w = xc.GetLocalToWorldTransform(body)
            lp = w.GetInverse().Transform(p_world)
            lq = _rigid_quat(w).GetInverse() * q_joint
            getattr(joint, f"CreateLocalPos{side}Attr")(Gf.Vec3f(lp))
            getattr(joint, f"CreateLocalRot{side}Attr")(Gf.Quatf(lq))
        if spec.range is not None:
            lo_r, hi_r = sorted(spec.range)
            if spec.joint == "slide":
                lo_r, hi_r = lo_r / mpu, hi_r / mpu
            joint.CreateLowerLimitAttr(float(lo_r))
            joint.CreateUpperLimitAttr(float(hi_r))
        jp = joint.GetPrim()
        if spec.spring is not None:
            jp.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Double).Set(float(spec.spring))
        if spec.damping is not None:
            jp.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double).Set(float(spec.damping))

        mass = spec.mass
        source = "given" if mass is not None else None
        pkey = str(parent.GetPath())
        if mass is None and pkey in total_mass and parent_volume.get(pkey, 0) > 0:
            frac = _hull_volume([part, *extra], root) / parent_volume[pkey]
            frac = min(max(frac, 0.0), 0.5)
            mass = total_mass[pkey] * kpu * frac
            source = f"{frac:.0%} of the object's authored mass, by volume"
        if mass is not None:
            UsdPhysics.MassAPI.Apply(part).CreateMassAttr(float(mass) / kpu)
            if pkey in total_mass:
                left = total_mass[pkey] - float(mass) / kpu
                total_mass[pkey] = max(left, 0.1 * total_mass[pkey])
                parent.GetAttribute("physics:mass").Set(total_mass[pkey])

        out.append(
            {
                "part": str(part.GetPath()),
                "joint": str(jpath),
                "type": spec.joint,
                "anchor_m": [round(float(c) * mpu, 4) for c in anchor],
                "axis": spec.axis,
                "range": list(spec.range) if spec.range else None,
                "mass_kg": round(mass, 4) if mass is not None else None,
                "mass_source": source or "computed from colliders",
            }
        )
    return out


def describe_parts(stage: Usd.Stage, max_depth: int = 4) -> list[dict]:
    """Rows for `srmj parts`: prims under the default prim that have geometry (bounds in the
    asset frame, metres, and physics role) plus authored joints."""
    from pxr import UsdShade  # noqa: PLC0415

    root = stage.GetDefaultPrim()
    if not root:
        return []
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    bbc = _bbox_cache()
    depth0 = root.GetPath().pathElementCount
    rows = []
    rng = Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    it = iter(rng)
    for p in it:
        depth = p.GetPath().pathElementCount - depth0
        rel = "." if p == root else str(p.GetPath().MakeRelativePath(root.GetPath()))
        if p.IsA(UsdShade.Material) or p.IsA(UsdShade.Shader):
            it.PruneChildren()
            continue
        if p.IsA(UsdPhysics.Joint):
            rows.append({"path": rel, "depth": depth, "type": p.GetTypeName(), "role": "joint"})
            continue
        if not p.IsA(UsdGeom.Imageable):
            continue
        box = local_bbox(p, root, bbc)
        if box is None:
            it.PruneChildren()
            continue
        roles = []
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            roles.append("body")
        if p.HasAPI(UsdPhysics.CollisionAPI):
            roles.append("collider")
        if UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.guide:
            roles.append("guide")
        lo, hi = box[0] * mpu, box[1] * mpu
        rows.append(
            {
                "path": rel,
                "depth": depth,
                "type": p.GetTypeName(),
                "role": ",".join(roles),
                "center": ((lo + hi) / 2).round(3).tolist(),
                "size": (hi - lo).round(3).tolist(),
            }
        )
        if depth >= max_depth:
            it.PruneChildren()
    return rows
