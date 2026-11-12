"""Report which physics properties a USD asset authors.

This is step 1 of the pipeline: before converting, know what's in the asset and
which of those properties the current converter would drop.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom, UsdPhysics

from simready_mujoco.usd.physics import parse_physics

JOINT_KINDS = (
    "RevoluteJoint",
    "PrismaticJoint",
    "SphericalJoint",
    "FixedJoint",
    "D6Joint",
    "DistanceJoint",
    "CustomJoint",
)
SHAPE_KINDS = (
    "CubeShape",
    "SphereShape",
    "CapsuleShape",
    "Capsule1Shape",
    "CylinderShape",
    "Cylinder1Shape",
    "ConeShape",
    "PlaneShape",
    "MeshShape",
    "CustomShape",
)


@dataclass
class JointInfo:
    path: str
    type: str
    body0: str
    body1: str
    axis: str | None = None
    limits: tuple[float, float] | None = None  # as authored (degrees for angular)
    drive: dict[str, Any] | None = None
    exclude_from_articulation: bool = False
    mimic: bool = False


@dataclass
class BodyInfo:
    path: str
    mass: float | None = None  # authored on the body itself
    density: float | None = None
    com_authored: bool = False
    inertia_authored: bool = False
    collider_count: int = 0


@dataclass
class InspectionReport:
    file: str
    up_axis: str
    meters_per_unit: float
    kilograms_per_unit: float
    has_physics_scene: bool
    articulations: list[str] = field(default_factory=list)
    bodies: list[BodyInfo] = field(default_factory=list)
    joints: list[JointInfo] = field(default_factory=list)
    collider_shapes: dict[str, int] = field(default_factory=dict)
    mesh_approximations: dict[str, int] = field(default_factory=dict)
    materials: list[dict[str, Any]] = field(default_factory=list)
    physx_schemas: list[str] = field(default_factory=list)
    dropped_by_baseline: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _authored(attr: Usd.Attribute) -> Any | None:
    return attr.Get() if attr and attr.HasAuthoredValue() else None


def _finite(x: float | None) -> float | None:
    if x is None or not math.isfinite(x) or abs(x) > 1e30:
        return None
    return float(x)


def _joint_info(path: str, kind: str, desc: Any, stage: Usd.Stage) -> JointInfo:
    info = JointInfo(
        path=path,
        type=kind.removesuffix("Joint"),
        body0=str(getattr(desc, "body0", "") or ""),
        body1=str(getattr(desc, "body1", "") or ""),
        exclude_from_articulation=bool(getattr(desc, "excludeFromArticulation", False)),
    )
    axis = getattr(desc, "axis", None)
    if axis is not None:
        info.axis = str(axis)
    limit = getattr(desc, "limit", None)
    if limit is not None and getattr(limit, "enabled", True):
        lo, hi = _finite(limit.lower), _finite(limit.upper)
        if lo is not None or hi is not None:
            info.limits = (lo, hi)
    drive = getattr(desc, "drive", None)
    if drive is not None and drive.enabled:
        info.drive = {
            "stiffness": drive.stiffness,
            "damping": drive.damping,
            "max_force": _finite(drive.forceLimit),
            "target_position": drive.targetPosition,
            "target_velocity": drive.targetVelocity,
            "acceleration": bool(drive.acceleration),
        }
    prim = stage.GetPrimAtPath(path)
    info.mimic = any("Mimic" in s for s in prim.GetAppliedSchemas())
    return info


def _body_info(path: str, desc: Any, stage: Usd.Stage) -> BodyInfo:
    prim = stage.GetPrimAtPath(path)
    info = BodyInfo(path=path, collider_count=len(getattr(desc, "collisions", []) or []))
    if prim.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(prim)
        info.mass = _authored(m.GetMassAttr())
        info.density = _authored(m.GetDensityAttr())
        info.com_authored = _authored(m.GetCenterOfMassAttr()) is not None
        info.inertia_authored = _authored(m.GetDiagonalInertiaAttr()) is not None
    return info


def inspect_usd(path: str | Path) -> InspectionReport:
    """Open a USD file and report the physics it authors."""
    path = Path(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD file: {path}")

    parsed = parse_physics(stage)

    report = InspectionReport(
        file=str(path),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
        kilograms_per_unit=UsdPhysics.GetStageKilogramsPerUnit(stage),
        has_physics_scene=bool(parsed.get("Scene")),
    )
    report.articulations = [p for p, _ in parsed.get("Articulation", [])]
    report.bodies = [_body_info(p, d, stage) for p, d in parsed.get("RigidBody", [])]
    for kind in JOINT_KINDS:
        report.joints += [_joint_info(p, kind, d, stage) for p, d in parsed.get(kind, [])]

    shapes: Counter[str] = Counter()
    approx: Counter[str] = Counter()
    for kind in SHAPE_KINDS:
        for p, _ in parsed.get(kind, []):
            shapes[kind.removesuffix("Shape")] += 1
            if kind == "MeshShape":
                prim = stage.GetPrimAtPath(p)
                a = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                approx[str(a or "none")] += 1
    report.collider_shapes = dict(shapes)
    report.mesh_approximations = dict(approx)

    for p, d in parsed.get("RigidBodyMaterial", []):
        report.materials.append(
            {
                "path": p,
                "static_friction": d.staticFriction,
                "dynamic_friction": d.dynamicFriction,
                "restitution": d.restitution,
                "density": d.density,
            }
        )

    physx: set[str] = set()
    for prim in stage.Traverse():
        physx.update(s for s in prim.GetAppliedSchemas() if s.startswith("Physx"))
    report.physx_schemas = sorted(physx)

    report.dropped_by_baseline = _baseline_gaps(report)
    return report


def _baseline_gaps(r: InspectionReport) -> list[str]:
    """Properties this asset authors that upstream usd2mjcf would ignore."""
    gaps = []
    if any(b.mass is not None or b.density is not None for b in r.bodies):
        gaps.append("mass/density (MassAPI) -> replaced by geometry-based guess")
    if r.materials:
        gaps.append("physics materials (friction/restitution) -> MuJoCo defaults")
    if any(j.drive for j in r.joints):
        gaps.append("joint drives -> no actuators, asset can't be driven")
    unsupported = Counter(
        j.type for j in r.joints if j.type not in ("Revolute", "Prismatic", "Fixed")
    )
    for t, n in unsupported.items():
        gaps.append(f"{n} {t} joint(s) -> skipped")
    if any(j.mimic for j in r.joints):
        gaps.append("mimic joints -> fingers move independently")
    if any(j.exclude_from_articulation for j in r.joints):
        gaps.append("loop-closing joints -> not represented")
    if r.collider_shapes:
        gaps.append("authored colliders -> not exported (regenerated from visuals if requested)")
    return gaps
