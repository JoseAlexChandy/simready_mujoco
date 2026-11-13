"""Intermediate representation (IR) between USD and MJCF.

Plain dataclasses with no USD or MuJoCo types, so mapping functions are easy to test.
Units: metres, kilograms, radians; gains in N/m or N·m/rad. Quaternions are (w, x, y, z).

The body list is a tree: each Body names its parent ("" = world) and the joints that connect
it to that parent. Constraints that don't fit the tree (loops, mimic couplings) are Equality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class JointType(str, Enum):
    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"
    FREE = "free"


class ShapeType(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    MESH = "mesh"
    PLANE = "plane"


@dataclass
class Pose:
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Inertial:
    """Mass properties. `source` records where they came from, for the report."""

    mass: float
    pos: Vec3 = (0.0, 0.0, 0.0)  # centre of mass in the body frame
    diaginertia: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)  # principal axes in the body frame
    source: str = "computed"  # "authored" | "computed" | "default"


@dataclass
class MeshData:
    """Triangle mesh in the geom's own frame (scale already baked in)."""

    name: str
    vertices: Any  # (N, 3) float array
    faces: Any  # (M, 3) int array
    inertia: str | None = None  # MuJoCo mesh inertia mode ("shell", "convex", ...)
    maxhullvert: int | None = None
    source: str | None = None  # for convex parts: the mesh they were decomposed from
    texcoords: Any = None  # (N, 2) per-vertex UVs (USD convention, v=1 at image top), or None


@dataclass
class VisualMaterial:
    """Render appearance for visual geoms (MJCF <material> + optional <texture>)."""

    name: str
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    texture_src: str | None = None  # absolute path of the source image
    specular: float | None = None
    shininess: float | None = None


@dataclass
class Geom:
    name: str
    shape: ShapeType
    pose: Pose = field(default_factory=Pose)
    size: tuple[float, ...] = ()  # MJCF size semantics (half-extents, radius, half-length)
    mesh: str | None = None  # MeshData.name
    collision: bool = True
    rgba: tuple[float, float, float, float] | None = None
    visual_material: str | None = None  # VisualMaterial name (visual geoms)
    friction: tuple[float, ...] | None = None  # (slide, torsion, roll); None = MuJoCo default
    extra: dict[str, Any] = field(default_factory=dict)  # passthrough MJCF attributes (mjc:*)
    source_path: str = ""


@dataclass
class Joint:
    name: str
    type: JointType
    pos: Vec3 = (0.0, 0.0, 0.0)  # in the child body frame
    axis: Vec3 = (0.0, 0.0, 1.0)
    range: tuple[float, float] | None = None
    ref: float = 0.0  # joint value at the authored pose (MJCF `ref`)
    extra: dict[str, Any] = field(default_factory=dict)  # armature, damping, frictionloss, ...
    source_path: str = ""


@dataclass
class Body:
    name: str
    parent: str = ""  # "" = world
    pose: Pose = field(default_factory=Pose)  # relative to parent
    joints: list[Joint] = field(default_factory=list)
    inertial: Inertial | None = None
    geoms: list[Geom] = field(default_factory=list)
    source_path: str = ""


@dataclass
class Actuator:
    name: str
    joint: str  # target joint name, or tendon name when target == "tendon"
    attrs: dict[str, Any] = field(default_factory=dict)  # MJCF <general> attributes
    source: str = "mjc"  # "mjc" (MjcActuator prim) | "drive" (UsdPhysics DriveAPI)
    target: str = "joint"  # "joint" | "tendon"


@dataclass
class Tendon:
    """Fixed tendon: a linear combination of joint coordinates."""

    name: str
    joints: list[tuple[str, float]] = field(default_factory=list)  # (joint name, coef)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Equality:
    kind: str  # "joint" | "connect" | "weld"
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    name: str
    bodies: list[Body] = field(default_factory=list)  # parents always listed before children
    world_geoms: list[Geom] = field(default_factory=list)  # static colliders/visuals
    meshes: dict[str, MeshData] = field(default_factory=dict)
    visual_materials: dict[str, VisualMaterial] = field(default_factory=dict)
    actuators: list[Actuator] = field(default_factory=list)
    equalities: list[Equality] = field(default_factory=list)
    tendons: list[Tendon] = field(default_factory=list)
    excludes: list[tuple[str, str]] = field(default_factory=list)  # body pairs that never collide
    option: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)  # stats for the report

    def body(self, name: str) -> Body:
        return next(b for b in self.bodies if b.name == name)
