"""Intermediate representation (IR) between USD and MJCF.

Plain dataclasses with no USD or MuJoCo types, so mapping functions are easy to test.
Everything is in SI units: metres, kilograms, radians, N/m or N·m/rad gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JointType(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPHERICAL = "spherical"
    D6 = "d6"


class ShapeType(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    MESH = "mesh"
    PLANE = "plane"


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo convention


@dataclass
class Pose:
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Inertial:
    """Mass properties of a body. `source` records where they came from, for the report."""

    mass: float
    com: Vec3 = (0.0, 0.0, 0.0)
    diag_inertia: Vec3 | None = None
    principal_axes: Quat = (1.0, 0.0, 0.0, 0.0)
    source: str = "default"  # "authored" | "density" | "computed" | "default"


@dataclass
class Material:
    name: str
    dynamic_friction: float | None = None
    static_friction: float | None = None
    restitution: float | None = None
    density: float | None = None


@dataclass
class Geom:
    name: str
    shape: ShapeType
    pose: Pose = field(default_factory=Pose)
    size: tuple[float, ...] = ()
    mesh_file: str | None = None
    material: str | None = None  # name of a Material
    collision: bool = True
    mesh_approximation: str | None = None  # USD MeshCollisionAPI approximation


@dataclass
class Drive:
    """Joint drive in SI units (already converted from USD degrees where needed)."""

    stiffness: float = 0.0
    damping: float = 0.0
    max_force: float | None = None  # None = unlimited
    target_position: float = 0.0
    target_velocity: float = 0.0
    acceleration: bool = False  # USD drive type "acceleration"


@dataclass
class Joint:
    name: str
    type: JointType
    parent: str
    child: str
    axis: Vec3 = (0.0, 0.0, 1.0)
    pose_in_parent: Pose = field(default_factory=Pose)
    pose_in_child: Pose = field(default_factory=Pose)
    limits: tuple[float, float] | None = None
    drive: Drive | None = None
    friction: float = 0.0
    armature: float = 0.0
    exclude_from_articulation: bool = False  # loop-closing joint -> equality constraint


@dataclass
class Mimic:
    follower: str
    leader: str
    gearing: float = 1.0
    offset: float = 0.0


@dataclass
class Body:
    name: str
    pose: Pose = field(default_factory=Pose)
    inertial: Inertial | None = None
    geoms: list[Geom] = field(default_factory=list)


@dataclass
class Model:
    name: str
    bodies: list[Body] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    materials: dict[str, Material] = field(default_factory=dict)
    mimics: list[Mimic] = field(default_factory=list)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    warnings: list[str] = field(default_factory=list)
