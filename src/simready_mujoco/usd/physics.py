"""Thin wrapper around the official UsdPhysics stage parser."""

from __future__ import annotations

from typing import Any

from pxr import Usd, UsdPhysics


def parse_physics(stage: Usd.Stage, root: str = "/") -> dict[str, list[tuple[str, Any]]]:
    """Parse every physics object under `root`.

    Returns {object type name: [(prim path, descriptor), ...]}, e.g.
    {"RigidBody": [...], "RevoluteJoint": [...], "MeshShape": [...]}.
    Descriptors are UsdPhysics *Desc objects (RigidBodyDesc, RevoluteJointDesc, ...).
    """
    loader = (
        getattr(UsdPhysics, "UsdPhysicsLoadStageFromPrimRange", None)
        or UsdPhysics.LoadUsdPhysicsFromRange
    )
    raw = loader(stage, [root])
    out: dict[str, list[tuple[str, Any]]] = {}
    for obj_type, (paths, descs) in raw.items():
        out[str(obj_type)] = [(str(p), d) for p, d in zip(paths, descs, strict=False)]
    return out
