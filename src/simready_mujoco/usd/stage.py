"""Opening SimReady assets with the right physics runtime variant selected.

SimReady assets keep a neutral UsdPhysics base and add solver-specific data through variant
sets on the default prim (PhysX / Newton / MuJoCo, each Disabled|Enabled). Selecting
MuJoCo=Enabled composes runnables/physics/mujoco.usd, authored with the mjcPhysics schema.

Selections are made on the session layer, so the asset files on disk are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pxr import Sdf, Usd

# runtime name -> variant set that switches it on (SimReady Foundation convention)
RUNTIME_VARIANT_SETS = {"mujoco": "MuJoCo", "physx": "PhysX", "newton": "Newton"}
RUNTIMES = (*RUNTIME_VARIANT_SETS, "authored")


@dataclass
class Selection:
    runtime: str
    runtime_available: bool | None = None  # None when runtime == "authored"
    applied: dict[str, str] = field(default_factory=dict)  # variant set -> option
    ignored: dict[str, str] = field(default_factory=dict)  # requested but not on the asset


def parse_variant_args(items: list[str] | None) -> dict[str, str]:
    """['Physics=Newton_parallel_grip', ...] -> {'Physics': 'Newton_parallel_grip'}"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Variant must look like SET=OPTION, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def open_stage(
    path: str | Path,
    runtime: str = "mujoco",
    variants: dict[str, str] | None = None,
) -> tuple[Usd.Stage, Selection]:
    """Open a USD file, enable one physics runtime variant, then apply explicit selections.

    runtime: "mujoco" | "physx" | "newton" enable that runtime's variant set and disable the
             others; "authored" keeps whatever the file selects by default.
    variants: explicit {variant set: option} selections, applied last (they win).
    """
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime '{runtime}'. Choose from: {', '.join(RUNTIMES)}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD file: {path}")

    sel = Selection(runtime=runtime)
    prim = stage.GetDefaultPrim()
    if not prim:
        return stage, sel

    stage.SetEditTarget(stage.GetSessionLayer())
    vsets = prim.GetVariantSets()
    names = set(vsets.GetNames())

    def select(set_name: str, option: str) -> None:
        if set_name in names and option in vsets.GetVariantSet(set_name).GetVariantNames():
            vsets.GetVariantSet(set_name).SetVariantSelection(option)
            sel.applied[set_name] = option
        else:
            sel.ignored[set_name] = option

    if runtime != "authored":
        target = RUNTIME_VARIANT_SETS[runtime]
        sel.runtime_available = target in names
        for vs_name in RUNTIME_VARIANT_SETS.values():
            if vs_name in names:
                select(vs_name, "Enabled" if vs_name == target else "Disabled")

    for k, v in (variants or {}).items():
        select(k, v)

    stage.SetEditTarget(stage.GetRootLayer())
    return stage, sel


def applied_schemas(prim: Usd.Prim) -> list[str]:
    """All API schemas listed on a prim, including ones whose plugin isn't installed.

    Usd.Prim.GetAppliedSchemas() silently drops unregistered schemas, and usd-core ships
    neither the PhysX nor the MuJoCo schema plugins, so read the composed list op instead.
    """
    op = prim.GetMetadata("apiSchemas")
    if not isinstance(op, Sdf.TokenListOp):
        return []
    return [str(s) for s in op.ApplyOperations([])]
