# 0001 — Parse with UsdPhysics into an IR instead of patching usd2mjcf in place

- **Date:** 2026-11-12
- **Status:** Accepted

## Context

usd2mjcf converts visuals, materials and basic joints well, but ignores mass, physics materials, drives, spherical/D6/mimic joints and authored colliders. It walks the USD prim tree itself, so adding each feature means re-implementing USD physics precedence rules by hand.

## Decision

1. Parse physics with `UsdPhysics.UsdPhysicsLoadStageFromPrimRange` (part of `usd-core`), which already resolves material bindings, drives, limits, articulations and collider shapes.
2. Convert the parse result into our own intermediate representation (`ir.py`), normalising units there.
3. Write MJCF from the IR. Reuse upstream usd2mjcf only for visual mesh and material export.
4. Keep upstream as a pinned git submodule, used as the baseline backend until our backend covers it.

## Consequences

- Each mapping is a small, separately tested function.
- We can compare our output against the upstream baseline on every asset.
- A small amount of duplicated logic with upstream (joint frames) during the transition.
