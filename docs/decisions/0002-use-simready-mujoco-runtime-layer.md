# 0002 — Read the asset's MuJoCo runtime layer, translate UsdPhysics/PhysX only as a fallback

- **Date:** 2026-11-12
- **Status:** Accepted. Pilot inspection showed the FR3 and Franka Hand carry `MjcActuator`/`MjcJointAPI` data; NVIDIA props only change collider approximation under the MuJoCo variant.

## Context

Inspecting the pilot subset showed that SimReady assets follow SimReady Foundation's
runtime-variant layout: a neutral UsdPhysics base, plus `PhysX`, `Newton` and `MuJoCo`
variant sets on the default prim. `MuJoCo=Enabled` composes `runnables/physics/mujoco.usd`,
authored with Google DeepMind's `mjcPhysics` schema (`MjcActuator` prims, `Mjc*API` schemas,
`mjc:*` attributes). Under MuJoCo, colliders must be convex hulls rather than SDF meshes.

Our first inspections used the authored default (PhysX for props, nothing for the Franka),
so they did not see this data.

Separately, `Usd.Prim.GetAppliedSchemas()` drops schemas whose plugin isn't installed;
`usd-core` ships neither the PhysX nor the mjcPhysics plugins, so PhysX mimic joints and
all `Mjc*` schemas were invisible.

## Decision

1. Always compose the MuJoCo variant (`runtime: mujoco`) unless told otherwise.
2. Where the MuJoCo layer authors a value (actuator gains, contact params, options),
   map it 1:1 to MJCF. It was written for MuJoCo, so no translation is needed.
3. Translate UsdPhysics / PhysX values (the earlier plan) only where the MuJoCo layer is
   missing or silent, e.g. assets like Robotiq 2F-85 that ship no MuJoCo variant.
4. Read applied schemas from the composed `apiSchemas` list op, never `GetAppliedSchemas()`.

## Consequences

- Less guesswork: much of the PhysX→MuJoCo translation in the mapping doc becomes a fallback.
- The converter needs an `mjc:*` → MJCF mapping table (attribute names mirror MJCF closely).
- We depend on the mjcPhysics schema, which MuJoCo marks experimental; pin and test against
  the asset revisions in `configs/subset.yaml`.
