# USD → MJCF mapping

Living document: how each USD physics property becomes MJCF, and its implementation status. Update the **Status** column in the same PR as the code.

Legend: ✅ done · 🟡 baseline only (upstream usd2mjcf) · 🔲 planned

| Area | USD source | MJCF target | Status |
|---|---|---|---|
| Visual meshes + materials | `UsdGeom.Mesh`, `UsdShade` | `<mesh>`, `<material>`, geoms in `visual` class | 🟡 |
| Revolute / prismatic joints | `RevoluteJoint`, `PrismaticJoint` | `hinge` / `slide` with `range` | 🟡 |
| Fixed joints | `FixedJoint` | merged body | 🟡 |
| Mass / CoM / inertia | `MassAPI`, density, collider volumes | `<inertial>`; visual geoms `mass="0"` | 🔲 |
| Physics materials | `PhysicsMaterialAPI` (purpose `physics`) | geom `friction`, `solref` | 🔲 |
| Joint drives | `DriveAPI` (angular/linear) | `<general>` PD / `<velocity>`, `forcerange` | 🔲 |
| Initial state | `JointStateAPI`, drive targets | `<keyframe>` | 🔲 |
| Spherical joints | `SphericalJoint` | `ball` | 🔲 |
| D6 joints | `Joint` + per-axis `LimitAPI` | stacked `hinge`/`slide` (or `ball`) | 🔲 |
| Mimic joints | `PhysxMimicJointAPI` | `<equality><joint polycoef>` | 🔲 |
| Closed loops | joints with `excludeFromArticulation` | `<equality><connect>` | 🔲 |
| Primitive colliders | Cube/Sphere/Capsule/Cylinder + `CollisionAPI` | `box`/`sphere`/`capsule`/`cylinder` | 🔲 |
| Mesh colliders | `MeshCollisionAPI.approximation` | mesh / CoACD hulls | 🟡 (visual-mesh CoACD only) |
| Collision filtering | `FilteredPairsAPI`, collision groups, self-collision | `<exclude>`, `contype`/`conaffinity` | 🔲 |
| Gravity | `PhysicsScene` | `<option gravity>` | 🔲 |
| Solver settings | — (MuJoCo-specific) | presets in `src/simready_mujoco/sim_profiles.yaml` | ✅ |

## Known issues in the upstream baseline

Found on `tests/fixtures/cabinet.usda` (2 kg authored base). Each has an `xfail` test in `tests/test_baseline_upstream.py`.

| Issue | Effect | Evidence |
|---|---|---|
| Joint anchors placed relative to world origin, not the child body | Door swings around the wrong point | hinge anchor at (0, 0.2, 0) instead of (0.21, 0.2, 0.4) |
| `MassAPI` read but only printed (`# TODO` in upstream) | Mass guessed at 1000 kg/m³ | 152 kg total instead of ~2 kg base |
| Visual *and* collision geoms both carry mass | Mass doubles with `--collision` | 304 kg with collision on |
| Meshes with `CollisionAPI` are skipped entirely | Authored colliders lost; assets that use one mesh for both lose their visuals too | — |
| Drives ignored | `<actuator>` empty | 0 actuators |

## Conversion rules and gotchas

**Units.** Read `metersPerUnit` and `kilogramsPerUnit` from the stage. USD angles, angular limits and angular drive gains are in **degrees**; MJCF uses radians, so angular stiffness/damping are multiplied by 180/π.

**Mass.** Use `UsdPhysics.RigidBodyAPI.ComputeMassProperties` with a callback that supplies per-collider volume/inertia/CoM from the *collision* mesh (trimesh). This applies USD's precedence rules (explicit mass > density > default). If nothing is authored, fall back to density 1000 kg/m³ and log it. Visual geoms must have `mass="0"` or MuJoCo adds their mass too.

**Friction.** MuJoCo `friction = [sliding, torsional, rolling]`. sliding ← `dynamicFriction`; `staticFriction` has no MuJoCo equivalent (logged). PhysX averages two materials by default, while MuJoCo takes the maximum.

**Restitution.** MuJoCo has no restitution coefficient. Approximate via damping ratio ζ = −ln e / √(π² + ln² e), `solref = [timeconst, ζ]`, with `timeconst ≥ 2·timestep`.

**Drives.** stiffness > 0 → `<general gaintype="fixed" gainprm="kp" biastype="affine" biasprm="0 -kp -kv">`. stiffness = 0, damping > 0 → `<velocity kv>`. `maxForce` → `forcerange` (none if infinite). `type="acceleration"` drives: scale gains by the joint's effective inertia at the default pose and flag the asset.

**Spherical.** MuJoCo ball limits are a symmetric cone: use min(coneAngle0, coneAngle1) and warn if they differ.

**Capsules.** USD capsule `height` excludes the caps; MJCF `size = [radius, half-length of the cylinder part]`.

## Simulation settings

Don't copy Isaac's 60 Hz step. MuJoCo presets live in `src/simready_mujoco/sim_profiles.yaml`:
- `timestep 0.002` — stable contacts for small objects
- `integrator implicitfast` — MuJoCo's recommended integrator for actuated, damped systems
- `cone elliptic` + `impratio 10` — less slip in grasps
- `multiccd` — multiple contact points for flat mesh-on-mesh contact
