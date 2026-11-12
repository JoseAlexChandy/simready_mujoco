# USD → MJCF mapping

Living document: how each USD physics property becomes MJCF, and its implementation status. Update the **Status** column in the same PR as the code.

Legend: ✅ done (native backend) · 🔲 planned

| Area | USD source | MJCF target | Status |
|---|---|---|---|
| Kinematic tree | UsdPhysics joints (body0 → body1), flipped where needed | nested `<body>`; joint frames from `localPose0/1` | ✅ |
| Joint rest angle | joint frames at the authored pose | `ref` | ✅ |
| Revolute / prismatic | `RevoluteJoint`, `PrismaticJoint` | `hinge` / `slide`, `range` (deg → rad) | ✅ |
| Fixed | `FixedJoint` | child body without joint (welded) | ✅ |
| Spherical | `SphericalJoint` | `ball`, symmetric cone `range` | ✅ |
| D6 | `Joint` + per-axis `LimitAPI` | — | 🔲 (warned, treated as fixed) |
| Loose bodies | rigid body not attached to world | `freejoint` (kinematic → welded) | ✅ |
| Loop closures | `excludeFromArticulation` / cycles | `<equality><connect>` / `<weld>` | ✅ |
| Mass / CoM / inertia | `MassAPI`, collider density, defaults | `<inertial>` via `ComputeMassProperties` | ✅ |
| Collision | authored colliders (primitives + meshes) | geoms in `collision` class | ✅ |
| Concave colliders | mesh collider whose convex hull would overlap other parts | convex parts via CoACD (`--decompose auto`) | ✅ |
| Collision filtering | joint `collisionEnabled=false`, `FilteredPairsAPI` | `<contact><exclude>` | ✅ |
| Visuals | renderable gprims | geoms in `visual` class, material base colour | ✅ (no textures yet) |
| Physics materials | `PhysicsMaterialAPI` (+ `MjcMaterialAPI`) | geom `friction` | ✅ (restitution 🔲) |
| MuJoCo actuators | `MjcActuator` (joint or tendon target) | `<general>` 1:1; `--actuators auto` falls back to the drive if the MjcActuator can't hold a position | ✅ |
| UsdPhysics drives | `DriveAPI` angular/linear | PD `<general>` (per-degree → per-radian) | ✅ |
| Joint dynamics | `MjcJointAPI`, `newton:armature/damping/friction` | `armature`, `damping`, `frictionloss`, … | ✅ |
| Mimic joints | `NewtonMimicAPI`, `PhysxMimicJointAPI` | `<equality><joint polycoef>` (re-based on `ref`) | ✅ |
| Fixed tendons | `MjcTendon` type fixed | `<tendon><fixed>` | ✅ (spatial 🔲) |
| Contact params | `MjcCollisionAPI` | `condim`, `margin`, `solref`, … | ✅ |
| Equality softness | `MjcEqualityAPI` | `solref`, `solimp` | ✅ |
| Scene | `PhysicsScene` gravity, `MjcSceneAPI` | `<option>`, `<flag>` (asset values beat presets) | ✅ |
| Keyframes | `MjcKeyframe` | `<keyframe>` | 🔲 |
| Stiff-drive check | — | warning + suggested `armature` in `conversion_report.json` | ✅ |

## Source priority (ADR 0002)

SimReady assets carry a MuJoCo runtime layer (`MuJoCo=Enabled` → `runnables/physics/mujoco.usd`, `mjcPhysics` schema). Values authored there map 1:1 to MJCF and take priority. The UsdPhysics/PhysX translations in the table above are the fallback for assets or properties the MuJoCo layer doesn't cover.

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
