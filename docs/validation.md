# Validation

## Round trip against MuJoCo Menagerie

The converter's ground truth: take a hand-tuned Menagerie model, convert it to USD with NVIDIA's
`mujoco-usd-converter` (which authors `mjcPhysics` schemas the way SimReady MuJoCo layers do),
convert it back with `srmj` (native backend), and compare with the original.

```bash
make roundtrip                                              # Franka FR3
scripts/roundtrip_menagerie.sh universal_robots_ur5e/ur5e.xml   # any Menagerie model
```

Results (2026-11-12, MuJoCo 4.0.1, mujoco-usd-converter 0.9.0):

| Model | Total mass | Per-joint mass / range / armature / damping / frictionloss | Actuators kp / kv / force range | Behaviour |
|---|---|---|---|---|
| Franka FR3 | 18.3156 vs 18.3155 kg | identical | identical | hold-pose error 0.00582439952 vs 0.00582439918 rad |
| UR5e | 20.99 vs 20.99 kg | identical | identical | hold-pose error 0.0081 vs 0.0081 rad |
| Kinova Gen3 | 8.188 vs 8.188 kg | identical | identical | hold-pose error 0.0044 vs 0.0044 rad |
| Robotiq 2F-85 (tendon + loop closures) | 1.0526 vs 1.0526 kg | identical | identical (tendon actuator) | closed: 0.7808 / 0.7799 / −0.7555 vs 0.7808 / 0.7799 / −0.7556 rad |

Visual check (FR3, home pose; left = converted, right = Menagerie original):

![FR3 round trip](img/fr3_roundtrip.png)

What the round trip does **not** cover: assets authored only with UsdPhysics/PhysX (no MuJoCo
layer), e.g. the SimReady Robotiq 2F-85 and NVIDIA props. Those are covered by the synthetic
fixtures in `tests/` and by `srmj compare` against Menagerie where a model exists.

## Unit tests

`tests/test_native.py` builds small USD assets in code and checks the compiled MuJoCo model:
authored vs computed mass, hinge anchors, limits in SI units, drive → PD actuator gains, drive
reaching its target in simulation, physics-material friction, contact filtering, `MjcActuator`
pass-through, free joints for loose bodies, PhysX and Newton mimic joints tracking in
simulation, joint reference angles, fixed tendons.

## Real SimReady assets: findings

### Franka FR3 (`frankarobotics/simready-assets` @ b54ce583, `fr3v2_1`)

| | SimReady MuJoCo layer (`MjcActuator`) | SimReady UsdPhysics drive (`ControlMode=position`) | Menagerie |
|---|---|---|---|
| kp | 0.003, and **no position term** (`biasprm = [0, 0, −0.003]`) | 78.54 N·m/° = **4500** N·m/rad | 4500 |
| kv | 0.003 | 7.854 N·m/° = **450** | 450 |
| force range | ±87 / ±12 | ±87 / ±12 | ±87 / ±12 |

The MuJoCo layer's actuator gains equal the URDF damping value (`urdf:dynamics:damping = 0.003`),
so they look like an authoring bug in the asset's MJCF/MuJoCo generation; with them the arm
cannot hold a pose and falls under gravity. `--actuators auto` (default) detects an
`MjcActuator` without position feedback on a joint that has a stiff position drive, uses the
drive, and logs a warning. Worth reporting upstream to Franka/NVIDIA.

Other differences vs Menagerie are genuine model differences, not conversion errors: link
masses (total 15.93 vs 18.32 kg, SimReady has no separate base body), joint frictionloss 0.2
on all joints, joint damping not authored for MuJoCo.

### Refrigerator (`nvidia/simready-assets` @ 653d91e1, `refrigerator_a01`)

Under the MuJoCo variant every collider is `convexHull`. The carcass is one hollow mesh, so its
hull is a solid block that starts 106 mm into the two door compartments (`locker_01/02`);
MuJoCo pushes them out and both doors swing open (97° and −102° within 2 s). PhysX avoided this
with SDF colliders. `--decompose auto` finds concave colliders that overlap other parts at the
authored pose and splits them into convex parts (CoACD); reproduced and fixed on the synthetic
`build_hollow_cabinet` fixture (door 63° open → 0°).

On the real fridge the first round (tolerance 0.08, 19 parts) left two carcass pieces 24–35 mm
inside the lockers, so one locker still swung open; `auto` now re-checks after each round and
refines meshes that still overlap.
