# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `srmj scene` / `make scene`: composes converted assets into one scene (floor, table, lights) with automatic placement from collision bounding boxes; big items on the floor, small ones on the table, fixed-base robots on the table edge; per-asset overrides in `configs/scene.yaml`; settle check reports anything that falls or slides.
- **Native USD → MJCF backend** (now the default; `--backend upstream` keeps the baseline):
  kinematic tree from UsdPhysics joints with correct joint frames and `ref`; mass via USD's
  `ComputeMassProperties`; authored colliders (primitives + meshes) and visuals with material
  colours; physics-material friction; `MjcActuator`, `MjcJointAPI`, `MjcCollisionAPI`,
  `MjcEqualityAPI`, `MjcSceneAPI`, Newton joint attributes; UsdPhysics drives → PD actuators;
  PhysX and Newton mimic joints → joint equalities; loop closures → connect/weld; fixed tendons;
  contact excludes from USD joint/filter semantics. Writes `conversion_report.json`.
- Stiff-drive check: warns when a drive is too stiff for the timestep and suggests armature.
- `srmj compare A.xml B.xml`: masses, joints, actuators and a hold-pose test side by side.
- `scripts/roundtrip_menagerie.sh` / `make roundtrip`: Menagerie → USD → MJCF validation.
  FR3, UR5e, Kinova Gen3 and Robotiq 2F-85 match Menagerie (see docs/validation.md).
- ADR 0003 (native backend semantics); docs/validation.md.

### Fixed (found on real SimReady assets)
- SimReady refrigerator's doors were blown open: under the MuJoCo variant its hollow carcass is a single `convexHull` collider, so MuJoCo treats it as a solid block that overlaps the compartments inside (106 mm). New `--decompose auto|concave|all|off` (default `auto`): concave colliders whose hull overlaps another part at the authored pose are split into convex parts with CoACD (cached in `~/.cache/simready-mujoco`).
- SimReady FR3 fell under gravity: its `MjcActuator`s have no position feedback (gains 0.003), while its UsdPhysics drives carry Menagerie's gains. New `--actuators auto|mjc|drive` (default `auto`) uses the drive in that case and warns. See docs/validation.md.
- Meshes over 200k faces are written as OBJ (MuJoCo's STL reader limit; refrigerator_a01).
- Mesh assets use `inertia="shell"` so open visual meshes load (FR3 link 5); masses always come from explicit inertials.
- Meshes with fewer than 4 vertices are skipped with a warning.

### Changed
- Asset-authored MuJoCo options now take precedence over simulation presets.
- Batch table shows conversion warnings; Hugging Face progress bars silenced.

### Added
- Pilot subset of 10 real SimReady Central assets (rigid props, file cabinet, fridge, faucet, Franka FR3, Franka Hand, Robotiq 2F-85) in `configs/subset.yaml`.
- `simready:` config entries: paste a package URL from the SimReady Central index; the fetcher downloads that commit and finds the root USD.
- `srmj batch --only ID` and `--fetch-only`; per-asset `inspection.json`; licence and failing stage in the batch report.
- `srmj inspect` reports USD variant sets (SimReady physics can live in a variant).
- Physics runtime selection: `--runtime mujoco|physx|newton|authored` (default `mujoco`) and `--variant SET=OPTION` on `inspect` and `convert`; `runtime:`/`variants:` in batch configs. Selections are made in memory, never written to the asset.
- `inspect` reports PhysX, Newton and MuJoCo (`Mjc*`) schemas, `MjcActuator`-type prims and `mjc:*` attributes; batch table shows what the MuJoCo layer contains.
- Third-party log output (upstream converter, CoACD) goes to `<output>/convert.log`; `--verbose` to show it.
- ADR 0002: use the asset's MuJoCo runtime layer as the primary source.

### Fixed
- Schemas without an installed plugin (all PhysX and mjcPhysics schemas) were invisible because `GetAppliedSchemas()` drops them; mimic joints were missed as a result.

### Fixed
- Tests no longer break when ROS 2 is sourced (`PYTHONPATH` cleared in `make test`).

## [0.1.0] — 2026-11-12

### Added
- Project skeleton: package layout, VS Code workspace, CI, issue/PR templates, docs.
- `srmj doctor`: environment check.
- `srmj inspect`: reports the physics an asset authors (bodies, mass, materials, joints, drives, colliders).
- `srmj convert` / `srmj batch`: baseline conversion through the upstream usd2mjcf backend.
- `srmj view`: open an MJCF in the MuJoCo viewer.
- Synthetic cabinet test asset and unit tests.
