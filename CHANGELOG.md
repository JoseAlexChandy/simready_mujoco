# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Scene: objects' moving parts behave as they would in your hand. Robots and grippers are unchanged, and the asset files keep everything, so motors can still be driven elsewhere.
  - **Motors off** (`articulation: motors: passive`, default). Drives on objects (laptop lid, microwave door and release button, air-fryer basket and dial) held their joint at a target, so a dragged part sprang back. Their gains are zeroed in the scene and their damping is kept. `motors: keep` restores them.
  - **Momentary controls** (`momentary: true`). Short-travel joints (slides up to 15 mm, hinges up to 10°: buttons, triggers, the fridge door switch) spring back when let go, at 4 Hz with 0.8 damping, sized from each joint's own inertia.
  - **Parts hold their own weight.** Hinges and slides get friction of 1.3x the largest gravity torque about their axis, so a lid stays at any angle like a real hinge. Vertical door hinges and level drawers get none.
- Health checks: every converted asset is stepped on its own for 1 s. Divergence (NaN/Inf) and non-finite or zero mass properties are recorded as warnings naming the body, and `srmj batch` prints them in red. The scene settle check now fails and names the body when the simulation diverges: MuJoCo resets on NaN, which made a broken scene look settled ("-0 mm ✓" for everything).
- **Articulation for one-piece assets**: an `articulation:` list on an asset in `configs/subset.yaml` turns named parts (doors, lids, baskets, buttons, knobs) into hinges or sliders with limits, optional spring-back and damping. They are authored as UsdPhysics joints on the session layer, and the mass is shared by volume. `srmj parts ID|all|FILE` lists parts with their size and position in the asset frame. See docs/articulation.md.
- Viewer help now lists the lift (Ctrl + right-drag), slide-along-the-floor (Ctrl + Shift + right-drag) and twist controls.
- Visual appearance: textures through UsdPreviewSurface → UsdUVTexture (and OmniPBR `diffuse_texture`/`diffuse_color_constant` as fallback), exported as PNG (≤2048 px) with UV-mapped OBJ meshes; meshes split per GeomSubset material (e.g. laptop: 15 materials on one mesh); roughness → shininess; meshes without UVs use the texture's mean colour. Previously only constant colours were read, so most SimReady objects rendered white.
- `srmj add PATH`: finds USD assets (a file, a SimReady package folder, or a folder of either) and appends them to `configs/subset.yaml`, keeping its comments.
- Scene: default layout is now a light-brown sheet on the floor with objects in rows (big at the back, small in front); the table layout is kept behind `table: {enabled: true}`.
- Scene: robots and grippers are left out unless `robots: true`.
- Scene: unactuated doors/drawers get light damping + friction (`articulation:`), so parts dragged in the viewer stay where they are left; asset files unchanged.
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
- kitchenware_mug_d01 tipped over: its **authored** MassAPI values (on the collider) put the centre of mass 37 mm toward the handle, past the edge of its base. The real shape's is 1 mm off-axis and the hull's 9 mm, so the file's values are simply wrong. Two corrections now apply to authored mass properties (`authored_mass: auto`, the default):
  - **Centre of mass outside the base.** When the object at its authored rest pose would topple, but its real shape would not, the centre of mass and inertia come from the shape and the authored mass is kept (mug_d01). kitchenware_mug_e01's centre of mass is 10 mm off but still over its base, so it is left alone.
  - **Hull-derived values.** When the authored centre of mass sits on the colliders' hull centroid, far from the real shape's, the mass is also rescaled by real volume / hull volume.
  Original values go to `conversion_report.json` (`mass_corrections`), and the batch shows a warning. `authored_mass: keep` uses the file's values. Mass authored on colliders now counts as authored in `inertial_sources`. `scripts/mass_probe.py ID` prints authored, hull and shape values side by side.
- Mugs and other hollow or hooked objects tipped over: collider mass came from the convex hull, which fills the cup and the gap under the handle (kitchenware_mug_d01: 1.96 kg, centre of mass 37 mm toward the handle, outside its base). The hull is no longer used for mass. A closed mesh uses its own volume, after welding seam vertices and filling small holes. An open mesh is read as a 3 mm wall (both modelled faces count as one), which puts the centre of mass where the material is. `conversion_report.json` lists `mass_from: {mesh, shell, hull}`.
- `srmj parts` missed joints kept in a Scope with no geometry (how SimReady stores them), so articulated assets read "No joints authored". Joints are now listed in their own table (type, bodies, axis, limits), and authored masses are shown.
- SimReady refrigerator's doors were blown open: under the MuJoCo variant its hollow carcass is a single `convexHull` collider, so MuJoCo treats it as a solid block that overlaps the compartments inside (106 mm). New `--decompose auto|concave|all|off` (default `auto`): concave colliders whose hull overlaps another part at the authored pose are split into convex parts with CoACD (cached in `~/.cache/simready-mujoco`). Meshes still overlapping after a round are refined (tolerance 0.08 → 0.04 → 0.02, up to 128 parts); rounds are recorded in `conversion_report.json`. Overlap checks use fresh temporary folders because MuJoCo caches mesh files by path within a process (same-named files from an earlier round were being re-used).
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
