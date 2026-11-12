# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Pilot subset of 10 real SimReady Central assets (rigid props, file cabinet, fridge, faucet, Franka FR3, Franka Hand, Robotiq 2F-85) in `configs/subset.yaml`.
- `simready:` config entries: paste a package URL from the SimReady Central index; the fetcher downloads that commit and finds the root USD.
- `srmj batch --only ID` and `--fetch-only`; per-asset `inspection.json`; licence and failing stage in the batch report.
- `srmj inspect` reports USD variant sets (SimReady physics can live in a variant).

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
