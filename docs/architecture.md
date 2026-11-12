# Architecture

## Pipeline

Stages 1 (inspection) and the IR types exist today; stages 3–5 are being built. Until then `srmj convert` uses the upstream usd2mjcf backend.

```
 SimReady USD asset
        │
        ▼
 ┌──────────────────┐   UsdPhysics.UsdPhysicsLoadStageFromPrimRange
 │ 1. Parse         │   (official USD physics parser: bodies, colliders,
 │   usd/parse.py   │    joints, drives, materials, articulations)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   Plain dataclasses, no USD or MuJoCo types.
 │ 2. IR            │   Units normalised here: metres, kg, radians, SI gains.
 │   ir.py          │
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   One function per concern: inertial, materials,
 │ 3. Map           │   joints, actuators, equality, collision, options.
 │   mapping/*.py   │   Each one is unit-tested on its own.
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   MJCF XML + meshes. Visual meshes/materials reuse the
 │ 4. Export        │   upstream usd2mjcf exporter.
 │   export/        │
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   Drop / push / joint-sweep / drive demos, videos,
 │ 5. Demo+Validate │   USD-vs-MJCF comparison report.
 │   sim/           │
 └──────────────────┘
```

**Why an intermediate representation (IR):** it decouples "what does the USD say" from "how do we express it in MuJoCo". Each mapping is testable in isolation, and the same IR could later target other formats (e.g. MuJoCo `MjSpec` directly, or another simulator) — useful when this joins the larger project.

## Package layout

| Path | Purpose |
|---|---|
| `src/simready_mujoco/cli.py` | `srmj` command-line tool |
| `src/simready_mujoco/api.py` | Python API for other projects to import |
| `src/simready_mujoco/usd/` | Reading USD: inspection and parsing to IR |
| `src/simready_mujoco/ir.py` | Intermediate representation |
| `src/simready_mujoco/backends/` | Conversion backends. `upstream` = vendored usd2mjcf (baseline) |
| `src/simready_mujoco/sim/` | Loading, viewing and (later) demo scenarios in MuJoCo |
| `configs/` | Asset subsets and simulation presets (YAML) |
| `tests/` | Unit tests; `tests/fixtures/` holds tiny synthetic USD assets |
| `third_party/usd2mjcf` | Upstream converter, git submodule pinned to a commit |
| `docs/decisions/` | Short records of design decisions (ADRs) |

## Integrating into a larger project

Use the Python API rather than shelling out to the CLI:

```python
from simready_mujoco import inspect_usd, convert

report = inspect_usd("asset.usd")          # dict: what physics the asset authors
mjcf_path = convert("asset.usd", "outputs/asset", sim_profile="manipulation")
```

The package is pip-installable (`pip install -e path/to/simready-mujoco` or from git), has no hard-coded paths, and reads all settings from arguments or YAML.
