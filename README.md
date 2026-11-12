# simready-mujoco

Bring [NVIDIA SimReady Central](https://huggingface.co/spaces/nvidia/simready-central) OpenUSD assets into **MuJoCo** with their physics intact (mass, friction, joints, drives, collisions), and demo that they behave correctly.

Has its own USD → MJCF converter that reads each asset's MuJoCo runtime layer (`mjcPhysics`) and UsdPhysics data, validated against MuJoCo Menagerie ([docs/validation.md](docs/validation.md)). [LightwheelAI/usd2mjcf](https://github.com/LightwheelAI/usd2mjcf) is kept as a pinned submodule for baseline comparison. Coverage: [docs/usd_to_mjcf_mapping.md](docs/usd_to_mjcf_mapping.md).

---

## Quick start

```bash
git clone --recursive <this-repo-url> simready-mujoco
cd simready-mujoco
make setup      # creates .venv, installs everything
make doctor     # checks the environment
make test       # runs the tests
```

Then, with the environment active (`source .venv/bin/activate`):

```bash
srmj inspect tests/fixtures/cabinet.usda          # what physics does the asset author?
srmj convert tests/fixtures/cabinet.usda          # convert + check it loads and is stable
srmj view outputs/cabinet/cabinet.xml              # look at it in the MuJoCo viewer
srmj batch configs/subset.yaml                    # fetch + convert the pilot subset
srmj batch configs/subset.yaml --only franka_fr3  # just one asset
```

**Adding SimReady assets:** find the asset in the [SimReady Central index](https://github.com/NVIDIA-Omniverse/simready-central/tree/main/submissions) and copy its line into `configs/subset.yaml` as `simready: <url>`. Downloads are cached in `data/`.

**In VS Code:** open the folder, accept the recommended extensions, then use *Run and Debug* (`F5`) → pick `srmj: inspect USD file`, `srmj: convert USD file`, etc. *Terminal → Run Task* has setup, test, lint and convert-subset.

## Commands

| Command | What it does |
|---|---|
| `srmj doctor` | Checks Python, USD, MuJoCo, the upstream converter and optional tools |
| `srmj inspect FILE [--json]` | Lists bodies, mass, materials, joints, drives, colliders, and **what the baseline converter would lose** |
| `srmj convert FILE [-o DIR] [--collision] [--sim-profile NAME]` | Converts one asset, applies a MuJoCo simulation preset, verifies it loads and settles |
| `srmj batch CONFIG.yaml` | Converts every asset in a config (local or Hugging Face), writes `outputs/batch_report.json` |
| `srmj view FILE.xml` | Opens the interactive MuJoCo viewer |
| `srmj compare OURS.xml REF.xml` | Side-by-side masses, joints, actuators and a hold-pose test (e.g. vs Menagerie) |
| `make roundtrip` | Validates the converter: Menagerie FR3 → USD → MJCF, compared with the original |

Common options: `--runtime mujoco|physx|newton|authored` (which physics variant to read, default `mujoco`), `--variant SET=OPTION`, `--backend native|upstream`, `--verbose`.

## Use from Python

```python
from simready_mujoco import inspect_usd, convert, summarize_mjcf

report = inspect_usd("asset.usd")
print(report.dropped_by_baseline)

mjcf = convert("asset.usd", "outputs/asset", sim_profile="manipulation")
print(summarize_mjcf(mjcf))
```

## Repository layout

```
configs/                 asset lists (subset.yaml) for batch runs
docs/                    architecture, USD→MJCF mapping + status, design decisions
src/simready_mujoco/
  cli.py                 the `srmj` command
  api.py                 Python API
  ir.py                  intermediate representation (USD-free, MuJoCo-free)
  usd/                   reading USD: physics parsing and inspection
  backends/upstream.py   baseline converter (usd2mjcf)
  sim/                   MuJoCo loading, viewer, simulation presets
  sim_profiles.yaml      MuJoCo <option> presets
tests/                   unit tests; fixtures/ has small synthetic USD assets
third_party/usd2mjcf     upstream converter (git submodule, pinned)
data/                    downloaded assets (git-ignored)
outputs/                 conversion results (git-ignored)
```

## Roadmap

1. ✅ Project setup, inspection tool, baseline conversion, simulation presets
2. ✅ Pilot subset from SimReady Central, runtime-variant support
3. ✅ Native backend (validated against Menagerie); next: textures, keyframes, D6, spatial tendons
4. Demo scenarios (drop, push, joint sweep, drive) with recorded videos
5. Validation report: USD values vs MJCF values per asset
6. Scale to the full catalogue

Progress per mapping is tracked in [docs/usd_to_mjcf_mapping.md](docs/usd_to_mjcf_mapping.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Branches `feat/…`, conventional commits, tests for every mapping change.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). SimReady assets are not included; each has its own licence set by its publisher.
