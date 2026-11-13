# simready-mujoco

Bring [NVIDIA SimReady Central](https://huggingface.co/spaces/nvidia/simready-central) OpenUSD assets into **MuJoCo** with their physics intact: mass, friction, joints, drives, collisions and textures. Then use them in an interactive scene where doors open, buttons spring back, and objects can be dragged with the mouse or placed with the keyboard.

The converter reads each asset's MuJoCo runtime layer (`mjcPhysics`) and its UsdPhysics data, and has been checked against MuJoCo Menagerie ([docs/validation.md](docs/validation.md)). Per-feature coverage is listed in [docs/usd_to_mjcf_mapping.md](docs/usd_to_mjcf_mapping.md). [LightwheelAI/usd2mjcf](https://github.com/LightwheelAI/usd2mjcf) is kept as a pinned submodule, as a baseline for comparison.

**Contents**

1. [Install](#1-install)
2. [Try it on the bundled example](#2-try-it-on-the-bundled-example)
3. [Convert one USD asset](#3-convert-one-usd-asset)
4. [Convert many assets (downloads or SimReady links)](#4-convert-many-assets)
5. [Build a scene with all of them](#5-build-a-scene)
6. [Viewer controls: mouse, sliders, keyboard](#6-viewer-controls)
7. [Use the converted objects in your own MuJoCo environment](#7-use-the-objects-in-your-own-mujoco-environment)
8. [Troubleshooting](#8-troubleshooting)
9. [Reference: commands, Python API, layout](#9-reference)

---

## 1. Install

You need **Linux or macOS**, **Python ≥ 3.10**, **git**, and a display for the viewer. A GPU isn't required. For a scene with ~60 textured assets, 8 GB of RAM or more is recommended.

```bash
git clone --recursive https://github.com/JoseAlexChandy/simready_mujoco.git simready-mujoco
cd simready-mujoco
make setup        # creates .venv and installs everything (USD, MuJoCo, CoACD, Pillow, …)
source .venv/bin/activate
make doctor       # every line should show ✓
make test         # all tests should pass
```

**Get the converted assets (optional, ~1 GB download).** The repo holds code only. The converted models, the kitchen scene and Koyo's downloaded assets are published as a GitHub Release:

```bash
make fetch-assets     # downloads, checks and unpacks into outputs/ and data/
srmj view outputs/scene/scene.xml
```

Skip this to start from your own USD files (sections 3–4). Licences and sources for every asset are in `outputs/ATTRIBUTION.md`; all are CC-BY-4.0.

Every later step assumes you're in the repo folder with the environment active. Your prompt should start with `(.venv)`. **Open a new terminal → run `cd simready-mujoco && source .venv/bin/activate` first.**

> Using ROS? Its `PYTHONPATH` can shadow packages. `make test` already unsets it; for your own commands, run `unset PYTHONPATH` once per terminal.

## 2. Try it on the bundled example

A small synthetic cabinet (hinged door, sliding drawer, motor) is included, so nothing needs downloading:

```bash
srmj inspect tests/fixtures/cabinet.usda      # what physics the USD file defines
srmj convert tests/fixtures/cabinet.usda      # → outputs/cabinet/cabinet.xml (+ meshes/)
srmj view outputs/cabinet/cabinet.xml         # open it in the MuJoCo viewer
```

## 3. Convert one USD asset

```bash
srmj inspect path/to/asset.usd                # optional: what's in it
srmj convert path/to/asset.usd -o outputs     # → outputs/asset/asset.xml
srmj view outputs/asset/asset.xml
```

Each converted asset is a self-contained folder:

```
outputs/<asset>/
  <asset>.xml               the MuJoCo model (MJCF): bodies, joints, motors, colliders, materials
  meshes/                   visual + collision meshes (.obj / .stl)
  textures/                 PNG textures (up to 2048 px)
  conversion_report.json    what was converted, mass sources, corrections, warnings
```

Useful options:

- `--decompose concave`: hollow colliders, for containers such as bowls, baskets and racks.
- `--runtime physx|newton|authored`: which physics variant of a SimReady asset to read. The default is `mujoco`.
- `--sim-profile none`: keep MuJoCo's default solver settings.

`srmj convert --help` lists everything.

## 4. Convert many assets

Assets are listed in **`configs/subset.yaml`**. One `srmj batch` converts all of them and prints a table with any problems.

### 4a. Assets downloaded as zip files (e.g. from SimReady Central)

```bash
scripts/unpack.sh ~/Downloads data/koyo     # every .zip in Downloads → data/koyo/<name>/
srmj add data/koyo                          # adds new assets to configs/subset.yaml
srmj batch configs/subset.yaml              # convert everything (or: --only <id> --only <id>)
```

`srmj add` skips assets that are already listed, and writes paths relative to the repo so the config works on any machine. The `data/` folder is git-ignored, so on a fresh clone, put the same zip files in `~/Downloads` and run the two commands above.

### 4b. SimReady assets by link

Copy an asset's package URL from the [SimReady Central index](https://github.com/NVIDIA-Omniverse/simready-central/tree/main/submissions) into the config:

```yaml
  - id: refrigerator_a01
    category: articulated
    simready: https://huggingface.co/datasets/nvidia/simready-assets/resolve/<commit>/refrigerator_a01/com.nvidia.simready.packaging.json
```

`srmj batch` downloads it into `data/`, with a cache.

### 4c. See what you've got

Numbered downloads (`D00270`, …) don't say what they are, so render them all into one labelled picture with ID, size and mass:

```bash
python scripts/contact_sheet.py             # → outputs/contact_sheet.png
python scripts/contact_sheet.py --new       # only assets added since the last sheet
```

### 4d. Per-asset options

Set options without editing the YAML by hand:

```bash
srmj set decompose concave d00270 d00323 d00249   # hollow colliders for containers
srmj set authored_mass keep d00364                # use the file's mass values as-is
srmj set category robot my_arm                     # robots are left out of scenes by default
srmj batch configs/subset.yaml --only d00270 --only d00323 --only d00249
```

Every option is described at the top of `configs/subset.yaml`. Two more tools:

- **`srmj parts <id>`** lists an asset's parts and joints.
- **[docs/articulation.md](docs/articulation.md)** explains how to add hinges and sliders to assets that ship as one rigid piece.

## 5. Build a scene

```bash
make scene                                  # = srmj scene configs/scene.yaml --view
```

This places every converted asset from `configs/subset.yaml` on a light-brown floor sheet, with big items at the back. Items with an `on`/`in` placement are put in or on their container. It then lets everything settle, prints a ✓ or `!` line per object, and opens the viewer.

Everything is set in **`configs/scene.yaml`**:

| Section | What it does |
|---|---|
| `robots: true/false` | include robots and grippers |
| `sheet:` / `table:` | floor sheet (default) or a table |
| `textures: max_size` | texture size in the scene (default 1024 px). Lower it, or set `enabled: false`, if memory is short |
| `articulation:` | how moving parts behave: motors follow the Control sliders but give way to a drag, buttons spring back, lids hold their angle. See [docs/articulation.md](docs/articulation.md) |
| `place:` | per asset: `skip`, `[x, y, z]`, or **on/in another asset** (below) |
| `yaw:` | rotate an asset about the vertical, in degrees |

**Putting things on or in other things:**

```yaml
place:
  d00322: {in: d00270}                  # apple in the bowl (lands on the bowl's floor)
  d00253: {in: d00249, upright: true}   # knife standing in the cutlery caddy, handle down
  d00323: {on: d00326, level: 2}        # basket on the shelf's 2nd surface from the top
  d00080: {on: d00264, at: [0.3, 0]}    # loaf on the cutting board, right of centre
```

- **`in`** puts the item on the container's floor, under handles and between dividers.
- **`on`** counts surfaces from the top (`level: n`).
- **`at`** picks a spot on the container: −1 to 1 each way, 0 is the centre.

Chains work: fruit in a basket on a shelf. Containers need hollow colliders, so convert them with `srmj set decompose concave <ids>` first (section 4d).

To rebuild without opening the viewer, run `srmj scene configs/scene.yaml`. To open the last scene, run `srmj view outputs/scene/scene.xml`.

## 6. Viewer controls

Open any model with **`srmj view FILE.xml`**, or `make scene` for the full scene.

**Mouse** (standard MuJoCo):

| | |
|---|---|
| left-drag / right-drag / scroll | rotate / pan / zoom the camera |
| **double-click** | select an object or part (a door, a lid…) |
| **Ctrl + right-drag** | lift / move the selection |
| **Ctrl + Shift + right-drag** | slide it along the floor |
| **Ctrl + left-drag** | twist it |
| Space / Backspace | pause / reset |

Drag a door, drawer or lid to open it. It stays where you leave it.

**Control panel (right side):** sliders for each motor, such as the laptop lid, microwave door and toaster lever. A drag overpowers a motor; let go and the part returns to the slider's position.

**Keyboard placement.** Double-click an object, or any part of it, then:

| Keys | Action |
|---|---|
| **arrow keys** | move along the floor (↑ = away from you; follows the camera) |
| **W / S** | raise / lower |
| **A / D** | turn left / right |
| **Q / E** | tilt forward / back |
| **Z / C** | roll left / right |
| **R** | restore the original orientation |
| **Shift** + key | bigger steps (5 cm / 15° instead of 1 cm / 5°) |

- **Hold a key** to keep moving.
- **Pause (Space)** to place things precisely; otherwise they settle under gravity as you move them.
- **The terminal prints** the object's position and yaw/pitch/roll after each move.
- **While an object is selected** these keys go to the object. Double-click empty space to deselect, and they go back to the viewer's own shortcuts (W wireframe, S shadows, …).
- **Only free-floating objects move.** Anything fixed to the world, such as a robot base, is refused with a message.

## 7. Use the objects in your own MuJoCo environment

The converted files are plain MuJoCo models. Each object's root body has a `<freejoint/>`, so it can be picked up and moved in any environment.

**Add objects to your own model with `place_asset`** (Python):

```python
import mujoco
from simready_mujoco import place_asset

spec = mujoco.MjSpec.from_file("my_kitchen.xml")                    # your environment
place_asset(spec, "outputs/d00270/d00270.xml", pos=(0.5, 0.0, 0.80))  # bowl on your table
place_asset(spec, "outputs/d00322/d00322.xml", pos=(0.5, 0.0, 0.85))  # apple above it
place_asset(spec, "outputs/toaster/toaster.xml", pos=(0.9, 0.2, 0.80), yaw_deg=90)

model = spec.compile()                                  # use in your own code, or save it:
open("my_kitchen_with_assets.xml", "w").write(spec.to_xml())
```

`place_asset` fixes the object's mesh and texture paths so they resolve from anywhere. It prefixes every name with the asset ID (`d00270/…`) so objects don't clash, and adopts the solver settings the assets are tuned for (elliptic friction cones, implicit integrator), unless your model sets its own.

> Don't copy-paste or `<include>` an asset's XML into another file. Its `meshes/` and `textures/` paths are relative to its own folder, and MuJoCo would resolve them against the other file's folder.

**Open your environment with keyboard placement:**

```bash
srmj view my_kitchen_with_assets.xml
```

or from Python:

```python
from simready_mujoco.sim.viewer import view
view("my_kitchen_with_assets.xml")
```

Mouse, sliders and keyboard all work as in section 6.

**Which viewers support what:**

| Opened with | Mouse drag & sliders | Keyboard placement |
|---|---|---|
| `srmj view`, `simready_mujoco.sim.viewer.view()`, `make scene` | ✓ | ✓ |
| `python -m mujoco.viewer`, MuJoCo's `simulate` app | ✓ | — |
| your own loop with `mujoco.viewer.launch_passive` | ✓ | — (not yet) |

The scene's "hand feel" is added by `srmj scene` and isn't stored in the object files. That covers drag-friendly motors, floor grip, buttons springing back and lids holding their angle. The raw object XMLs keep their original motors and friction.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `srmj: command not found` | `cd simready-mujoco && source .venv/bin/activate` |
| `make scene` prints **Killed** (out of memory) | in `configs/scene.yaml` set `textures: {max_size: 512}`, or `enabled: false` |
| An object is white / untextured | `srmj doctor` must show Pillow ✓; then re-run `srmj batch` |
| `!` in the scene's settle list | it rose or sank more than 5 cm, slid more than 10 cm, or (if placed `upright`) fell over. Check its `place:` line; containers need `srmj set decompose concave` |
| An apple sits **on top of** a bowl | the bowl's collider is solid: `srmj set decompose concave <bowl id>` and re-run `srmj batch` |
| An object tips over / feels wrong | see its `outputs/<id>/conversion_report.json` (`mass_corrections`, `warnings`), and `python scripts/mass_probe.py <id>` |
| Tests fail with odd import errors | `unset PYTHONPATH` (ROS), then `make test` |
| Your config has `/home/<someone>/…` paths from before | `sed -i "s#path: $PWD/#path: #" configs/subset.yaml` |

## 9. Reference

### Commands

| Command | What it does |
|---|---|
| `srmj doctor` | checks Python, USD, MuJoCo, Pillow, CoACD and the upstream converter |
| `srmj inspect FILE [--json]` | lists bodies, mass, materials, joints, drives and colliders, and what the baseline converter would lose |
| `srmj convert FILE [-o DIR]` | converts one asset and checks that it loads and settles |
| `srmj add PATH` | adds downloaded USD assets (file, folder, or folder of them) to `configs/subset.yaml` |
| `srmj set KEY VALUE IDS…` | sets a per-asset option in the config (e.g. `decompose concave`) |
| `srmj batch CONFIG [--only ID]…` | converts every asset in a config and writes `outputs/batch_report.json` |
| `srmj parts ID\|all\|FILE` | lists an asset's parts and joints, for writing `articulation:` specs |
| `srmj scene [CONFIG] [--view]` | builds `outputs/scene/scene.xml` and checks that everything settles |
| `srmj view FILE.xml` | opens the viewer with keyboard placement |
| `srmj compare OURS.xml REF.xml` | compares masses, joints, actuators and a hold-pose test (e.g. against Menagerie) |
| `make setup / doctor / test / lint / scene / roundtrip / clean` | shortcuts; `make help` lists them |
| `scripts/unpack.sh SRC [DEST]` | unzips downloaded archives into `data/` |
| `scripts/contact_sheet.py` | renders every converted asset into one labelled picture |
| `scripts/mass_probe.py ID` | compares an asset's authored mass with its hull and its real shape |
| `make fetch-assets` | downloads the newest `assets-*` release into `outputs/` and `data/` |
| `make pack` / `make release` | packs `outputs/` and `data/koyo` (with attribution and checksums) and uploads them as a GitHub Release |

### Python API

```python
from simready_mujoco import inspect_usd, convert, summarize_mjcf, place_asset, load_asset

report = inspect_usd("asset.usd")                        # what the USD file defines
mjcf = convert("asset.usd", "outputs/asset")             # USD → MJCF (returns the .xml path)
print(summarize_mjcf(mjcf))                              # bodies, joints, mass, stability
place_asset(spec, mjcf, pos=(0, 0, 1))                   # into your own MjSpec (section 7)
spec = load_asset(mjcf)                                  # an asset as an MjSpec, paths fixed
```

### Repository layout

```
configs/subset.yaml      the assets to convert (ids, sources, per-asset options)
configs/scene.yaml       how the scene is laid out and how parts behave
src/simready_mujoco/
  cli.py                 the `srmj` command
  api.py                 Python API (convert, place_asset, …)
  native/                USD → MJCF converter (parse, appearance, decomposition, writing)
  usd/                   USD helpers: variants, inspection, adding joints (articulate.py)
  sim/                   scene building, on/in placement (nesting.py), viewer + keyboard
                         (viewer.py), health checks, simulation presets
scripts/                 unpack.sh, contact_sheet.py, mass_probe.py, roundtrip_menagerie.sh
docs/                    articulation, validation, USD→MJCF mapping, design decisions
tests/                   tests; fixtures/ has small synthetic USD assets
third_party/usd2mjcf     baseline converter (pinned git submodule)
data/                    downloaded assets (git-ignored)
outputs/                 conversion results and scenes (git-ignored)
```

What changed and why is recorded in [CHANGELOG.md](CHANGELOG.md), including every fix found on real assets. To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md): branches `feat/…`, conventional commits, and tests for every change.

### Publishing new assets

After converting more assets, share them with:

```bash
make release          # packs dist/ and creates GitHub Release assets-YYYY-MM-DD (needs gh)
```

The archives are split under GitHub's 2 GB file limit. `ATTRIBUTION.md`, generated from each asset's package info, is included, as CC-BY requires. SimReady assets added by link aren't packed, because `srmj batch` downloads them again. Clones get the new release with `make fetch-assets`.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). SimReady assets are **not** in the repository. They're published separately as release files (`make fetch-assets`), each under the licence set by its publisher, listed in `ATTRIBUTION.md`.
