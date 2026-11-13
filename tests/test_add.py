from pathlib import Path

import yaml
from build import build_cabinet
from test_fetch import _make_asset

from simready_mujoco.add import add_to_config, find_assets


def _config(tmp_path: Path) -> Path:
    c = tmp_path / "subset.yaml"
    c.write_text("# my assets\noutput_dir: outputs\nassets:\n  - id: existing\n    path: x.usda\n")
    return c


def test_single_file(tmp_path):
    usd = tmp_path / "Cool Chair.usda"
    build_cabinet(usd)
    found, problems = find_assets(usd)
    assert [f.id for f in found] == ["cool_chair"] and not problems


def test_folder_of_simready_packages(tmp_path):
    _make_asset(tmp_path / "downloads", name="Robotiq_2F_85")
    _make_asset(tmp_path / "downloads", name="mug_a01", usds=("simready_usd/mug_a01.usd",))
    found, _ = find_assets(tmp_path / "downloads")
    assert sorted(f.id for f in found) == ["mug_a01", "robotiq_2f_85"]
    assert all(f.path.name.endswith(".usd") for f in found)


def test_plain_folder_with_subfolders(tmp_path):
    d = tmp_path / "dl"
    (d / "lamp").mkdir(parents=True)
    build_cabinet(d / "lamp" / "lamp.usda")
    build_cabinet(d / "vase.usda")
    found, _ = find_assets(d)
    assert sorted(f.id for f in found) == ["lamp", "vase"]


def test_add_appends_keeps_comments_and_skips_duplicates(tmp_path):
    cfg = _config(tmp_path)
    usd = tmp_path / "chair.usda"
    build_cabinet(usd)
    found, _ = find_assets(usd)
    assert add_to_config(cfg, found) == ["chair"]
    assert add_to_config(cfg, found) == []  # same path again
    text = cfg.read_text()
    assert text.startswith("# my assets")
    ids = [a["id"] for a in yaml.safe_load(text)["assets"]]
    assert ids == ["existing", "chair"]


def test_set_option_edits_entries_in_place(tmp_path):
    from simready_mujoco.add import set_option

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "# keep me\nassets:\n"
        "  - id: bowl\n    category: object\n    path: a.usd\n"
        "  - id: plate\n    category: object\n    decompose: off\n    path: b.usd\n"
    )
    done, missing = set_option(cfg, ["bowl", "plate", "ghost"], "decompose", "concave")
    assert done == ["bowl", "plate"] and missing == ["ghost"]
    text = cfg.read_text()
    assert text.startswith("# keep me")
    import yaml

    entries = {e["id"]: e for e in yaml.safe_load(text)["assets"]}
    assert entries["bowl"]["decompose"] == "concave"
    assert entries["plate"]["decompose"] == "concave"
    assert entries["plate"]["path"] == "b.usd"
