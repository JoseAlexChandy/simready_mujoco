"""Scene composition: floor + table + auto-placed converted assets."""

import os

import mujoco
import pytest
from build import build_cabinet, build_mimic_gripper, build_runtime_variant_asset

from simready_mujoco import convert
from simready_mujoco.sim.scene import ArticulationSpec, Item, TableSpec, build_scene


@pytest.fixture
def converted(tmp_path):
    out = tmp_path / "outputs"
    cab = tmp_path / "cabinet.usda"
    build_cabinet(cab)
    return {
        "cabinet": convert(cab, out / "cabinet"),  # 0.8 m tall -> floor
        "prop": convert(build_runtime_variant_asset(tmp_path / "prop_a01"), out / "prop"),
        "gripper": convert(build_mimic_gripper(tmp_path / "g.usda"), out / "gripper"),
    }


def test_auto_placement_and_settling(converted, tmp_path):
    items = [Item(k, str(v)) for k, v in converted.items()]
    rep = build_scene(items, tmp_path / "scene" / "scene.xml", TableSpec(enabled=True, height=0.75))
    surfaces = {i["id"]: i["surface"] for i in rep.items}
    assert surfaces == {"cabinet": "floor", "prop": "table", "gripper": "table"}

    # the free-floating prop comes to rest on the tabletop instead of falling to the floor
    (prop,) = [s for s in rep.settle if s["body"].startswith("prop/")]
    assert prop["ok"] and prop["final_z"] > 0.75


def test_scene_file_is_portable(converted, tmp_path, monkeypatch):
    scene = tmp_path / "scene" / "scene.xml"
    build_scene([Item(k, str(v)) for k, v in converted.items()], scene)
    assert str(tmp_path) not in scene.read_text()  # mesh paths are relative
    monkeypatch.chdir(os.path.expanduser("~"))
    m = mujoco.MjModel.from_xml_path(str(scene))
    assert m.body("cabinet/door").id > 0  # assets are namespaced by id
    assert m.actuator("cabinet/door_hinge_drive").id >= 0


def test_explicit_placement_and_skip(converted, tmp_path):
    items = [
        Item("cabinet", str(converted["cabinet"]), place="skip"),
        Item("prop", str(converted["prop"]), place=[0.5, -2.0, 0.3]),
    ]
    rep = build_scene(items, tmp_path / "s.xml")
    assert [i["id"] for i in rep.items] == ["prop"]
    assert rep.items[0]["pos"] == (0.5, -2.0, 0.3)


def test_default_is_floor_sheet_without_table(converted, tmp_path):
    rep = build_scene([Item(k, str(v)) for k, v in converted.items()], tmp_path / "s.xml")
    assert rep.sheet is not None and {i["surface"] for i in rep.items} == {"sheet"}
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "s.xml"))
    assert m.geom("sheet").id >= 0
    with pytest.raises(KeyError):
        m.geom("table_top")
    # everything rests on the sheet (top at its thickness), big cabinet at the back
    assert all(s["ok"] for s in rep.settle)
    pos = {i["id"]: i["pos"] for i in rep.items}
    assert pos["cabinet"][1] > pos["prop"][1]


def test_robots_left_out_unless_enabled(converted, tmp_path):
    def items():
        return [
            Item("cabinet", str(converted["cabinet"]), category="articulated"),
            Item("gripper", str(converted["gripper"]), category="gripper"),
        ]

    rep = build_scene(items(), tmp_path / "a.xml")
    assert rep.skipped_robots == ["gripper"]
    assert [i["id"] for i in rep.items] == ["cabinet"]
    rep = build_scene(items(), tmp_path / "b.xml", include_robots=True)
    assert rep.skipped_robots == [] and len(rep.items) == 2


def test_unactuated_joints_get_damping_so_dragged_parts_stay(converted, tmp_path):
    items = [Item("cabinet", str(converted["cabinet"]))]
    build_scene(items, tmp_path / "s.xml", articulation=ArticulationSpec(damping=0.7))
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "s.xml"))
    slide = m.joint("cabinet/drawer_slide")  # no motor -> softened
    hinge = m.joint("cabinet/door_hinge")  # driven by an actuator -> untouched
    assert m.dof_damping[slide.dofadr[0]] == pytest.approx(0.7)
    assert m.dof_frictionloss[slide.dofadr[0]] > 0
    assert m.dof_damping[hinge.dofadr[0]] == 0

    # pull the drawer out, let go: it stays roughly where it was left
    d = mujoco.MjData(m)
    d.qpos[slide.qposadr[0]] = 0.15
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qpos[slide.qposadr[0]] == pytest.approx(0.15, abs=0.02)
