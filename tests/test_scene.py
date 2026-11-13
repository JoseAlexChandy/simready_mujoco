"""Scene composition: floor + table + auto-placed converted assets."""

import os

import mujoco
import numpy as np
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
    build_scene(
        items, tmp_path / "s.xml", articulation=ArticulationSpec(damping=0.7, motors="keep")
    )
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


GADGET = """
<mujoco model="gadget">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="housing" pos="0 0 0.05">
      <freejoint name="housing_free"/>
      <geom type="box" size="0.15 0.1 0.05" mass="2"/>
      <body name="lid" pos="-0.15 0 0.05">
        <joint name="lid_hinge" type="hinge" axis="0 -1 0" range="0 1.8"/>
        <geom type="box" size="0.15 0.1 0.005" pos="0.15 0 0.005" mass="0.3"/>
      </body>
      <body name="button" pos="0.1 -0.1 0.02">
        <joint name="button_slide" type="slide" axis="0 1 0" range="0 0.004"/>
        <geom type="box" size="0.01 0.003 0.005" mass="0.002"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lid_motor" joint="lid_hinge" kp="20" kv="0.5"/>
  </actuator>
</mujoco>
"""


def _drag_and_release(rep, joint, value, seconds=1.0):
    m = mujoco.MjModel.from_xml_path(rep.scene)
    d = mujoco.MjData(m)
    j = m.joint(joint)
    d.qpos[j.qposadr[0]] = value
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(d.qpos[j.qposadr[0]])


@pytest.fixture
def gadget(tmp_path):
    f = tmp_path / "gadget" / "gadget.xml"
    f.parent.mkdir()
    f.write_text(GADGET)
    return f


def test_slider_drives_lid_and_a_drag_overpowers_it(gadget, tmp_path):
    rep = build_scene([Item("g", str(gadget), category="object")], tmp_path / "s" / "scene.xml")
    assert rep.passive_motors == ["g/lid_motor"]
    m = mujoco.MjModel.from_xml_path(rep.scene)
    d = mujoco.MjData(m)
    lid, base = m.joint("g/lid_hinge"), m.body("g/housing")
    q = lambda: float(d.qpos[lid.qposadr[0]])  # noqa: E731

    def run(seconds):
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)

    run(0.5)
    start = d.xpos[base.id].copy()
    d.ctrl[m.actuator("g/lid_motor").id] = 1.0  # Control panel slider
    run(2.0)
    assert q() == pytest.approx(1.0, abs=0.05)

    # a hand pulling the lid shut (like a viewer drag) beats the capped motor ...
    d.xfrc_applied[m.body("g/lid").id] = [0, 0, -20, 0, 0, 0]
    run(1.0)
    assert q() < 0.3
    # ... without shoving the whole gadget across the floor
    assert np.linalg.norm(d.xpos[base.id][:2] - start[:2]) < 0.005
    d.xfrc_applied[:] = 0
    run(2.0)
    assert q() == pytest.approx(1.0, abs=0.05)  # back to where the slider says


def test_passive_motors_lid_stays_where_left(gadget, tmp_path):
    rep = build_scene(
        [Item("g", str(gadget), category="object")],
        tmp_path / "s" / "scene.xml",
        articulation=ArticulationSpec(motors="passive"),
    )
    assert _drag_and_release(rep, "g/lid_hinge", 1.0) == pytest.approx(1.0, abs=0.1)


def test_motors_keep_pull_lid_back(gadget, tmp_path):
    kept = build_scene(
        [Item("g", str(gadget), category="object")],
        tmp_path / "k" / "scene.xml",
        articulation=ArticulationSpec(motors="keep"),
    )
    assert kept.passive_motors == []
    assert _drag_and_release(kept, "g/lid_hinge", 1.0) < 0.2


def test_objects_with_moving_parts_grip_the_floor(gadget, tmp_path):
    rep = build_scene([Item("g", str(gadget), category="object")], tmp_path / "s" / "scene.xml")
    m = mujoco.MjModel.from_xml_path(rep.scene)
    housing = m.body("g/housing").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == housing]
    assert all(m.geom_friction[g][0] >= 1.5 for g in geoms)


def test_button_springs_back(gadget, tmp_path):
    rep = build_scene([Item("g", str(gadget), category="object")], tmp_path / "s" / "scene.xml")
    assert rep.momentary_joints == ["g/button_slide"]
    assert _drag_and_release(rep, "g/button_slide", 0.0035) < 0.0005


def test_robot_motors_are_kept(gadget, tmp_path):
    rep = build_scene(
        [Item("g", str(gadget), category="robot", place=[0, 0, 0])],
        tmp_path / "s" / "scene.xml",
        include_robots=True,
    )
    assert rep.passive_motors == [] and rep.momentary_joints == []
