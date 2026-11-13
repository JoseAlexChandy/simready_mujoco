"""Joints added from `articulation:` specs to assets that ship as one rigid piece."""

import json
import math

import mujoco
import numpy as np
import pytest
from build import build_static_appliance
from typer.testing import CliRunner

from simready_mujoco import convert
from simready_mujoco.cli import app
from simready_mujoco.usd.articulate import PartJoint, apply_articulation, parse_axis
from simready_mujoco.usd.stage import open_stage

DOOR = {"part": "door", "joint": "hinge", "axis": "-z", "at": ["-x", "-y"], "range": [0, 110]}
BUTTON = {
    "part": "button",
    "joint": "slide",
    "axis": "+y",
    "range": [0, 0.005],
    "spring": 200.0,
    "damping": 1.0,
}


@pytest.fixture
def appliance(tmp_path):
    return build_static_appliance(tmp_path / "microwave.usda")


def _load(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    return m, mujoco.MjData(m)


def _in_root_frame(m, d, point_world):
    root = next(b for b in range(m.nbody) if m.body_parentid[b] == 0 and b > 0)
    R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
    return R.T @ (np.asarray(point_world) - p)


def test_spec_validation():
    with pytest.raises(ValueError, match="unknown keys"):
        PartJoint.from_dict({"part": "door", "hinge_at": "left"})
    with pytest.raises(ValueError, match="hinge or slide"):
        PartJoint.from_dict({"part": "door", "joint": "ball"})
    assert parse_axis("-z").tolist() == [0, 0, -1]
    with pytest.raises(ValueError):
        parse_axis("w")


def test_unknown_part_lists_help(appliance):
    stage, _ = open_stage(appliance)
    with pytest.raises(ValueError, match="srmj parts"):
        apply_articulation(stage, [{"part": "lid"}])


def test_without_spec_it_is_one_rigid_piece(appliance, tmp_path):
    m, _ = _load(convert(appliance, tmp_path / "plain"))
    assert m.njnt == 1 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_door_hinges_on_its_front_left_edge(appliance, tmp_path):
    out = convert(appliance, tmp_path / "o", articulation=[DOOR, BUTTON])
    m, d = _load(out)
    assert m.njnt == 3
    hinge = m.joint("door_hinge")
    assert int(hinge.type[0]) == mujoco.mjtJoint.mjJNT_HINGE
    assert np.allclose(np.degrees(hinge.range), [0, 110])
    door = m.body("door").id
    mujoco.mj_forward(m, d)
    closed = _in_root_frame(m, d, d.xipos[door])

    d.qpos[hinge.qposadr[0]] = math.radians(90)
    mujoco.mj_forward(m, d)
    opened = _in_root_frame(m, d, d.xipos[door])
    # door centre (asset frame, m) is ~(-0.075, -0.185); hinge line at x=-0.25, y=-0.195.
    # Opened 90 deg outwards it sticks out to the front (-y), still left of the hinge.
    assert closed[0] == pytest.approx(-0.075, abs=0.01)
    assert opened[0] == pytest.approx(-0.25 + 0.01, abs=0.02)
    assert opened[1] == pytest.approx(-0.195 - 0.175, abs=0.02)
    assert opened[2] == pytest.approx(closed[2], abs=1e-6)


def test_mass_is_shared_and_report_lists_joints(appliance, tmp_path):
    out = convert(appliance, tmp_path / "o", articulation=[DOOR, BUTTON])
    m, _ = _load(out)
    total = sum(m.body_mass[1:])
    assert total == pytest.approx(12.0, rel=1e-3)
    assert 0.1 < m.body_mass[m.body("door").id] < 2.0
    rep = json.loads((out.parent / "conversion_report.json").read_text())
    assert [j["type"] for j in rep["added_joints"]] == ["hinge", "slide"]


def test_button_springs_back(appliance, tmp_path):
    out = convert(appliance, tmp_path / "o", articulation=[DOOR, BUTTON])
    m, d = _load(out)
    j = m.joint("button_slide")
    assert m.jnt_stiffness[j.id] == pytest.approx(200.0)
    # Pin the appliance so only the button moves, press it and let go.
    m.opt.gravity[:] = 0
    d.qpos[j.qposadr[0]] = 0.004
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qpos[j.qposadr[0]] < 0.001


def test_door_does_not_fight_the_carcass(appliance, tmp_path):
    """Door and carcass colliders touch: the added joint excludes that pair, so the door
    stays shut when the object rests on the floor."""
    out = convert(appliance, tmp_path / "o", articulation=[DOOR])
    m, d = _load(out)
    hinge = m.joint("door_hinge")
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert abs(math.degrees(d.qpos[hinge.qposadr[0]])) < 2


def test_parts_command(appliance):
    r = CliRunner().invoke(app, ["parts", str(appliance)])
    assert r.exit_code == 0, r.output
    assert "door_panel" in r.output and "No joints authored" in r.output
