"""Keyboard placement in the viewer (logic only; no display needed)."""

import math

import mujoco
import numpy as np
import pytest

from simready_mujoco.sim import viewer as kv

SCENE = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="table/base" pos="0 0 0.5"><geom type="box" size="0.3 0.3 0.02"/></body>
    <body name="laptop/base" pos="1 0 0.2">
      <freejoint/>
      <geom type="box" size="0.15 0.1 0.01" mass="1"/>
      <body name="laptop/display" pos="-0.15 0 0.01">
        <joint type="hinge" axis="0 1 0" range="0 2"/>
        <geom type="box" size="0.01 0.1 0.1" pos="0 0 0.1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def mover():
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pert, cam = mujoco.MjvPerturb(), mujoco.MjvCamera()
    cam.azimuth = 90.0  # looking along +y: away = +y, right = +x
    out = kv.KeyboardMover(m, d, pert, cam, echo=None)
    return out, m, d, pert


def test_nothing_selected(mover):
    mv, _, d, _ = mover
    before = d.qpos.copy()
    mv.on_key(kv.KEY_HOME)
    assert np.array_equal(d.qpos, before) and "double-click" in mv.status


def test_moves_follow_the_camera(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/display").id  # a part selects its whole object
    mv.on_key(kv.KEY_HOME)  # away from the camera: +y
    mv.on_key(kv.KEY_DELETE)  # right: +x
    mv.on_key(kv.KEY_PAGE_UP)  # up
    assert d.qpos[:3] == pytest.approx([1.02, 0.02, 0.22])
    assert "laptop" in mv.status


def test_turn_about_centre_of_mass_and_restore(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/base").id
    com = d.subtree_com[m.body("laptop/base").id].copy()
    mv.on_key(kv.KEY_F9)  # rotate mode
    for _ in range(18):  # 18 x 5 deg = 90 deg turn left
        mv.on_key(kv.KEY_INSERT)
    w, x, y, z = d.qpos[3:7]
    assert math.degrees(2 * math.atan2(z, w)) == pytest.approx(90, abs=1e-6)
    assert d.subtree_com[m.body("laptop/base").id] == pytest.approx(com, abs=1e-9)
    mv.on_key(kv.KEY_F12)
    assert d.qpos[3:7] == pytest.approx([1, 0, 0, 0])
    assert d.subtree_com[m.body("laptop/base").id] == pytest.approx(com, abs=1e-9)


def test_step_sizes_and_velocity_cleared(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/base").id
    d.qvel[:6] = 1.0
    mv.on_key(kv.KEY_F10)  # coarse
    mv.on_key(kv.KEY_PAGE_UP)
    assert d.qpos[2] == pytest.approx(0.3)
    assert np.all(d.qvel[:6] == 0)


def test_fixed_objects_are_refused(mover):
    mv, m, d, pert = mover
    pert.select = m.body("table/base").id
    before = d.qpos.copy()
    mv.on_key(kv.KEY_HOME)
    assert np.array_equal(d.qpos, before) and "fixed to the world" in mv.status


def test_viewer_internals_still_exist():
    """view() hooks these; if MuJoCo renames them it falls back to the plain viewer."""
    import inspect

    import mujoco.viewer as mv

    assert hasattr(mv, "_Simulate") and hasattr(mv, "_launch_internal")
    assert "key_callback" in inspect.signature(mv._launch_internal).parameters
