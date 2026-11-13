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


def test_nothing_selected_keys_stay_with_the_viewer(mover):
    mv, _, d, _ = mover
    before = d.qpos.copy()
    assert not mv.handle(kv.KEY_W)  # W = wireframe in the viewer
    assert np.array_equal(d.qpos, before)


def test_moves_follow_the_camera(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/display").id  # a part selects its whole object
    assert mv.handle(kv.KEY_UP)  # away from the camera: +y
    assert mv.handle(kv.KEY_RIGHT)  # right: +x
    assert mv.handle(kv.KEY_W)  # up
    assert d.qpos[:3] == pytest.approx([1.01, 0.01, 0.21])
    assert "laptop" in mv.status


def test_shift_takes_bigger_steps_and_ctrl_goes_to_the_viewer(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/base").id
    assert mv.handle(kv.KEY_S, kv.MOD_SHIFT)
    assert d.qpos[2] == pytest.approx(0.15)
    assert not mv.handle(kv.KEY_S, kv.MOD_CONTROL)
    assert d.qpos[2] == pytest.approx(0.15)


def test_turn_about_centre_of_mass_and_restore(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/base").id
    com = d.subtree_com[m.body("laptop/base").id].copy()
    for _ in range(18):  # 18 x 5 deg = 90 deg turn left
        mv.handle(kv.KEY_A)
    w, x, y, z = d.qpos[3:7]
    assert math.degrees(2 * math.atan2(z, w)) == pytest.approx(90, abs=1e-6)
    assert d.subtree_com[m.body("laptop/base").id] == pytest.approx(com, abs=1e-9)
    mv.handle(kv.KEY_Q)  # tilt
    mv.handle(kv.KEY_C)  # roll
    mv.handle(kv.KEY_R)
    assert d.qpos[3:7] == pytest.approx([1, 0, 0, 0])
    assert d.subtree_com[m.body("laptop/base").id] == pytest.approx(com, abs=1e-9)


def test_velocity_cleared(mover):
    mv, m, d, pert = mover
    pert.select = m.body("laptop/base").id
    d.qvel[:6] = 1.0
    mv.handle(kv.KEY_LEFT)
    assert np.all(d.qvel[:6] == 0)


def test_fixed_objects_leave_keys_to_the_viewer(mover):
    mv, m, d, pert = mover
    pert.select = m.body("table/base").id
    before = d.qpos.copy()
    assert not mv.handle(kv.KEY_UP)
    assert np.array_equal(d.qpos, before) and "fixed to the world" in mv.status


def test_viewer_internals_still_exist():
    """view() hooks these; if MuJoCo renames them it falls back to the plain viewer."""
    import inspect

    import glfw
    import mujoco.viewer as mv

    assert hasattr(mv, "_Simulate") and hasattr(mv, "_launch_internal")
    assert "key_callback" in inspect.signature(mv._launch_internal).parameters
    assert hasattr(glfw, "_glfw") and hasattr(glfw._glfw, "glfwSetKeyCallback")
