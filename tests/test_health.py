"""Broken models must be named, not silently reset by MuJoCo."""

import mujoco

from simready_mujoco.sim.health import first_divergence, mass_problems
from simready_mujoco.sim.scene import settle_check

UNSTABLE = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="steady/root" pos="0 0 1"><freejoint/><geom size="0.1" mass="1"/></body>
    <body name="wild/root" pos="1 0 1">
      <freejoint/>
      <geom size="0.1" mass="1"/>
      <body name="wild/spring">
        <joint name="j" type="slide" axis="1 0 0" stiffness="1e14" springref="0.5"/>
        <geom type="box" size="0.01 0.01 0.01" mass="1e-6"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_divergence_is_located():
    m = mujoco.MjModel.from_xml_string(UNSTABLE)
    div = first_divergence(m, 0.5)
    assert div is not None
    assert div.body.startswith("wild/")


def test_stable_model_passes():
    m = mujoco.MjModel.from_xml_string(UNSTABLE.replace("1e14", "10").replace("1e-6", "0.1"))
    assert first_divergence(m, 0.2) is None
    assert mass_problems(m) == []


def test_settle_check_fails_instead_of_reporting_rest():
    m = mujoco.MjModel.from_xml_string(UNSTABLE)
    out = settle_check(m, [], 0.5)
    assert len(out) == 1 and not out[0]["ok"]
    assert "wild" in out[0]["error"]
