"""End-to-end conversion with the upstream usd2mjcf backend.

The `xfail(strict=True)` tests record known gaps. When the native backend fixes one, the test
starts passing, pytest reports XPASS as a failure, and you remove the marker. They double as
the to-do list.
"""

import mujoco
import numpy as np
import pytest

from simready_mujoco import convert, summarize_mjcf
from simready_mujoco.backends.upstream import UpstreamNotFound, upstream_path

try:
    upstream_path()
except UpstreamNotFound:
    pytest.skip("usd2mjcf submodule not checked out", allow_module_level=True)


@pytest.fixture
def cabinet_mjcf(cabinet_usd, tmp_path):
    return convert(cabinet_usd, tmp_path / "out", sim_profile="manipulation")


def _model(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_converts_and_is_stable(cabinet_mjcf):
    s = summarize_mjcf(cabinet_mjcf)
    assert s.joints == 2  # hinge + slide (fixed joint merged)
    assert s.timestep == 0.002  # sim profile applied
    assert s.settles


def test_joint_limits_converted_to_radians(cabinet_mjcf):
    m, _ = _model(cabinet_mjcf)
    assert np.allclose(m.joint("door_hinge").range, [0.0, np.pi / 2])
    assert np.allclose(m.joint("drawer_slide").range, [0.0, 0.3], atol=1e-6)


@pytest.mark.xfail(strict=True, reason="baseline: MassAPI ignored, mass guessed from visuals")
def test_authored_mass_respected(cabinet_mjcf):
    m, _ = _model(cabinet_mjcf)
    base_body = m.geom("base_visual").bodyid[0]
    assert m.body_mass[base_body] == pytest.approx(2.0)


@pytest.mark.xfail(strict=True, reason="baseline: DriveAPI not converted to actuators")
def test_drive_becomes_actuator(cabinet_mjcf):
    m, _ = _model(cabinet_mjcf)
    assert m.nu == 1


@pytest.mark.xfail(strict=True, reason="baseline: physics material friction ignored")
def test_friction_from_physics_material(cabinet_mjcf):
    m, _ = _model(cabinet_mjcf)
    assert any(np.isclose(m.geom_friction[:, 0], 0.6))


@pytest.mark.xfail(strict=True, reason="upstream bug: joint anchor placed at world origin")
def test_hinge_anchor_on_cabinet_edge(cabinet_mjcf):
    _, d = _model(cabinet_mjcf)
    m, _ = _model(cabinet_mjcf)
    anchor = d.xanchor[m.joint("door_hinge").id]
    assert np.allclose(anchor, [0.21, 0.2, 0.4], atol=1e-3)
