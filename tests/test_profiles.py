import xml.etree.ElementTree as ET

import pytest

from simready_mujoco.sim.profiles import apply_profile, get_profile, load_profiles


def test_shipped_profiles_load():
    profiles = load_profiles()
    assert {"default", "manipulation", "fast_preview"} <= set(profiles)


def test_unknown_profile_raises():
    with pytest.raises(KeyError, match="Unknown sim profile"):
        get_profile("does-not-exist")


def test_apply_profile_overwrites_option(tmp_path):
    xml = tmp_path / "m.xml"
    xml.write_text(
        '<mujoco><compiler angle="radian"/><option integrator="RK4" timestep="0.01"/>'
        "<worldbody/></mujoco>"
    )
    apply_profile(xml, get_profile("manipulation"))
    opt = ET.parse(xml).getroot().find("option")
    assert opt.get("timestep") == "0.002"
    assert opt.get("integrator") == "implicitfast"
    assert opt.get("cone") == "elliptic"
    assert opt.find("flag").get("multiccd") == "enable"


def test_apply_profile_creates_option_if_missing(tmp_path):
    xml = tmp_path / "m.xml"
    xml.write_text('<mujoco><compiler angle="radian"/><worldbody/></mujoco>')
    apply_profile(xml, {"timestep": 0.004})
    root = ET.parse(xml).getroot()
    assert [c.tag for c in root][:2] == ["compiler", "option"]
    assert root.find("option").get("timestep") == "0.004"
