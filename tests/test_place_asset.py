"""Using converted assets in your own MuJoCo model (the README's example)."""

import mujoco
import pytest
from build import build_textured_asset

from simready_mujoco import convert, place_asset

MY_ENV = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="table" pos="0 0 0.4"><geom type="box" size="0.5 0.4 0.02"/></body>
</worldbody></mujoco>"""


def test_place_assets_in_your_own_environment(tmp_path, monkeypatch):
    xml = convert(build_textured_asset(tmp_path / "tex"), tmp_path / "outputs" / "tex")
    env = tmp_path / "elsewhere" / "my_env.xml"  # a different folder: relative paths would break
    env.parent.mkdir()
    env.write_text(MY_ENV)
    monkeypatch.chdir(tmp_path / "elsewhere")

    spec = mujoco.MjSpec.from_file(str(env))
    place_asset(spec, xml, pos=(0.0, 0.0, 0.6))
    place_asset(spec, xml, pos=(0.3, 0.0, 0.6), yaw_deg=90, prefix="second/")
    model = spec.compile()
    assert model.ntex >= 1
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC  # the asset's solver settings
    roots = [model.body(b).name for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    assert any(r.startswith("tex/") for r in roots) and any(r.startswith("second/") for r in roots)

    d = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, d)
    first = [b for b in range(model.nbody) if model.body(b).name.startswith("tex/")][0]
    assert d.xpos[first][2] == pytest.approx(0.42, abs=0.01)  # base resting on the tabletop

    # save it and open it again from anywhere
    out = tmp_path / "elsewhere" / "my_env_with_assets.xml"
    out.write_text(spec.to_xml())
    assert mujoco.MjModel.from_xml_path(str(out)).nbody == model.nbody
