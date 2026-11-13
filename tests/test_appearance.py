"""Visual appearance: textures through UsdPreviewSurface, GeomSubset materials."""

import mujoco
import numpy as np
import pytest
from build import build_textured_asset

from simready_mujoco import convert


@pytest.fixture
def textured(tmp_path):
    return convert(build_textured_asset(tmp_path / "tex_a01"), tmp_path / "o")


def _materials(m):
    return {m.material(i).name: i for i in range(m.nmat)}


def test_texture_exported_and_bound(textured):
    m = mujoco.MjModel.from_xml_path(str(textured))
    assert m.ntex == 1
    assert (textured.parent / "textures").is_dir()
    assert all(p.suffix == ".png" for p in (textured.parent / "textures").iterdir())  # jpg -> png
    quad = m.geom("quad")
    mat = quad.matid[0]
    assert mat >= 0 and m.mat_texid[mat].max() >= 0
    assert m.mat_shininess[mat] == pytest.approx(0.75)  # 1 - roughness
    assert m.mesh_texcoordnum[m.geom_dataid[quad.id]] >= 4


def test_geomsubset_materials(textured):
    m = mujoco.MjModel.from_xml_path(str(textured))
    mats = _materials(m)
    assert {"green", "yellow"} <= set(mats)
    assert np.allclose(m.mat_rgba[mats["green"], :3], [0.1, 0.8, 0.1], atol=1e-4)
    box_visuals = [
        g for g in range(m.ngeom) if m.geom(g).name.startswith("box_") and m.geom_group[g] == 2
    ]
    assert {m.geom_matid[g] for g in box_visuals} == {mats["green"], mats["yellow"]}


def test_texture_orientation_renders_correctly(textured):
    """USD v=1 is the top of the image: the +y half of the quad must be red."""
    try:
        m = mujoco.MjModel.from_xml_path(str(textured))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 80, 80)
    except Exception as e:  # noqa: BLE001  (no OpenGL in this environment)
        pytest.skip(f"no offscreen rendering: {e}")
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.3]
    cam.distance, cam.elevation, cam.azimuth = 0.35, -90, 90
    r.update_scene(d, cam)
    px = r.render().astype(int)
    top, bottom = px[20, 40], px[60, 40]
    assert top[0] > top[2] and bottom[2] > bottom[0]
