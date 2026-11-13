"""Objects on/in other objects: fruit in a bowl, baskets on shelf tiers, knives upright."""

import mujoco
import numpy as np
import pytest

from simready_mujoco.sim.nesting import NestSpec, load_shape, rotation
from simready_mujoco.sim.scene import Item, build_scene

COL = 'group="3" contype="1" conaffinity="1"'

BOWL = f"""<mujoco model="bowl"><worldbody><body name="bowl" pos="0 0 0.0">
  <freejoint/>
  <geom type="box" size="0.08 0.08 0.005" pos="0 0 0.005" {COL} mass="0.2"/>
  <geom type="box" size="0.08 0.005 0.03" pos="0 0.075 0.03" {COL} mass="0.05"/>
  <geom type="box" size="0.08 0.005 0.03" pos="0 -0.075 0.03" {COL} mass="0.05"/>
  <geom type="box" size="0.005 0.08 0.03" pos="0.075 0 0.03" {COL} mass="0.05"/>
  <geom type="box" size="0.005 0.08 0.03" pos="-0.075 0 0.03" {COL} mass="0.05"/>
</body></worldbody></mujoco>"""

SHELF = f"""<mujoco model="shelf"><worldbody><body name="shelf" pos="0 0 0">
  <freejoint/>
  <geom type="box" size="0.3 0.2 0.01" pos="0 0 0.60" {COL} mass="3"/>
  <geom type="box" size="0.3 0.2 0.01" pos="0 0 0.30" {COL} mass="3"/>
  <geom type="box" size="0.3 0.2 0.01" pos="0 0 0.01" {COL} mass="3"/>
  <geom type="box" size="0.01 0.01 0.31" pos="0.29 0.19 0.31" {COL} mass="1"/>
  <geom type="box" size="0.01 0.01 0.31" pos="-0.29 0.19 0.31" {COL} mass="1"/>
  <geom type="box" size="0.01 0.01 0.31" pos="0.29 -0.19 0.31" {COL} mass="1"/>
  <geom type="box" size="0.01 0.01 0.31" pos="-0.29 -0.19 0.31" {COL} mass="1"/>
</body></worldbody></mujoco>"""

APPLE = f"""<mujoco model="apple"><worldbody><body name="apple" pos="0 0 0.03">
  <freejoint/><geom type="sphere" size="0.03" {COL} mass="0.2"/>
</body></worldbody></mujoco>"""

# 30 cm knife along x: light blade (+x), heavy handle (-x)
KNIFE = f"""<mujoco model="knife"><worldbody><body name="knife" pos="0 0 0.01">
  <freejoint/>
  <geom type="box" size="0.1 0.012 0.002" pos="0.05 0 0" {COL} mass="0.05"/>
  <geom type="box" size="0.05 0.012 0.008" pos="-0.1 0 0" {COL} mass="0.3"/>
</body></worldbody></mujoco>"""


# like the real display shelf: boards thicker than any fixed step (6 cm)
THICK_SHELF = f"""<mujoco model="tshelf"><worldbody><body name="tshelf" pos="0 0 0">
  <freejoint/>
  <geom type="box" size="0.3 0.2 0.03" pos="0 0 0.60" {COL} mass="3"/>
  <geom type="box" size="0.3 0.2 0.03" pos="0 0 0.30" {COL} mass="3"/>
  <geom type="box" size="0.3 0.2 0.03" pos="0 0 0.03" {COL} mass="3"/>
  <geom type="box" size="0.02 0.02 0.33" pos="0.28 0.18 0.33" {COL} mass="1"/>
  <geom type="box" size="0.02 0.02 0.33" pos="-0.28 -0.18 0.33" {COL} mass="1"/>
</body></worldbody></mujoco>"""

# bowl with a handle arching over its centre (like the wicker basket d00395)
BASKET = (
    BOWL.replace('model="bowl"', 'model="basket"')
    .replace('name="bowl"', 'name="basket"')
    .replace(
        "</body>",
        f'<geom type="box" size="0.08 0.005 0.005" pos="0 0 0.15" {COL} mass="0.02"/></body>',
    )
)


# wire rack: floor wires every 3 cm along x, none on the centre line
RACK = f"""<mujoco model="rack"><worldbody><body name="rack" pos="0 0 0">
  <freejoint/>
  {"".join(f'<geom type="box" size="0.002 0.1 0.002" pos="{x:.3f} 0 0.01" {COL} mass="0.02"/>' for x in (-0.09, -0.06, -0.03, 0.03, 0.06, 0.09))}
  <geom type="box" size="0.1 0.002 0.04" pos="0 0.1 0.04" {COL} mass="0.05"/>
  <geom type="box" size="0.1 0.002 0.04" pos="0 -0.1 0.04" {COL} mass="0.05"/>
</body></worldbody></mujoco>"""


@pytest.fixture
def assets(tmp_path):
    out = {}
    for name, xml in (
        ("bowl", BOWL),
        ("shelf", SHELF),
        ("apple", APPLE),
        ("knife", KNIFE),
        ("tshelf", THICK_SHELF),
        ("basket", BASKET),
        ("rack", RACK),
    ):
        f = tmp_path / name / f"{name}.xml"
        f.parent.mkdir()
        f.write_text(xml)
        out[name] = str(f)
    return out


def _settled(rep, seconds=1.5):
    m = mujoco.MjModel.from_xml_path(rep.scene)
    d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return m, d


def test_spec_parsing():
    assert NestSpec.parse("auto") is None
    s = NestSpec.parse({"in": "bowl", "at": [0.5, 0]})
    assert s.target == "bowl" and s.at == (0.5, 0.0) and s.level == 1
    with pytest.raises(ValueError, match="needs `on:`"):
        NestSpec.parse({"level": 2})
    with pytest.raises(ValueError, match="unknown keys"):
        NestSpec.parse({"on": "x", "inside": True})


def test_yaml_on_key_is_not_a_boolean(assets, tmp_path):
    """YAML 1.1 reads a bare `on` as true: {on: bowl} loads as {True: 'bowl'}."""
    import yaml

    place = yaml.safe_load("apple: {on: bowl, level: 1}\nknife: {in: bowl, upright: true}")
    assert NestSpec.parse(place["apple"]).target == "bowl"
    items = [Item("bowl", assets["bowl"]), Item("apple", assets["apple"], place=place["apple"])]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    assert rep.nested[0]["target"] == "bowl" and not rep.nesting_problems
    assert [i for i in rep.items if i["id"] == "apple"][0]["surface"] == "on bowl"


def test_apple_lands_inside_the_bowl(assets, tmp_path):
    items = [Item("bowl", assets["bowl"]), Item("apple", assets["apple"], place={"in": "bowl"})]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    (n,) = rep.nested
    assert n["target"] == "bowl" and n["note"].startswith("floor")
    assert all(s["ok"] for s in rep.settle)
    m, d = _settled(rep)
    apple, bowl = d.xpos[m.body("apple/apple").id], d.xpos[m.body("bowl/bowl").id]
    assert np.all(np.abs(apple[:2] - bowl[:2]) < 0.07)  # within the walls
    sheet_top = 0.004
    assert apple[2] == pytest.approx(sheet_top + 0.01 + 0.03, abs=0.005)  # on the bowl floor


def test_levels_of_a_shelf_and_chains(assets, tmp_path):
    items = [
        Item("shelf", assets["shelf"]),
        Item("bowl", assets["bowl"], place={"on": "shelf", "level": 2}),
        Item("apple", assets["apple"], place={"in": "bowl"}),  # in the bowl on the shelf
    ]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    assert not rep.nesting_problems
    m, d = _settled(rep)
    base = d.xpos[m.body("shelf/shelf").id][2]
    bowl_z = d.xpos[m.body("bowl/bowl").id][2] - base
    assert bowl_z == pytest.approx(0.31, abs=0.01)  # on the middle board (top at 0.31)
    apple_z = d.xpos[m.body("apple/apple").id][2] - base
    assert apple_z == pytest.approx(0.31 + 0.01 + 0.03, abs=0.01)


def test_items_sharing_a_target_are_spread(assets, tmp_path):
    items = [Item("shelf", assets["shelf"])] + [
        Item(f"apple{i}", assets["apple"], place={"on": "shelf"}) for i in range(4)
    ]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    xy = [np.array(i["pos"][:2]) for i in rep.items if i["id"].startswith("apple")]
    gaps = [np.linalg.norm(a - b) for k, a in enumerate(xy) for b in xy[k + 1 :]]
    assert min(gaps) > 0.08


def test_upright_puts_the_heavy_end_down(assets):
    shape = load_shape(assets["knife"])
    q = rotation(shape, NestSpec("caddy", upright=True))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    assert abs((R @ [1, 0, 0])[2]) == pytest.approx(1.0)  # long axis vertical
    handle = R @ np.array([-0.1, 0, 0])
    assert handle[2] < 0  # handle (heavier) at the bottom


def test_missing_target_is_reported(assets, tmp_path):
    items = [Item("apple", assets["apple"], place={"in": "nowhere"})]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    assert rep.nesting_problems and "nowhere" in rep.nesting_problems[0]


def test_levels_on_thick_boards(assets, tmp_path):
    """The middle tier, not the underside of the top board."""
    items = [
        Item("tshelf", assets["tshelf"]),
        Item("bowl", assets["bowl"], place={"on": "tshelf", "level": 2}),
    ]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    assert rep.nested[0]["note"].startswith("surface 2 (0.63/0.33/0.06")
    assert all(s["ok"] for s in rep.settle)
    m, d = _settled(rep)
    z = d.xpos[m.body("bowl/bowl").id][2] - d.xpos[m.body("tshelf/tshelf").id][2]
    assert z == pytest.approx(0.33, abs=0.01)


def test_in_goes_under_a_handle_to_the_floor(assets, tmp_path):
    items = [
        Item("basket", assets["basket"]),
        Item("apple", assets["apple"], place={"in": "basket"}),
    ]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    assert rep.nested[0]["note"].startswith("floor")
    m, d = _settled(rep)
    apple_z = d.xpos[m.body("apple/apple").id][2] - d.xpos[m.body("basket/basket").id][2]
    assert apple_z == pytest.approx(0.01 + 0.03, abs=0.005)  # on the floor, under the handle
    on_top = build_scene(
        [Item("basket", assets["basket"]), Item("apple", assets["apple"], place={"on": "basket"})],
        tmp_path / "t" / "scene.xml",
    )
    assert on_top.nested[0]["note"].startswith("surface 1 (0.15")  # `on` = the handle's top


def test_open_rack_finds_the_wires_next_to_the_spot(assets, tmp_path):
    items = [Item("rack", assets["rack"]), Item("apple", assets["apple"], place={"in": "rack"})]
    rep = build_scene(items, tmp_path / "s" / "scene.xml")
    note = rep.nested[0]["note"]
    assert note.startswith("floor (0.01") and "cm from the spot" in note
    assert not rep.nesting_problems and all(s["ok"] for s in rep.settle)


def test_upright_item_that_falls_over_is_flagged():
    """A pen stood on end, 10 deg off vertical, falls; the settle check says so."""
    from simready_mujoco.sim.scene import settle_check

    xml = f"""<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
      <body name="pen/root" pos="0 0 0.16" euler="10 0 0"><freejoint/>
        <geom type="capsule" size="0.005 0.15" {COL} mass="0.02"/></body>
      <body name="cup/root" pos="0.5 0 0.05"><freejoint/>
        <geom type="box" size="0.05 0.05 0.05" {COL} mass="0.2"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    items = [
        Item("pen", "-", place={"in": "cup", "upright": True}),
        Item("cup", "-"),
    ]
    out = {s["body"]: s for s in settle_check(m, items, 2.0)}
    assert not out["pen/root"]["ok"] and "fell over" in out["pen/root"]["note"]
    assert out["cup/root"]["ok"] and out["cup/root"]["tilt_deg"] < 1
