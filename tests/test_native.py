"""Native backend: USD -> IR -> MJCF, checked by compiling and simulating in MuJoCo."""

import json
import math

import mujoco
import numpy as np
import pytest
from build import build_mimic_gripper, build_runtime_variant_asset

from simready_mujoco import convert


def _load(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.fixture
def cabinet(cabinet_usd, tmp_path):
    return convert(cabinet_usd, tmp_path / "out")


def test_report_written(cabinet):
    report = json.loads((cabinet.parent / "conversion_report.json").read_text())
    assert report["bodies"] == 3
    assert report["actuators_from_drives"] == 1
    assert report["warnings"] == []


def test_authored_mass_respected(cabinet):
    m, _ = _load(cabinet)
    assert m.body("base").mass[0] == pytest.approx(2.0)


def test_unauthored_mass_computed_from_collider_volume(cabinet):
    m, _ = _load(cabinet)
    # 0.02 x 0.4 x 0.6 m door at USD's default density 1000 kg/m^3
    assert m.body("door").mass[0] == pytest.approx(0.02 * 0.4 * 0.6 * 1000, rel=1e-3)


def test_hinge_anchor_on_cabinet_edge(cabinet):
    m, d = _load(cabinet)
    assert np.allclose(d.xanchor[m.joint("door_hinge").id], [0.21, 0.2, 0.4], atol=1e-6)
    assert np.allclose(d.xanchor[m.joint("drawer_slide").id], [0.0, 0.0, 0.1], atol=1e-6)


def test_limits_in_radians_and_metres(cabinet):
    m, _ = _load(cabinet)
    assert np.allclose(m.joint("door_hinge").range, [0.0, math.pi / 2])
    assert np.allclose(m.joint("drawer_slide").range, [0.0, 0.3], atol=1e-6)


def test_drive_becomes_pd_actuator_in_si_units(cabinet):
    m, _ = _load(cabinet)
    assert m.nu == 1
    kp = 5.0 * 180 / math.pi  # USD: 5 N·m per degree
    kv = 0.5 * 180 / math.pi
    assert m.actuator_gainprm[0, 0] == pytest.approx(kp)
    assert np.allclose(m.actuator_biasprm[0, :3], [0, -kp, -kv])
    assert np.allclose(m.actuator_forcerange[0], [-20, 20])


def test_drive_reaches_target(cabinet):
    m, d = _load(cabinet)
    d.ctrl[0] = math.radians(45)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qpos[m.joint("door_hinge").qposadr[0]] == pytest.approx(math.radians(45), abs=0.02)


def test_friction_from_physics_material(cabinet):
    m, _ = _load(cabinet)
    collision = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    assert collision and all(m.geom_friction[g, 0] == pytest.approx(0.6) for g in collision)


def test_visual_geoms_do_not_collide(cabinet):
    m, _ = _load(cabinet)
    visual = [g for g in range(m.ngeom) if m.geom_group[g] == 2]
    assert visual and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in visual)


def test_base_is_welded_to_world(cabinet):
    m, _ = _load(cabinet)
    assert m.body("base").jntnum[0] == 0 and m.body("base").parentid[0] == 0


# --- SimReady multi-runtime asset: MjcActuator passthrough ------------------------------


@pytest.fixture
def prop(tmp_path):
    return build_runtime_variant_asset(tmp_path / "prop_a01")


def test_mjc_actuator_passed_through(prop, tmp_path):
    m, _ = _load(convert(prop, tmp_path / "o", runtime="mujoco"))
    assert m.nu == 1
    assert m.actuator_gainprm[0, 0] == pytest.approx(100.0)
    assert np.allclose(m.actuator_biasprm[0, :3], [0, -100, -10])
    assert np.allclose(m.actuator_forcerange[0], [-5, 5])
    assert m.joint(m.actuator_trnid[0, 0]).name == "lid_hinge"


def test_unattached_body_gets_free_joint(prop, tmp_path):
    m, _ = _load(convert(prop, tmp_path / "o", runtime="mujoco"))
    assert m.body("body").jntnum[0] == 1
    assert m.jnt_type[m.body("body").jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE


def test_mjc_collision_attrs(prop, tmp_path):
    m, _ = _load(convert(prop, tmp_path / "o", runtime="mujoco"))
    assert m.geom("body_collision").margin[0] == pytest.approx(0.001)


# --- mimic joints and joint reference angle -------------------------------------------


@pytest.fixture
def gripper(tmp_path):
    return convert(build_mimic_gripper(tmp_path / "gripper.usda"), tmp_path / "o")


def test_mimic_equalities(gripper):
    m, _ = _load(gripper)
    eq = {m.equality(i).name: i for i in range(m.neq)}
    assert set(eq) == {"physx_follower_mimic", "newton_follower_mimic"}
    lead0 = m.qpos0[m.joint("lead_joint").qposadr[0]]  # 10 deg reference angle
    # MuJoCo: f - f0 = a0 + a1*(l - l0); USD couplings are absolute, so a0 = c0 - f0 + c1*l0.
    # PhysX: q + g*q_ref + o = 0 with g = -1  =>  q = +q_ref
    assert np.allclose(m.eq_data[eq["physx_follower_mimic"], :2], [lead0, 1.0])
    # Newton: q = c0 + c1*q_ref with c1 = -1
    assert np.allclose(m.eq_data[eq["newton_follower_mimic"], :2], [-lead0, -1.0])


def test_followers_track_leader_in_simulation(gripper):
    m, d = _load(gripper)
    d.ctrl[0] = math.radians(30)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    def q(n):
        return d.qpos[m.joint(n).qposadr[0]]

    assert q("lead_joint") == pytest.approx(math.radians(30), abs=0.03)
    assert q("physx_follower") == pytest.approx(q("lead_joint"), abs=0.02)
    assert q("newton_follower") == pytest.approx(-q("lead_joint"), abs=0.02)


def test_joint_ref_from_authored_pose(gripper):
    m, _ = _load(gripper)
    # finger authored rotated 10 deg about the hinge axis -> MJCF ref = 10 deg
    assert m.qpos0[m.joint("lead_joint").qposadr[0]] == pytest.approx(math.radians(10), abs=1e-4)


def test_jointed_bodies_do_not_collide(gripper):
    # USD joints have physics:collisionEnabled = false by default; the palm is welded to the
    # world, so MuJoCo's own parent/child filter would not apply.
    m, d = _load(gripper)
    pairs = {
        tuple(
            sorted(
                (
                    m.body(m.exclude_signature[i] >> 16).name,
                    m.body(m.exclude_signature[i] & 0xFFFF).name,
                )
            )
        )
        for i in range(m.nexclude)
    }
    assert ("finger_lead", "palm") in pairs
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert d.ncon == 0


def test_fixed_tendon_and_tendon_actuator(tmp_path):
    from pxr import Sdf, Usd

    usd = build_mimic_gripper(tmp_path / "g.usda")
    stage = Usd.Stage.Open(str(usd))
    td = stage.DefinePrim("/gripper/Physics/split", "MjcTendon")
    td.CreateAttribute("mjc:type", Sdf.ValueTypeNames.Token).Set("fixed")
    td.CreateRelationship("mjc:path").SetTargets(
        [Sdf.Path("/gripper/joints/lead_joint"), Sdf.Path("/gripper/joints/physx_follower")]
    )
    td.CreateAttribute("mjc:path:coef", Sdf.ValueTypeNames.DoubleArray).Set([0.5, 0.5])
    td.CreateAttribute("mjc:path:indices", Sdf.ValueTypeNames.IntArray).Set([0, 1])
    act = stage.DefinePrim("/gripper/Physics/grip", "MjcActuator")
    act.CreateRelationship("mjc:target").SetTargets([td.GetPath()])
    act.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray).Set([2.0])
    stage.GetRootLayer().Save()

    m, _ = _load(convert(usd, tmp_path / "o"))
    assert m.ntendon == 1 and m.tendon(0).name == "split"
    assert np.allclose(m.wrap_prm[: m.nwrap], [0.5, 0.5])
    grip = m.actuator("grip")
    assert int(m.actuator_trntype[grip.id]) == int(mujoco.mjtTrn.mjTRN_TENDON)
    assert m.actuator_gainprm[grip.id, 0] == pytest.approx(2.0)
