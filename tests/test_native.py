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


# --- mesh robustness (found on real SimReady assets) ---------------------------------


def _add_visual_mesh(usd_path, prim_path, vertices, faces):
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(str(usd_path))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in vertices]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    stage.GetRootLayer().Save()


def test_open_visual_mesh_loads(cabinet_usd, tmp_path):
    # A flat, open mesh has no volume; MuJoCo refuses it unless mesh inertia is "shell".
    # (FR3_Version2_Link5 in the SimReady Franka FR3.)
    square = [(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0, 0.1, 0)]
    _add_visual_mesh(cabinet_usd, "/cabinet/door/door_logo", square, [(0, 1, 2), (0, 2, 3)])
    m, _ = _load(convert(cabinet_usd, tmp_path / "o"))
    assert m.body("base").mass[0] == pytest.approx(2.0)
    assert any(m.mesh(i).name.startswith("door_logo") for i in range(m.nmesh))


def test_tiny_mesh_skipped_with_warning(cabinet_usd, tmp_path):
    # MuJoCo needs >= 4 vertices per mesh
    _add_visual_mesh(
        cabinet_usd, "/cabinet/door/speck", [(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)], [(0, 1, 2)]
    )
    out = convert(cabinet_usd, tmp_path / "o")
    _load(out)
    report = json.loads((out.parent / "conversion_report.json").read_text())
    assert any("speck" in w and "4 vertices" in w for w in report["warnings"])


def test_very_large_mesh_loads(cabinet_usd, tmp_path):
    # MuJoCo's STL reader caps at 200k faces (SimReady refrigerator_a01).
    import trimesh

    big = trimesh.creation.icosphere(subdivisions=7, radius=0.05)  # ~327k faces
    assert len(big.faces) > 200_000
    _add_visual_mesh(cabinet_usd, "/cabinet/base/big_detail", big.vertices, big.faces)
    m, _ = _load(convert(cabinet_usd, tmp_path / "o"))
    big_mesh = [i for i in range(m.nmesh) if m.mesh(i).name.startswith("big_detail")]
    assert big_mesh and m.mesh_facenum[big_mesh[0]] == len(big.faces)


# --- actuator source policy (SimReady FR3: MjcActuator gains 0.003, drive gains = Menagerie) --


def _fr3_like(tmp_path):
    from pxr import Usd, UsdPhysics

    usd = build_runtime_variant_asset(tmp_path / "prop_a01")
    mj = tmp_path / "prop_a01" / "runnables" / "physics" / "mujoco.usda"
    layer = Usd.Stage.Open(str(mj))
    act = layer.GetPrimAtPath("/prop/actuators/hinge_motor")
    act.GetAttribute("mjc:gainPrm").Set([0.003])
    act.GetAttribute("mjc:biasPrm").Set([0.0, 0.0, -0.003])
    layer.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(usd))
    d = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath("/prop/lid_hinge"), "angular")
    d.CreateStiffnessAttr(78.5398178)  # 4500 N·m/rad, per degree
    d.CreateDampingAttr(7.85398178)
    d.CreateMaxForceAttr(87.0)
    stage.GetRootLayer().Save()
    return usd


@pytest.mark.parametrize(
    ("mode", "kp", "warned"),
    [("auto", 4500.0, True), ("drive", 4500.0, False), ("mjc", 0.003, False)],
)
def test_actuator_source_policy(tmp_path, mode, kp, warned):
    out = convert(_fr3_like(tmp_path), tmp_path / "o", actuators=mode)
    m, _ = _load(out)
    assert m.nu == 1
    assert m.actuator_gainprm[0, 0] == pytest.approx(kp, rel=1e-5)
    report = json.loads((out.parent / "conversion_report.json").read_text())
    assert any("no position feedback" in w for w in report["warnings"]) == warned


def test_good_mjc_actuator_beats_drive_in_auto(tmp_path):
    from pxr import Usd, UsdPhysics

    usd = build_runtime_variant_asset(tmp_path / "prop_a01")  # MjcActuator kp=100 with feedback
    stage = Usd.Stage.Open(str(usd))
    UsdPhysics.DriveAPI.Apply(
        stage.GetPrimAtPath("/prop/lid_hinge"), "angular"
    ).CreateStiffnessAttr(1.0)
    stage.GetRootLayer().Save()
    m, _ = _load(convert(usd, tmp_path / "o"))
    assert m.nu == 1 and m.actuator_gainprm[0, 0] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "case", ["closed", "small_holes", "open_rim"], ids=["closed", "repaired", "shell"]
)
def test_mug_mass_comes_from_its_walls_not_its_hull(tmp_path, case):
    """A closed mug mesh: mass and centre of mass from the ceramic walls. With the convex hull
    it was ~6x too heavy and its centre of mass sat out toward the handle, so it tipped over."""
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    T = trimesh.transformations.translation_matrix
    wall = trimesh.creation.annulus(r_min=0.035, r_max=0.04, height=0.1)
    base = trimesh.creation.cylinder(radius=0.0349, height=0.006, transform=T((0, 0, -0.047)))
    handle = trimesh.creation.box((0.01, 0.044, 0.06), T((0, -0.063, 0)))
    mug = trimesh.util.concatenate([wall, base, handle])  # closed, disjoint pieces
    walls = abs(mug.volume) * 1000.0
    if case == "small_holes":  # a few isolated missing triangles: filled -> solid
        keep = np.ones(len(mug.faces), bool)
        in_base = np.flatnonzero(
            (mug.triangles_center[:, 2] < -0.049)
            & (np.hypot(*mug.triangles_center[:, :2].T) < 0.03)
        )
        keep[in_base[[0, len(in_base) // 2]]] = False  # two small floor triangles
        mug = trimesh.Trimesh(mug.vertices, mug.faces[keep], process=False)
    elif case == "open_rim":  # the wall's top ring is missing: too big to fill -> thin shell
        top = mug.triangles_center[:, 2] > 0.0499
        mug = trimesh.Trimesh(mug.vertices, mug.faces[~top], process=False)
    mug.apply_translation((0, 0, 0.05))

    path = tmp_path / "mug.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/mug").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdGeom.Mesh.Define(stage, "/mug/mesh")
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mug.vertices])
    m.CreateFaceVertexCountsAttr([3] * len(mug.faces))
    m.CreateFaceVertexIndicesAttr([int(i) for f in mug.faces for i in f])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()

    out = convert(path, tmp_path / "o")
    mj = mujoco.MjModel.from_xml_path(str(out))
    b = mj.body("mug").id
    report = json.loads((out.parent / "conversion_report.json").read_text())
    expected = {"closed": "mesh", "small_holes": "mesh", "open_rim": "shell"}[case]
    assert report["mass_from"][expected] == 1
    if expected == "mesh":
        assert mj.body_mass[b] == pytest.approx(walls, rel=0.05)
    else:
        assert 0.2 * walls < mj.body_mass[b] < 3 * walls
    # centre of mass stays over the base (radius 0.04), not out at the handle
    assert np.hypot(*mj.body_ipos[b][:2]) < 0.02


def test_pinholes_sharing_a_vertex_are_filled_separately():
    """Two missing triangles of a fan cap share its centre vertex; each is filled on its own,
    and big openings are left alone (the mesh is then read as a surface)."""
    import trimesh

    from simready_mujoco.native.parse import _fill_pinholes, _simple_cycles

    assert _simple_cycles([1, 2, 3, 1, 4, 5]) == [[1, 2, 3], [1, 4, 5]]
    cyl = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=64)
    cyl.merge_vertices()
    cap = np.flatnonzero(cyl.triangles_center[:, 2] > 0.049)
    holed = trimesh.Trimesh(cyl.vertices, np.delete(cyl.faces, cap[[0, 5]], axis=0), process=False)
    filled = _fill_pinholes(holed, 0.01)
    assert filled is not None and filled.is_watertight
    assert filled.volume == pytest.approx(cyl.volume, rel=1e-6)
    open_top = trimesh.Trimesh(cyl.vertices, np.delete(cyl.faces, cap, axis=0), process=False)
    assert _fill_pinholes(open_top, 0.01) is None
