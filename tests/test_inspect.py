import pytest

from simready_mujoco import inspect_usd


def test_stage_metadata(cabinet_usd):
    r = inspect_usd(cabinet_usd)
    assert r.up_axis == "Z"
    assert r.meters_per_unit == 1.0
    assert r.kilograms_per_unit == 1.0
    assert r.articulations == ["/cabinet"]


def test_bodies_and_mass(cabinet_usd):
    r = inspect_usd(cabinet_usd)
    bodies = {b.path: b for b in r.bodies}
    assert set(bodies) == {"/cabinet/base", "/cabinet/door", "/cabinet/drawer"}
    assert bodies["/cabinet/base"].mass == 2.0
    assert bodies["/cabinet/door"].mass is None
    assert all(b.collider_count == 1 for b in bodies.values())


def test_joints_limits_and_drive(cabinet_usd):
    r = inspect_usd(cabinet_usd)
    joints = {j.path.rsplit("/", 1)[-1]: j for j in r.joints}
    assert joints["door_hinge"].type == "Revolute"
    assert joints["door_hinge"].limits == (0.0, 90.0)  # degrees, as authored
    assert joints["door_hinge"].drive["stiffness"] == 5.0
    assert joints["door_hinge"].drive["max_force"] == 20.0
    assert joints["drawer_slide"].type == "Prismatic"
    assert joints["drawer_slide"].drive is None
    assert joints["base_fixed"].type == "Fixed"


def test_colliders_and_materials(cabinet_usd):
    r = inspect_usd(cabinet_usd)
    assert r.collider_shapes == {"Mesh": 3}
    assert r.mesh_approximations == {"convexHull": 3}
    (mat,) = r.materials
    assert mat["dynamic_friction"] == pytest.approx(0.6)
    assert mat["static_friction"] == pytest.approx(0.8)


def test_reports_what_baseline_drops(cabinet_usd):
    gaps = " ".join(inspect_usd(cabinet_usd).dropped_by_baseline)
    for word in ("mass", "materials", "drives", "colliders"):
        assert word in gaps
