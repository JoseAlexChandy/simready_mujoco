"""Physics runtime variant selection on SimReady-style multi-runtime assets."""

import pytest
from build import build_runtime_variant_asset
from pxr import Usd

from simready_mujoco import inspect_usd
from simready_mujoco.usd.stage import applied_schemas, open_stage, parse_variant_args


@pytest.fixture
def multi_runtime_usd(tmp_path):
    return build_runtime_variant_asset(tmp_path / "prop_a01")


def test_authored_default_is_physx(multi_runtime_usd):
    r = inspect_usd(multi_runtime_usd, runtime="authored")
    assert "PhysxCollisionAPI" in r.physx_schemas
    assert r.mjc_schemas == []
    assert r.variants["PhysX"]["selected"] == "Enabled"


def test_mujoco_runtime_composes_mujoco_layer_only(multi_runtime_usd):
    r = inspect_usd(multi_runtime_usd, runtime="mujoco")
    assert r.runtime_available is True
    assert r.selection == {"PhysX": "Disabled", "MuJoCo": "Enabled"}
    assert r.mjc_schemas == ["MjcCollisionAPI"]
    assert "PhysxCollisionAPI" not in r.physx_schemas  # PhysX layer not composed
    assert r.mjc_prim_types == {"MjcActuator": 1}
    assert "mjc:margin" in r.mjc_attributes


def test_missing_runtime_is_reported(multi_runtime_usd):
    r = inspect_usd(multi_runtime_usd, runtime="newton")
    assert r.runtime_available is False


def test_explicit_variant_wins(multi_runtime_usd):
    r = inspect_usd(multi_runtime_usd, runtime="mujoco", variants={"PhysX": "Enabled"})
    assert r.selection["PhysX"] == "Enabled"
    assert "PhysxCollisionAPI" in r.physx_schemas


def test_selection_does_not_modify_file(multi_runtime_usd):
    before = multi_runtime_usd.read_text()
    open_stage(multi_runtime_usd, runtime="mujoco")
    assert multi_runtime_usd.read_text() == before


def test_unregistered_mimic_schema_detected(multi_runtime_usd):
    # Usd.Prim.GetAppliedSchemas() drops unregistered schemas; we must not.
    stage = Usd.Stage.Open(str(multi_runtime_usd))
    hinge = stage.GetPrimAtPath("/prop/lid_hinge")
    assert "PhysxMimicJointAPI:rotZ" not in hinge.GetAppliedSchemas()  # the USD pitfall
    assert "PhysxMimicJointAPI:rotZ" in applied_schemas(hinge)
    joints = {j.path: j for j in inspect_usd(multi_runtime_usd).joints}
    assert joints["/prop/lid_hinge"].mimic


def test_parse_variant_args():
    assert parse_variant_args(["Physics=Newton_parallel_grip", " A = b "]) == {
        "Physics": "Newton_parallel_grip",
        "A": "b",
    }
    with pytest.raises(ValueError):
        parse_variant_args(["nope"])


def test_unknown_runtime_rejected(multi_runtime_usd):
    with pytest.raises(ValueError, match="Unknown runtime"):
        open_stage(multi_runtime_usd, runtime="bullet")
