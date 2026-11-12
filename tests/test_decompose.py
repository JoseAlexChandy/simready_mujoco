"""Convex decomposition of concave colliders (SimReady fridge: compartments inside the carcass)."""

import json
import math

import mujoco
import pytest
import trimesh
from build import build_hollow_cabinet

from simready_mujoco import convert
from simready_mujoco.native.decompose import concavity, is_concave


def test_concavity_metric():
    box = trimesh.creation.box((0.4, 0.3, 0.2))
    assert concavity(box.vertices, box.faces) < 0.02
    cup = trimesh.creation.annulus(r_min=0.04, r_max=0.05, height=0.1)  # open tube
    shell = trimesh.util.concatenate(
        [
            cup,
            trimesh.creation.box(
                (0.1, 0.1, 0.005), trimesh.transformations.translation_matrix((0, 0, -0.05))
            ),
        ]
    )
    assert is_concave(shell.vertices, shell.faces)


def _door_after_settling(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return math.degrees(d.qpos[m.joint("door_hinge").qposadr[0]])


def test_without_decomposition_the_door_is_blown_open(tmp_path):
    out = convert(build_hollow_cabinet(tmp_path / "h.usda"), tmp_path / "o", decompose="off")
    assert abs(_door_after_settling(out)) > 20


def test_auto_decomposes_only_the_overlapping_concave_collider(tmp_path):
    pytest.importorskip("coacd")
    out = convert(build_hollow_cabinet(tmp_path / "h.usda"), tmp_path / "o", decompose="auto")
    report = json.loads((out.parent / "conversion_report.json").read_text())
    assert set(report["decomposed_meshes"]) == {"carcass_col"}  # compartment is convex: kept
    assert report["decomposed_meshes"]["carcass_col"] > 1
    assert abs(_door_after_settling(out)) < 1.0
    # authored mass untouched; no stale mesh files from the first pass
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.body("carcass").mass[0] == pytest.approx(20.0)
    assert len(list((out.parent / "meshes").iterdir())) == m.nmesh


def test_decomposition_is_cached(tmp_path, monkeypatch):
    pytest.importorskip("coacd")
    monkeypatch.setenv("SRMJ_CACHE", str(tmp_path / "cache"))
    convert(build_hollow_cabinet(tmp_path / "a.usda"), tmp_path / "o1", decompose="concave")
    cached = list((tmp_path / "cache" / "coacd").glob("*.npz"))
    assert len(cached) >= 1
    # second conversion reuses the cache (no new entries)
    convert(build_hollow_cabinet(tmp_path / "b.usda"), tmp_path / "o2", decompose="concave")
    assert len(list((tmp_path / "cache" / "coacd").glob("*.npz"))) == len(cached)
