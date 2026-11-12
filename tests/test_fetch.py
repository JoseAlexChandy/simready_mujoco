"""Offline tests for resolving SimReady package entries (no network)."""

import json

import pytest

from simready_mujoco.fetch import (
    PACKAGING_FILE,
    find_interface_usd,
    parse_package_url,
    read_package_info,
    resolve_asset,
)

URL = (
    "https://huggingface.co/datasets/Robotiq-Official/simready-assets/resolve/"
    "3c296ea93075d743e8ef9a0324d374b639d28123/Robotiq_2F_85/" + PACKAGING_FILE
)


def test_parse_package_url():
    ref = parse_package_url(URL)
    assert ref.repo == "Robotiq-Official/simready-assets"
    assert ref.revision == "3c296ea93075d743e8ef9a0324d374b639d28123"
    assert ref.asset == "Robotiq_2F_85"


def test_parse_rejects_other_urls():
    with pytest.raises(ValueError, match="Not a SimReady package URL"):
        parse_package_url("https://example.com/asset.usd")


def _make_asset(root, name="Robotiq_2F_85", usds=("simready_usd/Robotiq_2F_85.usd",)):
    asset = root / name
    for u in usds:
        p = asset / u
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#usda 1.0\n")
    thumbs = asset / "simready_usd" / ".thumbs" / "256x256"
    thumbs.mkdir(parents=True)
    (thumbs / "Robotiq_2F_85.usd.png").write_bytes(b"")
    (asset / PACKAGING_FILE).write_text(
        json.dumps({"package_id": "robotiq.2f85", "license": "CC-BY-4.0", "format_version": 1})
    )
    return asset


def test_find_interface_usd_simready_layout(tmp_path):
    asset = _make_asset(tmp_path)
    assert find_interface_usd(asset).name == "Robotiq_2F_85.usd"


def test_find_interface_prefers_asset_named_file(tmp_path):
    asset = _make_asset(
        tmp_path, usds=("simready_usd/Robotiq_2F_85.usd", "simready_usd/helper_layer.usd")
    )
    assert find_interface_usd(asset).name == "Robotiq_2F_85.usd"


def test_find_interface_ambiguous_asks_for_hint(tmp_path):
    asset = _make_asset(tmp_path, usds=("a/one.usd", "a/two.usd"))
    with pytest.raises(ValueError, match="interface"):
        find_interface_usd(asset)
    assert find_interface_usd(asset, hint="a/two.usd").name == "two.usd"


def test_read_package_info(tmp_path):
    info = read_package_info(_make_asset(tmp_path))
    assert info["license"] == "CC-BY-4.0"
    assert info["package_id"] == "robotiq.2f85"


def test_resolve_simready_entry_uses_download(tmp_path, monkeypatch):
    _make_asset(tmp_path)
    monkeypatch.setattr("simready_mujoco.fetch._download", lambda repo, rev, patterns: tmp_path)
    res = resolve_asset({"id": "g", "simready": URL})
    assert res.usd.name == "Robotiq_2F_85.usd"
    assert res.license == "CC-BY-4.0"


def test_resolve_local_path(cabinet_usd):
    assert resolve_asset({"id": "c", "path": str(cabinet_usd)}).usd == cabinet_usd
