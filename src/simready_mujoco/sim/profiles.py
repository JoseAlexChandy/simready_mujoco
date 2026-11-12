"""Simulation presets (MuJoCo <option>) and applying them to an MJCF file."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILES_FILE = Path(__file__).resolve().parents[1] / "sim_profiles.yaml"


def load_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    with open(path or DEFAULT_PROFILES_FILE) as f:
        return yaml.safe_load(f)


def get_profile(name: str, path: Path | None = None) -> dict[str, Any]:
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f"Unknown sim profile '{name}'. Available: {', '.join(profiles)}")
    return profiles[name]


def apply_profile(mjcf_file: Path, profile: dict[str, Any]) -> None:
    """Overwrite <option> (and its <flag>) in an MJCF file with the profile's settings."""
    tree = ET.parse(mjcf_file)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        idx = list(root).index(compiler) + 1 if compiler is not None else 0
        root.insert(idx, option)

    for key, value in profile.items():
        if key == "flags":
            flag = option.find("flag")
            if flag is None:
                flag = ET.SubElement(option, "flag")
            for fk, fv in value.items():
                flag.set(fk, str(fv))
        else:
            option.set(key, str(value))

    ET.indent(tree)
    tree.write(mjcf_file, encoding="utf-8", xml_declaration=False)
