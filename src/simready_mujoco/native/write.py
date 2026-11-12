"""IR Model -> MJCF XML + mesh files."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simready_mujoco.ir import Geom, JointType, MeshData, Model, ShapeType

STL_MAX_FACES = 200_000  # MuJoCo's STL reader limit; larger meshes are written as OBJ
VISUAL_GROUP = 2
COLLISION_GROUP = 3


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.9g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(_fmt(float(x)) if not isinstance(x, str) else x for x in v)
    return str(v)


def _set(el: ET.Element, **attrs) -> ET.Element:
    for k, v in attrs.items():
        if v is not None:
            el.set(k, _fmt(v))
    return el


def write_stl(path: Path, md: MeshData) -> None:
    """Binary STL (fast, compact; MuJoCo reads it natively)."""
    v = np.asarray(md.vertices, dtype=np.float32)
    f = np.asarray(md.faces, dtype=np.int64)
    tri = v[f]  # (M, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0).astype(np.float32)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"simready-mujoco".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def write_obj(path: Path, md: MeshData) -> None:
    """Wavefront OBJ (no face limit in MuJoCo)."""
    v = np.asarray(md.vertices, dtype=np.float64)
    f = np.asarray(md.faces, dtype=np.int64) + 1
    with open(path, "w") as fh:
        fh.write("# simready-mujoco\n")
        np.savetxt(fh, v, fmt="v %.7g %.7g %.7g")
        np.savetxt(fh, f, fmt="f %d %d %d")


def _geom_el(parent: ET.Element, g: Geom) -> None:
    el = ET.SubElement(parent, "geom", name=g.name, type=g.shape.value)
    el.set("class", "collision" if g.collision else "visual")
    if g.shape == ShapeType.MESH:
        el.set("mesh", g.mesh)
    else:
        _set(el, size=list(g.size))
    if any(abs(c) > 1e-12 for c in g.pose.pos):
        _set(el, pos=list(g.pose.pos))
    if tuple(round(c, 12) for c in g.pose.quat) != (1.0, 0.0, 0.0, 0.0):
        _set(el, quat=list(g.pose.quat))
    if g.rgba and not g.collision:
        _set(el, rgba=list(g.rgba))
    if g.friction is not None:
        _set(el, friction=list(g.friction))
    for k, v in g.extra.items():
        _set(el, **{k: v})


def write_mjcf(model: Model, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    root = ET.Element("mujoco", model=model.name)
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir="meshes",
        autolimits="true",
        inertiafromgeom="auto",
        balanceinertia="true",
    )
    opt = _set(ET.SubElement(root, "option"), gravity=list(model.gravity))
    for k, v in model.option.items():
        _set(opt, **{k: v})
    if model.flags:
        _set(ET.SubElement(opt, "flag"), **model.flags)

    default = ET.SubElement(root, "default")
    vis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(
        vis, "geom", contype="0", conaffinity="0", group=str(VISUAL_GROUP), density="0", mass="0"
    )
    col = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(col, "geom", group=str(COLLISION_GROUP), rgba="0.9 0.4 0.2 0.4", mass="0")

    asset = ET.SubElement(root, "asset")
    for md in model.meshes.values():
        if len(md.faces) > STL_MAX_FACES:
            fname = f"{md.name}.obj"
            write_obj(mesh_dir / fname, md)
        else:
            fname = f"{md.name}.stl"
            write_stl(mesh_dir / fname, md)
        el = ET.SubElement(asset, "mesh", name=md.name, file=fname)
        # Masses always come from explicit <inertial>, so mesh volumes are never needed; "shell"
        # stops MuJoCo rejecting open (non-watertight) visual meshes. An authored mode wins.
        _set(el, maxhullvert=md.maxhullvert, inertia=md.inertia or "shell")

    world = ET.SubElement(root, "worldbody")
    for g in model.world_geoms:
        _geom_el(world, g)

    elements: dict[str, ET.Element] = {"": world}
    for b in model.bodies:  # parents precede children
        el = ET.SubElement(elements[b.parent], "body", name=b.name)
        _set(el, pos=list(b.pose.pos), quat=list(b.pose.quat))
        elements[b.name] = el
        if b.inertial is not None:
            i = b.inertial
            _set(
                ET.SubElement(el, "inertial"),
                pos=list(i.pos),
                quat=list(i.quat),
                mass=float(i.mass),
                diaginertia=list(i.diaginertia),
            )
        for j in b.joints:
            if j.type == JointType.FREE:
                ET.SubElement(el, "freejoint", name=j.name)
                continue
            jel = ET.SubElement(el, "joint", name=j.name, type=j.type.value)
            if any(abs(c) > 1e-12 for c in j.pos):
                _set(jel, pos=list(j.pos))
            if j.type != JointType.BALL:
                _set(jel, axis=list(j.axis))
            if j.range is not None:
                _set(jel, range=list(j.range))
            if abs(j.ref) > 1e-9 and j.type in (JointType.HINGE, JointType.SLIDE):
                _set(jel, ref=j.ref)
            for k, v in j.extra.items():
                _set(jel, **{k: v})
        for g in b.geoms:
            _geom_el(el, g)

    if model.excludes:
        contact = ET.SubElement(root, "contact")
        for a, b in model.excludes:
            ET.SubElement(contact, "exclude", body1=a, body2=b)

    if model.tendons:
        tendon = ET.SubElement(root, "tendon")
        for td in model.tendons:
            fixed = _set(ET.SubElement(tendon, "fixed", name=td.name), **td.attrs)
            for jname, coef in td.joints:
                _set(ET.SubElement(fixed, "joint", joint=jname), coef=coef)

    if model.equalities:
        eq = ET.SubElement(root, "equality")
        for e in model.equalities:
            _set(ET.SubElement(eq, e.kind, name=e.name), **e.attrs)

    if model.actuators:
        act = ET.SubElement(root, "actuator")
        for a in model.actuators:
            el = ET.SubElement(act, "general", {"name": a.name, a.target: a.joint})
            _set(el, **a.attrs)

    ET.indent(root)
    out = out_dir / f"{model.name}.xml"
    ET.ElementTree(root).write(out, encoding="utf-8")
    return out
