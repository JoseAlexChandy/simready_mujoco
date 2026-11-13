"""Visual appearance: materials, textures, UVs and per-material mesh splitting.

SimReady assets describe appearance with UsdPreviewSurface networks (diffuseColor either a
constant or a UsdUVTexture image read through a UsdPrimvarReader "st"), usually next to an
OmniPBR (MDL) shader with the same inputs. Big meshes often carry several materials through
GeomSubsets (family "materialBind"), e.g. the laptop's plastic, keyboard and screen.

MuJoCo has one material per geom, a single 2D texture per material, and per-vertex UVs, so:
  - each mesh is split into one geom per material (GeomSubset),
  - vertices are welded per (position, uv) pair,
  - the base-colour texture becomes an MJCF <texture> (PNG); normal/ORM maps are ignored,
  - roughness maps to MuJoCo shininess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

DEFAULT_RGBA = (0.7, 0.7, 0.7, 1.0)


@dataclass
class MaterialSpec:
    key: str  # USD material path (or "displayColor:<rgb>")
    name: str
    rgba: tuple[float, float, float, float]
    texture: str | None = None  # absolute image path
    uv_primvar: str = "st"
    roughness: float | None = None
    specular: float | None = None


# ----------------------------------------------------------------------------- materials


def _asset_path(attr: Usd.Attribute | None, value) -> str | None:
    """Absolute path of an asset-valued attribute (resolved, or relative to its layer)."""
    if not isinstance(value, Sdf.AssetPath) or not value.path:
        return None
    if value.resolvedPath:
        return value.resolvedPath
    if attr is not None:
        for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
            layer = spec.layer
            if layer and layer.realPath:
                cand = os.path.normpath(os.path.join(os.path.dirname(layer.realPath), value.path))
                if os.path.exists(cand):
                    return cand
    return None


def _source_shader(inp: UsdShade.Input) -> tuple[UsdShade.Shader | None, str]:
    if not inp or not inp.HasConnectedSource():
        return None, ""
    sources = inp.GetConnectedSources()[0]
    if not sources:
        return None, ""
    src = sources[0]
    return UsdShade.Shader(src.source.GetPrim()), str(src.sourceName)


def _value(shader: UsdShade.Shader, name: str):
    inp = shader.GetInput(name)
    if not inp:
        return None
    for attr in inp.GetValueProducingAttributes():
        v = attr.Get()
        if v is not None:
            return v
    return inp.Get()


def _uv_name(tex: UsdShade.Shader) -> str:
    reader, _ = _source_shader(tex.GetInput("st"))
    if reader:
        name = _value(reader, "varname")
        if name:
            return str(name)
    return "st"


def _from_preview_surface(shader: UsdShade.Shader, spec: MaterialSpec) -> bool:
    if (shader.GetIdAttr().Get() or "") != "UsdPreviewSurface":
        return False
    diffuse = shader.GetInput("diffuseColor")
    tex, _ = _source_shader(diffuse)
    if tex and (tex.GetIdAttr().Get() or "") == "UsdUVTexture":
        file_inp = tex.GetInput("file")
        attrs = file_inp.GetValueProducingAttributes() if file_inp else []
        attr = attrs[0] if attrs else (file_inp.GetAttr() if file_inp else None)
        spec.texture = _asset_path(attr, attr.Get() if attr else None)
        spec.uv_primvar = _uv_name(tex)
        fallback = _value(tex, "fallback")
        if fallback is not None and spec.texture is None:
            spec.rgba = (float(fallback[0]), float(fallback[1]), float(fallback[2]), 1.0)
    else:
        c = _value(shader, "diffuseColor")
        if c is not None:
            spec.rgba = (float(c[0]), float(c[1]), float(c[2]), 1.0)
    op = _value(shader, "opacity")
    if isinstance(op, (int, float)):
        spec.rgba = (*spec.rgba[:3], float(op))
    r = _value(shader, "roughness")
    spec.roughness = float(r) if isinstance(r, (int, float)) else None
    s = _value(shader, "specular")
    spec.specular = float(s) if isinstance(s, (int, float)) else None
    return True


def _from_omnipbr(shader: UsdShade.Shader, spec: MaterialSpec) -> bool:
    inp = shader.GetInput("diffuse_texture")
    tex = None
    if inp:
        attrs = inp.GetValueProducingAttributes() or [inp.GetAttr()]
        tex = _asset_path(attrs[0], attrs[0].Get())
    color = _value(shader, "diffuse_color_constant")
    tint = _value(shader, "diffuse_tint")
    if tex is None and color is None:
        return False
    spec.texture = tex
    if color is not None:
        spec.rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if tint is not None and tex is not None:
        spec.rgba = (float(tint[0]), float(tint[1]), float(tint[2]), 1.0)
    r = _value(shader, "reflection_roughness_constant")
    spec.roughness = float(r) if isinstance(r, (int, float)) else None
    op = _value(shader, "opacity_constant")
    if isinstance(op, (int, float)) and _value(shader, "enable_opacity"):
        spec.rgba = (*spec.rgba[:3], float(op))
    return True


def resolve_material(material: UsdShade.Material, name: str) -> MaterialSpec:
    """Appearance of a bound USD material: UsdPreviewSurface first, then OmniPBR (MDL)."""
    spec = MaterialSpec(str(material.GetPath()), name, DEFAULT_RGBA)
    shader = material.ComputeSurfaceSource()[0]
    if shader and _from_preview_surface(shader, spec):
        return spec
    mdl = material.ComputeSurfaceSource(["mdl"])[0]
    if mdl and _from_omnipbr(mdl, spec):
        return spec
    # Last resort: any shader in the material network that looks like either.
    for prim in Usd.PrimRange(material.GetPrim()):
        sh = UsdShade.Shader(prim)
        if sh and (_from_preview_surface(sh, spec) or _from_omnipbr(sh, spec)):
            return spec
    return spec


def display_color(prim: Usd.Prim) -> tuple[float, float, float, float] | None:
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    if pv and pv.HasAuthoredValue():
        vals = pv.Get()
        if vals is not None and len(vals) > 0:
            c = vals[0]
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return None


# ----------------------------------------------------------------------------- meshes


@dataclass
class Topology:
    points: np.ndarray  # (N, 3)
    tris: np.ndarray  # (T, 3) point indices
    tri_face: np.ndarray  # (T,) source polygon index
    tri_corner: np.ndarray  # (T, 3) index into faceVertexIndices (for faceVarying data)


def topology(mesh: UsdGeom.Mesh) -> Topology | None:
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if pts is None or counts is None or idx is None or len(pts) == 0:
        return None
    points = np.asarray(pts, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    faces, corners = [], []
    for n in np.unique(counts):  # vectorised fan triangulation per polygon size
        if n < 3:
            continue
        sel = np.nonzero(counts == n)[0]
        s = starts[sel]
        for k in range(1, n - 1):
            c = np.stack([s, s + k, s + k + 1], axis=1)
            corners.append(c)
            faces.append(sel)
    if not corners:
        return None
    tri_corner = np.concatenate(corners)
    tri_face = np.concatenate(faces)
    order = np.argsort(tri_face, kind="stable")
    tri_corner, tri_face = tri_corner[order], tri_face[order]
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        tri_corner = tri_corner[:, ::-1]
    return Topology(points, idx[tri_corner], tri_face, tri_corner)


def corner_uvs(prim: Usd.Prim, name: str, topo: Topology) -> np.ndarray | None:
    """(T, 3, 2) UV per triangle corner, from the named texcoord primvar (or the first one)."""
    api = UsdGeom.PrimvarsAPI(prim)
    pv = api.GetPrimvar(name) if name else None
    if not pv or not pv.HasValue():
        pv = next(
            (
                p
                for p in api.GetPrimvars()
                if p.HasValue() and str(p.GetTypeName().role) == "TextureCoordinate"
            ),
            None,
        )
    if not pv:
        return None
    vals = pv.ComputeFlattened()
    if vals is None or len(vals) == 0:
        return None
    uv = np.asarray(vals, dtype=np.float64)[:, :2]
    interp = pv.GetInterpolation()
    try:
        if interp == UsdGeom.Tokens.faceVarying:
            return uv[topo.tri_corner]
        if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            return uv[topo.tris]
        if interp == UsdGeom.Tokens.uniform:
            return np.repeat(uv[topo.tri_face][:, None, :], 3, axis=1)
        return np.broadcast_to(uv[0], (len(topo.tris), 3, 2)).copy()
    except IndexError:
        return None


def material_groups(mesh: UsdGeom.Mesh, n_faces: int) -> list[tuple[Usd.Prim, np.ndarray | None]]:
    """[(prim whose binding applies, face indices or None = all remaining faces)].
    GeomSubsets of family "materialBind" each get their own group."""
    subsets = UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, "materialBind") or [
        s for s in UsdGeom.Subset.GetAllGeomSubsets(mesh) if s.GetElementTypeAttr().Get() == "face"
    ]
    groups: list[tuple[Usd.Prim, np.ndarray | None]] = []
    covered = np.zeros(n_faces, dtype=bool)
    for s in subsets:
        ids = np.asarray(s.GetIndicesAttr().Get() or [], dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < n_faces) & ~covered[np.clip(ids, 0, n_faces - 1)]]
        if len(ids):
            covered[ids] = True
            groups.append((s.GetPrim(), ids))
    rest = np.nonzero(~covered)[0]
    if not groups:
        return [(mesh.GetPrim(), None)]
    if len(rest):
        groups.append((mesh.GetPrim(), rest))
    return groups


def weld(
    points: np.ndarray, tris: np.ndarray, uvs: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compact vertices to those used by `tris`, splitting where UVs differ (seams)."""
    flat = tris.reshape(-1)
    if uvs is None:
        used, inverse = np.unique(flat, return_inverse=True)
        return points[used], inverse.reshape(-1, 3), None
    uv = uvs.reshape(-1, 2)
    keys = np.column_stack([flat.astype(np.float64), np.round(uv * 1e6)])
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return points[flat[first]], inverse.reshape(-1, 3), uv[first]
