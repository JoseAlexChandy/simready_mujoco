"""Transform and mesh helpers (USD Gf <-> plain numpy / tuples)."""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def quat_tuple(q) -> tuple[float, float, float, float]:
    """Gf.Quat* -> (w, x, y, z), normalised."""
    w = float(q.GetReal())
    x, y, z = (float(c) for c in q.GetImaginary())
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (w / n, x / n, y / n, z / n)


def decompose(m: Gf.Matrix4d) -> tuple[Gf.Matrix4d, np.ndarray]:
    """Split a USD (row-vector) matrix into a rigid matrix and a per-axis scale.

    Assumes no shear (scale applied before rotation, the standard xformOp order).
    A mirrored transform is returned as a proper rotation with one negative scale.
    """
    rows = np.array([[m[i][j] for j in range(3)] for i in range(3)], dtype=float)
    scale = np.linalg.norm(rows, axis=1)
    scale[scale == 0] = 1.0
    rot = rows / scale[:, None]
    if np.linalg.det(rot) < 0:
        scale[0] = -scale[0]
        rot[0] = -rot[0]
    rigid = Gf.Matrix4d(1.0)
    rigid.SetRotateOnly(Gf.Matrix3d(*rot.flatten().tolist()))
    rigid.SetTranslateOnly(m.ExtractTranslation())
    return rigid, scale


def rigid_pose(m: Gf.Matrix4d, length_scale: float = 1.0):
    """Rigid matrix -> (pos, quat) tuples, with positions scaled to metres."""
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    return tuple(float(c) * length_scale for c in t), quat_tuple(q)


def pose_matrix(pos, quat) -> Gf.Matrix4d:
    """(pos Gf.Vec3, quat Gf.Quat) -> rigid row-vector matrix (rotate, then translate)."""
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Rotation(Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(quat.GetImaginary()))))
    m.SetTranslateOnly(Gf.Vec3d(pos))
    return m


def relative(child_world: Gf.Matrix4d, parent_world: Gf.Matrix4d) -> Gf.Matrix4d:
    """Child transform expressed in the parent frame (row-vector convention)."""
    return child_world * parent_world.GetInverse()


def rotate(quat: tuple, v) -> tuple[float, float, float]:
    w, x, y, z = quat
    r = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    out = r.TransformDir(Gf.Vec3d(*v))
    return (float(out[0]), float(out[1]), float(out[2]))


AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def mesh_arrays(mesh: UsdGeom.Mesh, time=None):
    """Triangulated (vertices, faces) of a UsdGeom.Mesh in its local frame."""
    time = Usd.TimeCode.Default() if time is None else time
    pts = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    idx = mesh.GetFaceVertexIndicesAttr().Get(time)
    if pts is None or counts is None or idx is None or len(pts) == 0:
        return None
    v = np.asarray(pts, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    faces = []
    start = 0
    for n in counts:  # fan triangulation
        if n >= 3:
            a = idx[start]
            for k in range(1, n - 1):
                faces.append((a, idx[start + k], idx[start + k + 1]))
        start += n
    if not faces:
        return None
    f = np.asarray(faces, dtype=np.int64)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        f = f[:, ::-1]
    return v, f


def bake_scale(vertices: np.ndarray, faces: np.ndarray, scale: np.ndarray, length_scale: float):
    """Apply per-axis scale (and unit scale) to vertices; flip winding if mirrored."""
    v = vertices * (scale * length_scale)
    if np.prod(np.sign(scale)) < 0:
        faces = faces[:, ::-1]
    return v, faces
