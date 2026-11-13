"""Convex decomposition for concave collision meshes.

MuJoCo collides every mesh as its convex hull. SimReady's MuJoCo variant marks colliders as
`convexHull`, which is right for most parts but turns hollow ones (fridge and cabinet
carcasses, open boxes, cups) into solid blocks: the parts inside them start out overlapping
and get pushed out. We detect clearly concave meshes and split them with CoACD.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# auto:    concave colliders whose hull overlaps another part at the authored pose (the case
#          that breaks simulation, e.g. fridge carcass vs the compartments inside it)
# concave: every concave collider (also lets objects go *into* open boxes, cups, ...)
# all:     every mesh collider;  off: one convex hull per mesh (MuJoCo default)
MODES = ("auto", "concave", "all", "off")
OVERLAP = 0.005  # metres of penetration at the authored pose that counts as an overlap

# A mesh is "concave" when a noticeable share of its hull's surface is far from the mesh.
FAR_FRACTION = 0.05  # of the bounding-box diagonal
CONCAVE_SHARE = 0.10  # share of hull samples that are far

COACD_PARAMS = {
    "threshold": 0.08,
    "max_convex_hull": 32,
    "preprocess_resolution": 40,
    "resolution": 2000,
    "mcts_iterations": 100,
}


def cache_dir() -> Path:
    base = os.environ.get("SRMJ_CACHE", Path.home() / ".cache" / "simready-mujoco")
    d = Path(base) / "coacd"
    d.mkdir(parents=True, exist_ok=True)
    return d


def concavity(vertices: np.ndarray, faces: np.ndarray, samples: int = 1500) -> float:
    """Share of convex-hull surface samples lying far from the mesh surface (0 = convex)."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    try:
        hull = mesh.convex_hull
    except Exception:  # noqa: BLE001  (flat / degenerate)
        return 0.0
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diag <= 0 or mesh.area <= 0:
        return 0.0
    rng = np.random.default_rng(0)
    surface, _ = trimesh.sample.sample_surface(mesh, max(samples * 6, 6000), seed=rng)
    hull_pts, _ = trimesh.sample.sample_surface(hull, samples, seed=rng)
    dist, _ = cKDTree(surface).query(hull_pts)
    return float(np.mean(dist > FAR_FRACTION * diag))


def is_concave(vertices: np.ndarray, faces: np.ndarray) -> bool:
    return concavity(vertices, faces) > CONCAVE_SHARE


# Refinement levels for "auto": a mesh still overlapping after a round is redone at the next.
LEVELS = (
    {"threshold": 0.08, "max_convex_hull": 32},
    {"threshold": 0.04, "max_convex_hull": 64},
    {"threshold": 0.02, "max_convex_hull": 128},
)


def decompose(
    vertices: np.ndarray, faces: np.ndarray, level: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convex parts of a mesh (CoACD) at a refinement level, cached on disk by mesh content
    and parameters."""
    params = {**COACD_PARAMS, **LEVELS[min(level, len(LEVELS) - 1)]}
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    key = hashlib.sha1(v.tobytes() + f.tobytes() + repr(sorted(params.items())).encode())
    cached = cache_dir() / f"{key.hexdigest()}.npz"
    if cached.exists():
        data = np.load(cached)
        n = int(data["n"])
        return [(data[f"v{i}"], data[f"f{i}"]) for i in range(n)]

    import coacd  # noqa: PLC0415  (optional dependency)

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(v, f), **params)
    out = [(np.asarray(pv, dtype=np.float64), np.asarray(pf, dtype=np.int64)) for pv, pf in parts]
    out = [(pv, pf) for pv, pf in out if len(pv) >= 4 and len(pf) >= 4]
    np.savez_compressed(
        cached,
        n=len(out),
        **{f"v{i}": p[0] for i, p in enumerate(out)},
        **{f"f{i}": p[1] for i, p in enumerate(out)},
    )
    return out
