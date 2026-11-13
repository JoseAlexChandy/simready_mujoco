"""Compare an asset's authored mass properties with its colliders' hull and real shape.

    python scripts/mass_probe.py kitchenware_mug_d01 [configs/subset.yaml]

Prints, per collider with authored MassAPI and per rigid body, in the body frame (metres):
authored mass / centre of mass, the convex hull's volume and centroid, and the mesh's own
volume and centroid (closed mesh or thin shell). Used to check whether authored values were
computed from convex hulls (see native.parse._correct_hull_derived_mass).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml
from pxr import Usd, UsdGeom, UsdPhysics

from simready_mujoco.fetch import resolve_asset
from simready_mujoco.native.geometry import mesh_arrays
from simready_mujoco.native.parse import _mass_shape, _shape_props
from simready_mujoco.usd.stage import open_stage


def main(asset_id: str, config: str = "configs/subset.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    entry = next(e for e in cfg["assets"] if e["id"] == asset_id)
    usd = resolve_asset(entry).usd
    stage, _ = open_stage(usd, runtime=entry.get("runtime", cfg.get("runtime", "mujoco")))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    xc = UsdGeom.XformCache()
    print(f"{usd}\nmetersPerUnit {mpu}  kilogramsPerUnit {kpu}")

    def body_of(p):
        q = p
        while q and not q.IsPseudoRoot():
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                return q
            q = q.GetParent()
        return None

    for p in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        mass_attr = p.GetAttribute("physics:mass")
        has_mass = bool(mass_attr and mass_attr.HasAuthoredValue())
        if not (p.HasAPI(UsdPhysics.CollisionAPI) or has_mass):
            continue
        body = body_of(p)
        print(f"\n{p.GetPath()}  [{p.GetTypeName()}]  body: {body.GetPath() if body else None}")
        print("  approximation:", p.GetAttribute("physics:approximation").Get())
        for name in (
            "physics:mass",
            "physics:density",
            "physics:centerOfMass",
            "physics:diagonalInertia",
            "physics:principalAxes",
        ):
            a = p.GetAttribute(name)
            if a and a.HasAuthoredValue():
                print(f"  {name} = {a.Get()}")
        if not p.IsA(UsdGeom.Mesh) or body is None:
            continue
        arrays = mesh_arrays(UsdGeom.Mesh(p))
        if arrays is None:
            continue
        v, f = arrays[0], arrays[1]
        # mesh -> body frame, metres
        rel = xc.GetLocalToWorldTransform(p) * xc.GetLocalToWorldTransform(body).GetInverse()
        M = np.array(rel, dtype=float)
        vb = (np.c_[np.asarray(v, float), np.ones(len(v))] @ M)[:, :3] * mpu
        # authored COM is in the collider's own frame
        com_attr = p.GetAttribute("physics:centerOfMass")
        if com_attr and com_attr.HasAuthoredValue():
            c = np.array(com_attr.Get(), float)
            cb = (np.r_[c, 1.0] @ M)[:3] * mpu
            print("  authored COM in body frame (m):", cb.round(4))
        tm = trimesh.Trimesh(vb, f, process=False)
        hull = tm.convex_hull
        print(
            f"  hull: volume {hull.volume:.6g} m3 (x1000 kg/m3 = {hull.volume * 1000:.3f} kg), "
            f"centroid {np.asarray(hull.center_mass).round(4)}, verts {len(hull.vertices)}"
        )
        shape = _mass_shape(vb, f)
        if shape is not None:
            vol, com, _ = _shape_props(shape)
            print(
                f"  shape ({shape.metadata.get('mass_source')}): volume {vol:.6g} m3 "
                f"(x1000 = {vol * 1000:.3f} kg), centroid {np.asarray(com).round(4)}"
            )
        print(f"  bounds min {vb.min(0).round(4)} max {vb.max(0).round(4)}")

    xml = Path(cfg.get("output_dir", "outputs")) / asset_id
    files = [x for x in xml.glob("*.xml")]
    if files:
        m = mujoco.MjModel.from_xml_path(str(files[0]))
        print("\nconverted:")
        for b in range(1, m.nbody):
            if m.body_mass[b] > 0:
                print(
                    f"  {m.body(b).name}: mass {m.body_mass[b]:.4f}  ipos {m.body_ipos[b].round(4)}"
                )


if __name__ == "__main__":
    main(*sys.argv[1:])
