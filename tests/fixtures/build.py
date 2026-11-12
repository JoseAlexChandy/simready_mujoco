"""Build small synthetic USD assets for tests.

Regenerate the committed fixtures with:
    python tests/fixtures/build.py
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).parent


def _box_mesh(stage: Usd.Stage, path: str, size: tuple[float, float, float], pos) -> UsdGeom.Mesh:
    """An axis-aligned box as a UsdGeom.Mesh (upstream exporter needs real meshes)."""
    sx, sy, sz = (s / 2 for s in size)
    pts = [
        (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
        (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
    ]  # fmt: skip
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]  # fmt: skip
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateExtentAttr([Gf.Vec3f(-sx, -sy, -sz), Gf.Vec3f(sx, sy, sz)])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*pos))
    return mesh


def _collider(prim: Usd.Prim, approximation: str = "convexHull") -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def build_cabinet(path: Path | None = None) -> Usd.Stage:
    """Cabinet: fixed base (2 kg), hinged door with a position drive, sliding drawer.

    Exercises: MassAPI, physics material, revolute + limits + drive, prismatic + limits,
    mesh colliders, articulation root, fixed joint to world.
    """
    stage = Usd.Stage.CreateNew(str(path)) if path else Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/cabinet")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(world.GetPrim())

    # Physics material
    mat = UsdShade.Material.Define(stage, "/cabinet/Looks/wood_physics")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.8)
    pm.CreateDynamicFrictionAttr(0.6)
    pm.CreateRestitutionAttr(0.2)

    def body(name: str, pos, mass: float | None = None) -> Usd.Prim:
        x = UsdGeom.Xform.Define(stage, f"/cabinet/{name}")
        UsdGeom.XformCommonAPI(x).SetTranslate(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        if mass is not None:
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        return x.GetPrim()

    base = body("base", (0, 0, 0.4), mass=2.0)
    door = body("door", (0.21, 0, 0.4))
    drawer = body("drawer", (0, 0, 0.1))

    # Isaac/SimReady style: separate visual and collision meshes under each body.
    for prim, size in (
        (base, (0.4, 0.4, 0.8)),
        (door, (0.02, 0.4, 0.6)),
        (drawer, (0.36, 0.36, 0.15)),
    ):
        name = prim.GetName()
        _box_mesh(stage, f"{prim.GetPath()}/{name}_visual", size, (0, 0, 0))
        c = _box_mesh(stage, f"{prim.GetPath()}/{name}_collision", size, (0, 0, 0))
        UsdGeom.Imageable(c).CreatePurposeAttr(UsdGeom.Tokens.guide)
        _collider(c.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    # Base fixed to world
    fj = UsdPhysics.FixedJoint.Define(stage, "/cabinet/joints/base_fixed")
    fj.CreateBody1Rel().SetTargets([base.GetPath()])

    # Door hinge on the base's +x edge, 0..90 degrees, with a position drive
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/cabinet/joints/door_hinge")
    hinge.CreateBody0Rel().SetTargets([base.GetPath()])
    hinge.CreateBody1Rel().SetTargets([door.GetPath()])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.21, 0.2, 0.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.2, 0.0))
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(90.0)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(5.0)  # per degree, as USD authors it
    drive.CreateDampingAttr(0.5)
    drive.CreateMaxForceAttr(20.0)
    drive.CreateTargetPositionAttr(45.0)

    # Drawer slides along +x, 0..0.3 m
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/cabinet/joints/drawer_slide")
    slide.CreateBody0Rel().SetTargets([base.GetPath()])
    slide.CreateBody1Rel().SetTargets([drawer.GetPath()])
    slide.CreateAxisAttr("X")
    slide.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -0.3))
    slide.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    slide.CreateLowerLimitAttr(0.0)
    slide.CreateUpperLimitAttr(0.3)

    if path:
        stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    out = HERE / "cabinet.usda"
    out.unlink(missing_ok=True)
    build_cabinet(out)
    print(f"wrote {out}")
