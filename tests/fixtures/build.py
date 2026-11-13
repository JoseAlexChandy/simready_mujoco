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


def build_runtime_variant_asset(asset_dir: Path) -> Path:
    """SimReady-style multi-runtime prop: neutral base + PhysX / MuJoCo variant payloads.

    Layout (as in SimReady Foundation):
        <asset_dir>/simready_usd/prop.usda           root, default prim /prop
        <asset_dir>/runnables/physics/physx.usda     PhysX layer  (PhysX=Enabled, the default)
        <asset_dir>/runnables/physics/mujoco.usda    MuJoCo layer (MuJoCo=Enabled)
    Also has a joint with an unregistered multi-apply mimic schema in the base.
    """
    from pxr import Sdf

    phys = asset_dir / "runnables" / "physics"
    phys.mkdir(parents=True, exist_ok=True)
    root_dir = asset_dir / "simready_usd"
    root_dir.mkdir(parents=True, exist_ok=True)

    def runtime_layer(name: str, schema: str, attrs: dict, extra_prims: dict) -> None:
        layer = Sdf.Layer.CreateNew(str(phys / f"{name}.usda"))
        s = Usd.Stage.Open(layer)
        over = s.OverridePrim("/prop/body/body_collision")
        op = Sdf.TokenListOp()
        op.prependedItems = [schema]
        over.SetMetadata("apiSchemas", op)
        for k, v in attrs.items():
            over.CreateAttribute(k, Sdf.ValueTypeNames.Double).Set(v)
        for path, type_name in extra_prims.items():
            p = s.DefinePrim(path, type_name)
            p.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray).Set([100.0])
            p.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token).Set("affine")
            p.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray).Set(
                [0.0, -100.0, -10.0]
            )
            p.CreateAttribute("mjc:forceRange:min", Sdf.ValueTypeNames.Double).Set(-5.0)
            p.CreateAttribute("mjc:forceRange:max", Sdf.ValueTypeNames.Double).Set(5.0)
            p.CreateRelationship("mjc:target").SetTargets([Sdf.Path("/prop/lid_hinge")])
        s.SetDefaultPrim(s.GetPrimAtPath("/prop"))
        layer.Save()

    runtime_layer("physx", "PhysxCollisionAPI", {"physxCollision:contactOffset": 0.01}, {})
    runtime_layer(
        "mujoco",
        "MjcCollisionAPI",
        {"mjc:margin": 0.001},
        {"/prop/actuators/hinge_motor": "MjcActuator"},
    )

    root = root_dir / "prop.usda"
    stage = Usd.Stage.CreateNew(str(root))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prop = UsdGeom.Xform.Define(stage, "/prop").GetPrim()
    stage.SetDefaultPrim(prop)
    body = UsdGeom.Xform.Define(stage, "/prop/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.5)
    _box_mesh(stage, "/prop/body/body_visual", (0.1, 0.1, 0.1), (0, 0, 0))
    c = _box_mesh(stage, "/prop/body/body_collision", (0.1, 0.1, 0.1), (0, 0, 0))
    _collider(c.GetPrim(), "sdf")
    lid_x = UsdGeom.Xform.Define(stage, "/prop/lid")
    UsdGeom.XformCommonAPI(lid_x).SetTranslate(Gf.Vec3d(0, 0, 0.06))
    lid = lid_x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lid)
    lc = _box_mesh(stage, "/prop/lid/lid_collision", (0.1, 0.1, 0.02), (0, 0, 0))
    _collider(lc.GetPrim())
    j = UsdPhysics.RevoluteJoint.Define(stage, "/prop/lid_hinge")
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(-0.05, 0, 0.05))
    j.CreateLocalPos1Attr(Gf.Vec3f(-0.05, 0, -0.01))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(110.0)
    j.CreateBody0Rel().SetTargets([body.GetPath()])
    j.CreateBody1Rel().SetTargets([lid.GetPath()])
    op = Sdf.TokenListOp()
    op.prependedItems = ["PhysxMimicJointAPI:rotZ"]
    j.GetPrim().SetMetadata("apiSchemas", op)

    vsets = prop.GetVariantSets()
    for name, default in (("PhysX", "Enabled"), ("MuJoCo", "Disabled")):
        vs = vsets.AddVariantSet(name)
        for option in ("Disabled", "Enabled"):
            vs.AddVariant(option)
        vs.SetVariantSelection("Enabled")
        with vs.GetVariantEditContext():
            prop.GetPayloads().AddPayload(f"../runnables/physics/{name.lower()}.usda")
        vs.SetVariantSelection(default)
    stage.GetRootLayer().Save()
    return root


def build_mimic_gripper(path: Path) -> Path:
    """Palm + driven finger + two followers: one PhysX mimic (gearing -1 => same direction under
    PhysX's q + g*q_ref + o = 0) and one Newton mimic (coef1 = -1 => mirrored).
    The driven finger is authored rotated 10 deg about its axis, to test MJCF `ref`."""
    from pxr import Sdf

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/gripper").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    def body(name, pos, size, rot_deg=0.0):
        x = UsdGeom.Xform.Define(stage, f"/gripper/{name}")
        api = UsdGeom.XformCommonAPI(x)
        api.SetTranslate(Gf.Vec3d(*pos))
        if rot_deg:
            api.SetRotate(Gf.Vec3f(0, rot_deg, 0))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(0.05)
        c = _box_mesh(stage, f"/gripper/{name}/{name}_col", size, (0, 0, 0))
        _collider(c.GetPrim())
        return x.GetPrim()

    palm = body("palm", (0, 0, 0), (0.08, 0.04, 0.02))
    UsdPhysics.MassAPI(palm).GetMassAttr().Set(0.5)
    lead = body("finger_lead", (0.03, 0, 0.03), (0.01, 0.01, 0.04), rot_deg=10.0)
    phx = body("finger_physx", (-0.03, 0, 0.03), (0.01, 0.01, 0.04))
    nwt = body("finger_newton", (0, 0.035, 0.03), (0.01, 0.01, 0.04))

    fj = UsdPhysics.FixedJoint.Define(stage, "/gripper/joints/palm_fixed")
    fj.CreateBody1Rel().SetTargets([palm.GetPath()])

    def hinge(name, child, lo, hi, x, rot_deg=0.0):
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/gripper/joints/{name}")
        j.CreateBody0Rel().SetTargets([palm.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateAxisAttr("Y")  # fingers tilt about their base, like a real gripper
        j.CreateLocalPos0Attr(Gf.Vec3f(*x))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, -0.02))
        # joint frame on the child cancels its authored rotation => joint value at rest = rot_deg
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(lo)
        j.CreateUpperLimitAttr(hi)
        return j

    lead_j = hinge("lead_joint", lead, 0.0, 45.0, (0.03, 0, 0.01))
    d = UsdPhysics.DriveAPI.Apply(lead_j.GetPrim(), "angular")
    d.CreateStiffnessAttr(0.02)  # N·m/deg; stable at a 2 ms step for a 50 g finger
    d.CreateDampingAttr(0.0005)
    d.CreateMaxForceAttr(10.0)

    pj = hinge("physx_follower", phx, 0.0, 45.0, (-0.03, 0, 0.01))
    op = Sdf.TokenListOp()
    op.prependedItems = ["PhysxMimicJointAPI:rotZ"]
    pj.GetPrim().SetMetadata("apiSchemas", op)
    pj.GetPrim().CreateAttribute("physxMimicJoint:rotZ:gearing", Sdf.ValueTypeNames.Float).Set(-1.0)
    pj.GetPrim().CreateAttribute("physxMimicJoint:rotZ:offset", Sdf.ValueTypeNames.Float).Set(0.0)
    pj.GetPrim().CreateRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets(
        [lead_j.GetPath()]
    )

    nj = hinge("newton_follower", nwt, -45.0, 0.0, (0, 0.035, 0.01))
    nj.GetPrim().CreateRelationship("newton:mimicJoint").SetTargets([lead_j.GetPath()])
    nj.GetPrim().CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
    nj.GetPrim().CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(-1.0)
    stage.GetRootLayer().Save()
    return path


def build_hollow_cabinet(path: Path) -> Path:
    """Open-front cabinet carcass (ONE concave collision mesh, like SimReady fridges), a door
    hinged to it, and a compartment hinged to the door that sits inside the cavity.
    With convex-hull collision the carcass becomes a solid block that overlaps the compartment
    and blows the door open; convex decomposition keeps the cavity hollow."""
    import numpy as np
    import trimesh as tm

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/hollow").GetPrim()
    stage.SetDefaultPrim(root)

    W, D, H, t = 0.6, 0.5, 0.8, 0.02  # outer size, wall thickness; open at -y (front)
    panels = [
        tm.creation.box((W, D, t), tm.transformations.translation_matrix((0, 0, -H / 2 + t / 2))),
        tm.creation.box((W, D, t), tm.transformations.translation_matrix((0, 0, H / 2 - t / 2))),
        tm.creation.box((t, D, H), tm.transformations.translation_matrix((-W / 2 + t / 2, 0, 0))),
        tm.creation.box((t, D, H), tm.transformations.translation_matrix((W / 2 - t / 2, 0, 0))),
        tm.creation.box((W, t, H), tm.transformations.translation_matrix((0, D / 2 - t / 2, 0))),
    ]
    carcass_mesh = tm.util.concatenate(panels)

    def rigid(name, pos, mass):
        x = UsdGeom.Xform.Define(stage, f"/hollow/{name}")
        UsdGeom.XformCommonAPI(x).SetTranslate(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        return x.GetPrim()

    carcass = rigid("carcass", (0, 0, H / 2), 20.0)
    m = UsdGeom.Mesh.Define(stage, "/hollow/carcass/carcass_col")
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in carcass_mesh.vertices])
    m.CreateFaceVertexCountsAttr([3] * len(carcass_mesh.faces))
    m.CreateFaceVertexIndicesAttr([int(i) for f in carcass_mesh.faces for i in f])
    _collider(m.GetPrim(), "convexHull")

    door = rigid("door", (0, -D / 2 - 0.011, H / 2), 3.0)
    dc = _box_mesh(stage, "/hollow/door/door_col", (W, 0.02, H), (0, 0, 0))
    _collider(dc.GetPrim())
    # compartment hanging off the door, inside the cavity (clear of the walls)
    comp = rigid("compartment", (0, -D / 2 + 0.08, H / 2), 0.5)
    cc = _box_mesh(stage, "/hollow/compartment/comp_col", (0.4, 0.1, 0.1), (0, 0, 0))
    _collider(cc.GetPrim())

    fj = UsdPhysics.FixedJoint.Define(stage, "/hollow/joints/carcass_fixed")
    fj.CreateBody1Rel().SetTargets([carcass.GetPath()])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/hollow/joints/door_hinge")
    hinge.CreateBody0Rel().SetTargets([carcass.GetPath()])
    hinge.CreateBody1Rel().SetTargets([door.GetPath()])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(-W / 2, -D / 2 - 0.011, 0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(-W / 2, 0, 0))
    hinge.CreateLowerLimitAttr(-150.0)
    hinge.CreateUpperLimitAttr(0.0)
    cj = UsdPhysics.FixedJoint.Define(stage, "/hollow/joints/compartment_on_door")
    cj.CreateBody0Rel().SetTargets([door.GetPath()])
    cj.CreateBody1Rel().SetTargets([comp.GetPath()])
    cj.CreateLocalPos0Attr(Gf.Vec3f(0, 0.091, 0))
    stage.GetRootLayer().Save()
    assert np  # numpy used by trimesh transforms
    return path


def build_textured_asset(asset_dir: Path) -> Path:
    """A textured quad (UsdPreviewSurface <- UsdUVTexture <- primvar 'st', faceVarying, image in
    ../Textures like SimReady) plus a box mesh whose faces use two GeomSubset materials."""
    import numpy as np
    from PIL import Image
    from pxr import Sdf, UsdShade, Vt

    (asset_dir / "Textures").mkdir(parents=True, exist_ok=True)
    (asset_dir / "simready_usd").mkdir(parents=True, exist_ok=True)
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32] = (255, 0, 0)  # top half red
    img[32:] = (0, 0, 255)  # bottom half blue
    Image.fromarray(img).save(asset_dir / "Textures" / "t_quad_alb.jpg")  # jpg: must be converted

    path = asset_dir / "simready_usd" / "textured.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    quad = UsdGeom.Mesh.Define(stage, "/asset/quad")
    quad.CreatePointsAttr([Gf.Vec3f(-0.1, -0.1, 0.3), Gf.Vec3f(0.1, -0.1, 0.3),
                           Gf.Vec3f(0.1, 0.1, 0.3), Gf.Vec3f(-0.1, 0.1, 0.3)])  # fmt: skip
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))

    mat = UsdShade.Material.Define(stage, "/asset/Looks/quad_mat")
    pbs = UsdShade.Shader.Define(stage, "/asset/Looks/quad_mat/Principled_BSDF")
    pbs.CreateIdAttr("UsdPreviewSurface")
    pbs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    tex = UsdShade.Shader.Define(stage, "/asset/Looks/quad_mat/Albedo")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("../Textures/t_quad_alb.jpg")
    reader = UsdShade.Shader.Define(stage, "/asset/Looks/quad_mat/uvmap")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    pbs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    mat.CreateSurfaceOutput().ConnectToSource(pbs.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)

    box = _box_mesh(stage, "/asset/box", (0.2, 0.2, 0.2), (0, 0, 0.1))
    _collider(box.GetPrim())  # visible collider: also rendered
    colors = {"green": (0.1, 0.8, 0.1), "yellow": (0.9, 0.8, 0.1)}
    for i, (name, rgb) in enumerate(colors.items()):
        m = UsdShade.Material.Define(stage, f"/asset/Looks/{name}")
        s = UsdShade.Shader.Define(stage, f"/asset/Looks/{name}/Principled_BSDF")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        subset = UsdGeom.Subset.CreateGeomSubset(
            box, name, UsdGeom.Tokens.face, [0, 1, 2] if i == 0 else [3, 4, 5], "materialBind"
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(m)
    stage.GetRootLayer().Save()
    return path


def build_static_appliance(path: Path) -> Path:
    """A microwave authored like many SimReady props: ONE rigid body whose door and button are
    plain meshes, no joints. Centimetre units and a rotated, offset root so that articulation
    specs are checked in the asset frame, not the world frame.

    Asset frame (cm): carcass 50 x 35 x 30 centred on (0, 0, 15), front face at y = -17.5.
    Door: 35 x 2 x 28 panel in front of it, left edge at x = -25. Button: right of the door."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root_x = UsdGeom.Xform.Define(stage, "/Microwave")
    api = UsdGeom.XformCommonAPI(root_x)
    api.SetTranslate(Gf.Vec3d(100, 200, 0))
    api.SetRotate(Gf.Vec3f(0, 0, 90))
    root = root_x.GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(12.0)

    UsdGeom.Xform.Define(stage, "/Microwave/geo")
    carcass = _box_mesh(stage, "/Microwave/geo/carcass", (50, 35, 30), (0, 0, 15))
    _collider(carcass.GetPrim())
    UsdGeom.Xform.Define(stage, "/Microwave/geo/door")
    door = _box_mesh(stage, "/Microwave/geo/door/door_panel", (35, 2, 28), (-7.5, -18.5, 15))
    _collider(door.GetPrim())
    button = _box_mesh(stage, "/Microwave/geo/button", (4, 1, 2), (18, -18.0, 8))
    _collider(button.GetPrim())
    stage.GetRootLayer().Save()
    return path
