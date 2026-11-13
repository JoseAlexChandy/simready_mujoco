"""Public Python API. Import these from other projects instead of calling the CLI."""

from __future__ import annotations

from pathlib import Path

from simready_mujoco._quiet import redirect_output
from simready_mujoco.sim.load import ModelSummary, summarize_mjcf
from simready_mujoco.sim.profiles import apply_profile, get_profile
from simready_mujoco.usd.inspect import InspectionReport, inspect_usd
from simready_mujoco.usd.stage import open_stage

__all__ = ["InspectionReport", "ModelSummary", "convert", "inspect_usd", "summarize_mjcf"]


def convert(
    usd_file: str | Path,
    out_dir: str | Path,
    *,
    sim_profile: str | None = "manipulation",
    generate_collision: bool = False,
    backend: str = "native",
    preprocess_resolution: int = 20,
    resolution: int = 2000,
    runtime: str = "mujoco",
    variants: dict[str, str] | None = None,
    quiet: bool = True,
    actuators: str = "auto",
    decompose: str = "auto",
    articulation: list[dict] | None = None,
    authored_mass: str = "auto",
) -> Path:
    """Convert one USD asset to MJCF and return the MJCF file path.

    Args:
        usd_file: input .usd/.usda/.usdc/.usdz
        out_dir: folder to write the MJCF and meshes into
        sim_profile: name of a preset in sim_profiles.yaml, or None to keep backend defaults
        generate_collision: build convex collision meshes with CoACD
        backend: "native" (this project, default) or "upstream" (LightwheelAI/usd2mjcf baseline)
        runtime: physics variant to compose: "mujoco" (default), "physx", "newton", "authored"
        variants: extra {variant set: option} selections, e.g. {"Physics": "Newton_parallel_grip"}
        quiet: send third-party log output to <out_dir>/convert.log instead of the terminal
        actuators: native backend: "auto" (MuJoCo layer, unless an MjcActuator can't hold a
            position but a position drive exists), "mjc" (MuJoCo layer wins), "drive"
            (UsdPhysics drives win)
        decompose: native backend: "auto" splits concave collision meshes into convex parts
            (CoACD) only when their hull overlaps another part at the authored pose (hollow
            carcasses with things inside); "concave", "all" or "off"
        authored_mass: native backend: "auto" replaces authored mass properties that were
            evidently computed from convex hulls (centre of mass on the hull centroid, far
            from the real shape's); "keep" uses the file's values as they are
        articulation: native backend: joints to add for moving parts the asset models as
            plain meshes (see usd.articulate.PartJoint and docs/articulation.md)
    """
    usd_file, out_dir = Path(usd_file), Path(out_dir)
    if not usd_file.exists():
        raise FileNotFoundError(usd_file)

    if backend == "native":
        from simready_mujoco.native import parse_stage, write_mjcf  # noqa: PLC0415

        stage, _ = open_stage(usd_file, runtime=runtime, variants=variants)
        added = []
        if articulation:
            from simready_mujoco.usd.articulate import apply_articulation  # noqa: PLC0415

            added = apply_articulation(stage, articulation)
        if decompose == "auto":
            model = _auto_decompose(stage, usd_file.stem, actuators, out_dir, authored_mass)
        else:
            model = parse_stage(
                stage,
                name=usd_file.stem,
                actuators=actuators,
                decompose=decompose,
                authored_mass=authored_mass,
            )
        if added:
            model.notes["added_joints"] = added
        if generate_collision:
            model.notes["collision_flag"] = "ignored: native backend uses authored colliders"
        mjcf = write_mjcf(model, out_dir)
    elif backend == "upstream":
        from simready_mujoco.backends import upstream  # noqa: PLC0415

        stage, _ = open_stage(usd_file, runtime=runtime, variants=variants)
        with redirect_output(out_dir / "convert.log" if quiet else None):
            mjcf = upstream.convert(
                usd_file,
                out_dir,
                generate_collision,
                preprocess_resolution,
                resolution,
                stage=stage,
            )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Available: native, upstream")

    if sim_profile:
        authored = set(model.option) | set(model.flags) if backend == "native" else set()
        kept = apply_profile(mjcf, get_profile(sim_profile), keep=authored)
        if backend == "native" and kept:
            model.notes["asset_options_kept"] = kept
    if backend == "native":
        from simready_mujoco.sim.stability import check_drive_stability  # noqa: PLC0415

        stiff = check_drive_stability(mjcf)
        for s in stiff:
            model.warnings.append(
                f"actuator {s.actuator}: drive too stiff for timestep (omega*dt = {s.omega_dt:.2f}); "
                f"add armature >= {s.armature_needed:.3g} on {s.joint}, or lower the timestep"
            )
        model.notes["stiff_drives"] = [s.to_dict() for s in stiff]
        model.warnings.extend(_health_warnings(mjcf))
        _write_report(model, out_dir)
    return mjcf


def _health_warnings(mjcf: Path) -> list[str]:
    """Bad mass properties, and whether the asset alone diverges within a second."""
    import mujoco  # noqa: PLC0415

    from simready_mujoco.sim.health import first_divergence, mass_problems  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    out = mass_problems(m)
    div = first_divergence(m, 1.0)
    if div is not None:
        out.append(f"UNSTABLE: {div}")
    return out


def _write_report(model, out_dir: Path) -> None:
    """conversion_report.json: what was converted and every warning."""
    import json  # noqa: PLC0415

    report = {"model": model.name, **model.notes, "warnings": model.warnings}
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2, default=str))


def _auto_decompose(stage, name: str, actuators: str, out_dir: Path, authored_mass: str = "auto"):
    """Convert, then split concave colliders that overlap other parts at the authored pose.
    Meshes still overlapping after a round are redone at a finer level (decompose.LEVELS).

    Each check uses a fresh temporary folder: MuJoCo caches mesh assets by file path within a
    process, so re-writing same-named mesh files would make later checks see stale meshes.
    """
    import tempfile  # noqa: PLC0415

    from simready_mujoco.native import parse_stage, write_mjcf  # noqa: PLC0415
    from simready_mujoco.native.decompose import LEVELS  # noqa: PLC0415

    def check(model) -> set[str]:
        with tempfile.TemporaryDirectory(prefix="srmj_check_") as tmp:
            return _overlapping_meshes(write_mjcf(model, tmp), model)

    model = parse_stage(stage, name=name, actuators=actuators, authored_mass=authored_mass)
    overlapping = check(model)
    if not overlapping:
        return model
    levels: dict[str, int] = {}
    rounds = []
    for level in range(len(LEVELS)):
        for mesh in overlapping:
            levels[mesh] = level if mesh not in levels else max(levels[mesh], level)
        model = parse_stage(
            stage,
            name=name,
            actuators=actuators,
            decompose="concave",
            decompose_only=dict(levels),
            authored_mass=authored_mass,
        )
        still = check(model)
        rounds.append({"level": level, **LEVELS[level], "still_overlapping": sorted(still)})
        # only meshes that were decomposed can be refined further
        decomposable = {m for m, parts in _parts_by_source(model).items() if parts}
        overlapping = still & decomposable
        if not overlapping:
            break
    model.notes["overlapping_colliders"] = sorted(levels)
    model.notes["decomposition_rounds"] = rounds
    remaining = set(rounds[-1]["still_overlapping"]) & {
        m for m, parts in _parts_by_source(model).items() if parts
    }
    if remaining:
        model.warnings.append(
            "colliders still overlap after the finest decomposition: "
            + ", ".join(sorted(remaining))
        )
    _clear_meshes(out_dir)
    return model


def _parts_by_source(model) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for md in model.meshes.values():
        if md.source:
            out.setdefault(md.source, []).append(md.name)
    return out


def _overlapping_meshes(mjcf: Path, model) -> set[str]:
    """Source mesh names of collision geoms that penetrate another body's collider at the
    authored pose (more than decompose.OVERLAP). Convex parts map back to their source."""
    import mujoco  # noqa: PLC0415

    from simready_mujoco.native.decompose import OVERLAP  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def source(mesh: str) -> str:
        md = model.meshes.get(mesh)
        return md.source if md is not None and md.source else mesh

    geom_mesh = {
        g.name: source(g.mesh) for b in model.bodies for g in b.geoms if g.collision and g.mesh
    }
    hits: set[str] = set()
    for c in d.contact[: d.ncon]:
        if c.dist < -OVERLAP:
            for gid in (c.geom1, c.geom2):
                mesh = geom_mesh.get(m.geom(gid).name)
                if mesh:
                    hits.add(mesh)
    return hits


def _clear_meshes(out_dir: Path) -> None:
    mesh_dir = out_dir / "meshes"
    if mesh_dir.is_dir():
        for f in mesh_dir.iterdir():
            f.unlink()
