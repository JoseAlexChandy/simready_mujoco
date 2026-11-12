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
    """
    usd_file, out_dir = Path(usd_file), Path(out_dir)
    if not usd_file.exists():
        raise FileNotFoundError(usd_file)

    if backend == "native":
        from simready_mujoco.native import parse_stage, write_mjcf  # noqa: PLC0415

        stage, _ = open_stage(usd_file, runtime=runtime, variants=variants)
        if decompose == "auto":
            # Pass 1 without decomposition; find concave hulls that overlap other parts.
            model = parse_stage(stage, name=usd_file.stem, actuators=actuators)
            write_mjcf(model, out_dir)
            overlapping = _overlapping_meshes(out_dir / f"{usd_file.stem}.xml", model)
            if overlapping:
                model = parse_stage(
                    stage,
                    name=usd_file.stem,
                    actuators=actuators,
                    decompose="concave",
                    decompose_only=overlapping,
                )
                model.notes["overlapping_colliders"] = sorted(overlapping)
                _clear_meshes(out_dir)
        else:
            model = parse_stage(stage, name=usd_file.stem, actuators=actuators, decompose=decompose)
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
        _write_report(model, out_dir)
    return mjcf


def _write_report(model, out_dir: Path) -> None:
    """conversion_report.json: what was converted and every warning."""
    import json  # noqa: PLC0415

    report = {"model": model.name, **model.notes, "warnings": model.warnings}
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2, default=str))


def _overlapping_meshes(mjcf: Path, model) -> set[str]:
    """Mesh names of collision geoms that penetrate another body's collider at the authored
    pose (more than decompose.OVERLAP). Those are candidates for convex decomposition."""
    import mujoco  # noqa: PLC0415

    from simready_mujoco.native.decompose import OVERLAP  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geom_mesh = {g.name: g.mesh for b in model.bodies for g in b.geoms if g.collision and g.mesh}
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
