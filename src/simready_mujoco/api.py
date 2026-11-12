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
    backend: str = "upstream",
    preprocess_resolution: int = 20,
    resolution: int = 2000,
    runtime: str = "mujoco",
    variants: dict[str, str] | None = None,
    quiet: bool = True,
) -> Path:
    """Convert one USD asset to MJCF and return the MJCF file path.

    Args:
        usd_file: input .usd/.usda/.usdc/.usdz
        out_dir: folder to write the MJCF and meshes into
        sim_profile: name of a preset in sim_profiles.yaml, or None to keep backend defaults
        generate_collision: build convex collision meshes with CoACD
        backend: "upstream" (LightwheelAI/usd2mjcf). Our own backend will be added as "native".
        runtime: physics variant to compose: "mujoco" (default), "physx", "newton", "authored"
        variants: extra {variant set: option} selections, e.g. {"Physics": "Newton_parallel_grip"}
        quiet: send third-party log output to <out_dir>/convert.log instead of the terminal
    """
    usd_file, out_dir = Path(usd_file), Path(out_dir)
    if not usd_file.exists():
        raise FileNotFoundError(usd_file)

    if backend == "upstream":
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
        raise ValueError(f"Unknown backend '{backend}'. Available: upstream")

    if sim_profile:
        apply_profile(mjcf, get_profile(sim_profile))
    return mjcf
