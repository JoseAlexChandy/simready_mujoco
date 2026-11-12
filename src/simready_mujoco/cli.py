"""`srmj` command-line tool. Run `srmj --help` to see all commands."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from simready_mujoco import __version__

app = typer.Typer(
    help="Convert NVIDIA SimReady (OpenUSD) assets to MuJoCo and check their physics.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"simready-mujoco {__version__}")


@app.command()
def doctor() -> None:
    """Check that all dependencies are installed and working."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            checks.append((name, True, str(fn() or "")))
        except Exception as e:  # noqa: BLE001
            checks.append((name, False, f"{type(e).__name__}: {e}"))

    check(
        "python >= 3.10", lambda: sys.version.split()[0] if sys.version_info >= (3, 10) else 1 / 0
    )
    check("usd-core (pxr)", lambda: __import__("pxr.Usd").Usd.GetVersion())
    check("mujoco", lambda: __import__("mujoco").__version__)
    check("trimesh", lambda: __import__("trimesh").__version__)

    def upstream() -> str:
        from simready_mujoco.backends.upstream import upstream_path

        return str(upstream_path())

    check("usd2mjcf submodule", upstream)
    check("coacd (optional: collision)", lambda: __import__("coacd") and "ok")
    check("huggingface_hub (optional: download)", lambda: __import__("huggingface_hub").__version__)

    table = Table(title="Environment")
    table.add_column("Check")
    table.add_column("OK")
    table.add_column("Detail", overflow="fold")
    for name, ok, detail in checks:
        table.add_row(name, "[green]✓[/]" if ok else "[red]✗[/]", detail)
    console.print(table)
    required_failed = [n for n, ok, _ in checks if not ok and "optional" not in n]
    if required_failed:
        console.print(f"[red]Missing required: {', '.join(required_failed)}. Run `make setup`.[/]")
        raise typer.Exit(1)


@app.command()
def inspect(
    usd_file: Path = typer.Argument(..., exists=True, help="USD file to inspect"),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show which physics properties an asset authors, and what the baseline converter drops."""
    from simready_mujoco.usd.inspect import inspect_usd

    r = inspect_usd(usd_file)
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, default=str))
        return

    console.rule(f"[bold]{usd_file.name}")
    console.print(
        f"up axis [cyan]{r.up_axis}[/] · metres/unit [cyan]{r.meters_per_unit}[/] · "
        f"kg/unit [cyan]{r.kilograms_per_unit}[/] · physics scene "
        f"[cyan]{'yes' if r.has_physics_scene else 'no'}[/] · "
        f"articulations [cyan]{len(r.articulations)}[/]"
    )

    if r.bodies:
        t = Table(title="Rigid bodies")
        for c in ("Body", "Mass", "Density", "CoM", "Inertia", "Colliders"):
            t.add_column(c)
        for b in r.bodies:
            t.add_row(
                b.path,
                "-" if b.mass is None else f"{b.mass:g}",
                "-" if b.density is None else f"{b.density:g}",
                "✓" if b.com_authored else "-",
                "✓" if b.inertia_authored else "-",
                str(b.collider_count),
            )
        console.print(t)

    if r.joints:
        t = Table(title="Joints")
        for c in ("Joint", "Type", "Axis", "Limits", "Drive (k / d / max)", "Notes"):
            t.add_column(c)
        for j in r.joints:
            drive = (
                f"{j.drive['stiffness']:g} / {j.drive['damping']:g} / {j.drive['max_force'] or '∞'}"
                if j.drive
                else "-"
            )
            notes = ", ".join(
                n for n, on in (("mimic", j.mimic), ("loop", j.exclude_from_articulation)) if on
            )
            lim = (
                "-"
                if not j.limits
                else " … ".join("free" if v is None else f"{v:.4g}" for v in j.limits)
            )
            t.add_row(j.path, j.type, j.axis or "-", lim, drive, notes)
        console.print(t)

    if r.collider_shapes:
        console.print(
            f"Colliders: {r.collider_shapes}  mesh approximations: {r.mesh_approximations}"
        )
    for m in r.materials:
        console.print(
            f"Material {m['path']}: static μ {m['static_friction']:g}, dynamic μ "
            f"{m['dynamic_friction']:g}, restitution {m['restitution']:g}"
        )
    if r.physx_schemas:
        console.print(f"PhysX schemas: {', '.join(r.physx_schemas)}")
    for name, v in r.variants.items():
        console.print(
            f"Variant set [cyan]{name}[/]: selected [cyan]{v['selected'] or '(none)'}[/] "
            f"of {v['options']}"
        )
    if not r.bodies and any("physic" in n.lower() for n in r.variants):
        console.print(
            "[yellow]No rigid bodies found with the current variant selection. "
            "The physics may live in another variant.[/]"
        )

    if r.dropped_by_baseline:
        console.print("\n[yellow]Lost by the baseline (upstream usd2mjcf) converter:[/]")
        for g in r.dropped_by_baseline:
            console.print(f"  • {g}")


@app.command()
def convert(
    usd_file: Path = typer.Argument(..., exists=True, help="USD file to convert"),
    out: Path = typer.Option(Path("outputs"), "--out", "-o", help="Output folder"),
    sim_profile: str = typer.Option(
        "manipulation", help="Preset from sim_profiles.yaml, or 'none'"
    ),
    collision: bool = typer.Option(False, "--collision", help="Generate convex collision meshes"),
    backend: str = typer.Option("upstream", help="Conversion backend"),
) -> None:
    """Convert one USD asset to MJCF and check it loads in MuJoCo."""
    from simready_mujoco.api import convert as _convert
    from simready_mujoco.sim.load import summarize_mjcf

    out_dir = out / usd_file.stem
    with console.status(f"Converting {usd_file.name} …"):
        mjcf = _convert(
            usd_file,
            out_dir,
            sim_profile=None if sim_profile == "none" else sim_profile,
            generate_collision=collision,
            backend=backend,
        )
    s = summarize_mjcf(mjcf)
    console.print(f"[green]✓[/] {mjcf}")
    console.print(
        f"  bodies {s.bodies} · joints {s.joints} · actuators {s.actuators} · geoms {s.geoms} · "
        f"mass {s.total_mass_kg:.3f} kg · stable {'yes' if s.settles else '[red]no[/]'}"
    )
    console.print(f"  View it: [cyan]srmj view {mjcf}[/]")


@app.command()
def batch(
    config: Path = typer.Argument(..., exists=True, help="YAML file listing assets"),
    stop_on_error: bool = typer.Option(False, help="Stop at the first failing asset"),
    only: list[str] = typer.Option(None, "--only", help="Run only these asset ids (repeatable)"),
    fetch_only: bool = typer.Option(False, "--fetch-only", help="Download and inspect, no convert"),
) -> None:
    """Fetch, inspect and convert every asset in a YAML config, and write a report."""
    from simready_mujoco.api import convert as _convert
    from simready_mujoco.fetch import resolve_asset
    from simready_mujoco.sim.load import summarize_mjcf
    from simready_mujoco.usd.inspect import inspect_usd

    cfg = yaml.safe_load(config.read_text())
    out_root = Path(cfg.get("output_dir", "outputs"))
    profile = cfg.get("sim_profile", "manipulation")
    results = []

    assets = [e for e in cfg.get("assets", []) if not only or e["id"] in only]
    for entry in assets:
        aid = entry["id"]
        row: dict = {"id": aid, "category": entry.get("category"), "ok": False, "stage": "fetch"}
        try:
            with console.status(f"{aid}: fetching …"):
                res = resolve_asset(entry)
            row.update(usd=str(res.usd), license=res.license, package_id=res.package_id)
            row["stage"] = "inspect"
            report_i = inspect_usd(res.usd)
            row["dropped_by_baseline"] = report_i.dropped_by_baseline
            (out_root / aid).mkdir(parents=True, exist_ok=True)
            (out_root / aid / "inspection.json").write_text(
                json.dumps(report_i.to_dict(), indent=2, default=str)
            )
            if fetch_only:
                row["ok"] = True
                console.print(f"[green]✓[/] {aid} (fetched + inspected)")
                results.append(row)
                continue
            row["stage"] = "convert"
            usd = res.usd
            with console.status(f"{aid}: converting …"):
                mjcf = _convert(
                    usd,
                    out_root / aid,
                    sim_profile=profile,
                    generate_collision=cfg.get("generate_collision", False),
                )
            row["stage"] = "load"
            row.update(summarize_mjcf(mjcf).to_dict(), ok=True, stage="done")
            console.print(f"[green]✓[/] {aid}")
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
            console.print(f"[red]✗[/] {aid} (failed at {row['stage']}): {row['error']}")
            if stop_on_error:
                raise
        results.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    report = out_root / "batch_report.json"
    report.write_text(json.dumps(results, indent=2, default=str))

    t = Table(title="Batch results")
    for c in ("Asset", "Category", "OK", "Bodies", "Joints", "Actuators", "Mass kg", "Lost"):
        t.add_column(c)
    for r in results:
        t.add_row(
            r["id"],
            str(r.get("category") or "-"),
            "[green]✓[/]" if r["ok"] else f"[red]✗ {r['stage']}[/]",
            str(r.get("bodies", "-")),
            str(r.get("joints", "-")),
            str(r.get("actuators", "-")),
            f"{r['total_mass_kg']:.3f}" if "total_mass_kg" in r else "-",
            str(len(r.get("dropped_by_baseline", []))),
        )
    console.print(t)
    licences = {r["id"]: r.get("license") for r in results if r.get("license")}
    if licences:
        console.print("Licences: " + ", ".join(f"{k}: {v}" for k, v in licences.items()))
    console.print(f"Report: {report}  (per-asset inspection.json alongside)")
    if not all(r["ok"] for r in results):
        raise typer.Exit(1)


@app.command()
def view(mjcf_file: Path = typer.Argument(..., exists=True, help="MJCF .xml file")) -> None:
    """Open an MJCF in the interactive MuJoCo viewer."""
    from simready_mujoco.sim.load import view as _view

    _view(mjcf_file)


if __name__ == "__main__":
    app()
