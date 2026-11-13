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

RUNTIME_HELP = "Physics variant to compose: mujoco | physx | newton | authored (file default)"
VARIANT_HELP = "Extra variant selection SET=OPTION, repeatable (e.g. Physics=Newton_parallel_grip)"


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
    check("pillow (textures)", lambda: __import__("PIL").__version__)

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
    runtime: str = typer.Option("mujoco", "--runtime", "-r", help=RUNTIME_HELP),
    variant: list[str] = typer.Option(None, "--variant", "-v", help=VARIANT_HELP),
) -> None:
    """Show which physics properties an asset authors, and what the baseline converter drops."""
    from simready_mujoco.usd.inspect import inspect_usd
    from simready_mujoco.usd.stage import parse_variant_args

    r = inspect_usd(usd_file, runtime=runtime, variants=parse_variant_args(variant))
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
    if r.runtime != "authored":
        avail = "" if r.runtime_available else " [yellow](asset has no variant for it)[/]"
        console.print(f"runtime [cyan]{r.runtime}[/]{avail} · selection {r.selection or '{}'}")

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
    for label, items in (
        ("PhysX schemas", r.physx_schemas),
        ("Newton schemas", r.newton_schemas),
        ("MuJoCo schemas", r.mjc_schemas),
    ):
        if items:
            console.print(f"{label}: {', '.join(items)}")
    if r.mjc_prim_types:
        console.print(f"MuJoCo prims: {r.mjc_prim_types}")
    if r.mjc_attributes:
        console.print(f"mjc:* attributes ({len(r.mjc_attributes)}): {', '.join(r.mjc_attributes)}")
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
    backend: str = typer.Option("native", help="Conversion backend: native | upstream"),
    runtime: str = typer.Option("mujoco", "--runtime", "-r", help=RUNTIME_HELP),
    variant: list[str] = typer.Option(None, "--variant", "-v", help=VARIANT_HELP),
    verbose: bool = typer.Option(False, "--verbose", help="Show third-party log output"),
    actuators: str = typer.Option(
        "auto",
        "--actuators",
        help="Actuator source: auto | mjc (MuJoCo layer) | drive (UsdPhysics)",
    ),
    decompose: str = typer.Option(
        "auto",
        "--decompose",
        help="Convex decomposition: auto (concave + overlapping) | concave | all | off",
    ),
) -> None:
    """Convert one USD asset to MJCF and check it loads in MuJoCo."""
    from simready_mujoco.api import convert as _convert
    from simready_mujoco.sim.load import summarize_mjcf
    from simready_mujoco.usd.stage import parse_variant_args

    out_dir = out / usd_file.stem
    console.print(f"Converting {usd_file.name} (runtime {runtime}) …")
    mjcf = _convert(
        usd_file,
        out_dir,
        sim_profile=None if sim_profile == "none" else sim_profile,
        generate_collision=collision,
        backend=backend,
        runtime=runtime,
        variants=parse_variant_args(variant),
        quiet=not verbose,
        actuators=actuators,
        decompose=decompose,
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
    verbose: bool = typer.Option(False, "--verbose", help="Show third-party log output"),
) -> None:
    """Fetch, inspect and convert every asset in a YAML config, and write a report."""
    from simready_mujoco.api import convert as _convert
    from simready_mujoco.fetch import resolve_asset
    from simready_mujoco.sim.load import summarize_mjcf
    from simready_mujoco.usd.inspect import inspect_usd

    cfg = yaml.safe_load(config.read_text())
    out_root = Path(cfg.get("output_dir", "outputs"))
    profile = cfg.get("sim_profile", "manipulation")
    default_runtime = cfg.get("runtime", "mujoco")
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
            rt = entry.get("runtime", default_runtime)
            vs = entry.get("variants") or {}
            report_i = inspect_usd(res.usd, runtime=rt, variants=vs)
            row.update(
                runtime=rt,
                runtime_available=report_i.runtime_available,
                selection=report_i.selection,
                mjc_schemas=report_i.mjc_schemas,
                mjc_prim_types=report_i.mjc_prim_types,
            )
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
            console.print(f"  {aid}: converting …")
            mjcf = _convert(
                res.usd,
                out_root / aid,
                sim_profile=profile,
                generate_collision=cfg.get("generate_collision", False),
                runtime=rt,
                variants=vs,
                quiet=not verbose,
                actuators=entry.get("actuators", cfg.get("actuators", "auto")),
                decompose=entry.get("decompose", cfg.get("decompose", "auto")),
                articulation=entry.get("articulation"),
                authored_mass=entry.get("authored_mass", cfg.get("authored_mass", "auto")),
            )
            row["stage"] = "load"
            row.update(summarize_mjcf(mjcf).to_dict(), ok=True, stage="done")
            rep = mjcf.parent / "conversion_report.json"
            if rep.exists():
                conv = json.loads(rep.read_text())
                row["warnings"] = conv.get("warnings", [])
                row["stiff_drives"] = len(conv.get("stiff_drives", []))
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
    for c in (
        "Asset",
        "Category",
        "OK",
        "MuJoCo layer",
        "Bodies",
        "Joints",
        "Act.",
        "Mass kg",
        "Warn",
    ):
        t.add_column(c)
    for r in results:
        t.add_row(
            r["id"],
            str(r.get("category") or "-"),
            "[green]✓[/]" if r["ok"] else f"[red]✗ {r['stage']}[/]",
            _mjc_cell(r),
            str(r.get("bodies", "-")),
            str(r.get("joints", "-")),
            str(r.get("actuators", "-")),
            f"{r['total_mass_kg']:.3f}" if "total_mass_kg" in r else "-",
            str(len(r.get("warnings", []))) if "warnings" in r else "-",
        )
    console.print(t)
    for r in results:
        for w in r.get("warnings", []):
            if w.startswith("UNSTABLE") or "mass properties" in w or "zero mass" in w:
                console.print(f"[red]✗ {r['id']}: {w}[/]")
    licences = {r["id"]: r.get("license") for r in results if r.get("license")}
    if licences:
        console.print("Licences: " + ", ".join(f"{k}: {v}" for k, v in licences.items()))
    console.print(
        f"Report: {report}  (per asset: inspection.json, conversion_report.json, convert.log)"
    )
    if not all(r["ok"] for r in results):
        raise typer.Exit(1)


def _mjc_cell(r: dict) -> str:
    """Short summary of the MuJoCo runtime layer for the batch table."""
    if r.get("runtime") != "mujoco":
        return r.get("runtime") or "-"
    if not r.get("runtime_available"):
        return "[yellow]none[/]"
    n_prims = sum((r.get("mjc_prim_types") or {}).values())
    return f"{len(r.get('mjc_schemas') or [])} API, {n_prims} prims"


@app.command(name="compare")
def compare_cmd(
    ours: Path = typer.Argument(..., exists=True, help="Our converted MJCF (A)"),
    reference: Path = typer.Argument(..., exists=True, help="Reference MJCF, e.g. Menagerie (B)"),
    seconds: float = typer.Option(2.0, help="Duration of the hold-pose test"),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Compare two models of the same mechanism: masses, joints, actuators, hold-pose test."""
    import dataclasses
    import math

    from simready_mujoco.sim.compare import compare

    c = compare(ours, reference, seconds)
    if as_json:
        print(json.dumps(dataclasses.asdict(c), indent=2, default=str))
        return

    def pair(a, b, fmt="{:.4g}"):
        def f(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "-"
            if isinstance(x, tuple):
                return "[" + " ".join(fmt.format(v) for v in x) + "]"
            return fmt.format(x)

        same = a == b or (
            isinstance(a, float)
            and isinstance(b, float)
            and math.isclose(a, b, rel_tol=0.02, abs_tol=1e-6)
        )
        colour = "green" if same else "yellow"
        return f"[{colour}]{f(a)} | {f(b)}[/]"

    console.rule("A = ours  |  B = reference")
    console.print(f"A: {c.file_a}\nB: {c.file_b}")
    console.print(
        f"total mass {pair(*c.total_mass)} kg · joints {pair(*c.n_joints, '{}')} · "
        f"actuators {pair(*c.n_actuators, '{}')}"
    )
    t = Table(title="Joints (matched by order)  A | B")
    for col in ("A / B", "range", "armature", "damping", "frictionloss", "moved mass kg"):
        t.add_column(col)
    for j in c.joints:
        t.add_row(
            f"{j.name_a}\n{j.name_b}",
            pair(j.range_a, j.range_b),
            pair(*j.armature),
            pair(*j.damping),
            pair(*j.frictionloss),
            pair(*j.child_mass),
        )
    console.print(t)
    if c.actuators:
        t = Table(title="Actuators  A | B")
        for col in ("joint (A)", "kp", "kv", "forcerange"):
            t.add_column(col)
        for a in c.actuators:
            t.add_row(a.joint, pair(*a.kp), pair(*a.kv), pair(*a.forcerange))
        console.print(t)
    ha, hb = c.hold_error
    console.print(
        f"Hold-pose test ({seconds:g} s), max joint error: A {ha if ha is None else f'{ha:.4f}'} rad · B {hb if hb is None else f'{hb:.4f}'} rad"
    )
    for n in c.notes:
        console.print(f"[dim]{n}[/]")


@app.command()
def add(
    path: Path = typer.Argument(..., help="USD file, SimReady package folder, or folder of assets"),
    config: Path = typer.Option(Path("configs/subset.yaml"), "--config", "-c", exists=True),
    category: str = typer.Option(
        "object",
        help="Category for the new entries (robot/gripper are left out of scenes by default)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only list what would be added"),
) -> None:
    """Add downloaded USD assets to a batch config, ready for `srmj batch` and `srmj scene`."""
    from simready_mujoco.add import add_to_config, find_assets

    found, problems = find_assets(path)
    for p in problems:
        console.print(f"[yellow]! {p}[/]")
    if not found:
        console.print("[red]No USD assets found.[/]")
        raise typer.Exit(1)
    t = Table(title=f"Found in {path}")
    t.add_column("id")
    t.add_column("root USD", overflow="fold")
    for f in found:
        t.add_row(f.id, str(f.path))
    console.print(t)
    if dry_run:
        return
    added = add_to_config(config, found, category)
    console.print(f"Added {len(added)} to {config}: {', '.join(added) or '(all already listed)'}")
    if added:
        only = " ".join(f"--only {a}" for a in added)
        console.print(f"Next: [cyan]srmj batch {config} {only}[/] then [cyan]make scene[/]")


@app.command()
def scene(
    config: Path = typer.Argument(Path("configs/scene.yaml"), exists=True, help="Scene config"),
    out: Path = typer.Option(None, "--out", "-o", help="Scene file (default from config)"),
    open_viewer: bool = typer.Option(False, "--view", help="Open the scene in the viewer"),
) -> None:
    """Put converted assets in one scene: floor, table, objects placed automatically."""
    from simready_mujoco.sim.scene import (
        ArticulationSpec,
        Item,
        SheetSpec,
        TableSpec,
        build_scene,
    )

    cfg = yaml.safe_load(config.read_text()) or {}
    assets_cfg = yaml.safe_load(Path(cfg.get("assets_from", "configs/subset.yaml")).read_text())
    out_root = Path(assets_cfg.get("output_dir", "outputs"))
    place = cfg.get("place") or {}
    yaw = cfg.get("yaw") or {}
    items, missing = [], []
    for entry in assets_cfg.get("assets", []):
        aid = entry["id"]
        if place.get(aid) == "skip":
            continue
        xmls = sorted((out_root / aid).glob("*.xml"))
        if not xmls:
            missing.append(aid)
            continue
        items.append(
            Item(
                aid, str(xmls[0]), place.get(aid, "auto"), entry.get("category"), yaw.get(aid, 0.0)
            )
        )
    if missing:
        console.print(f"[yellow]Not converted yet (run srmj batch): {', '.join(missing)}[/]")
    out_file = out or Path(cfg.get("output", out_root / "scene" / "scene.xml"))
    table = TableSpec(**(cfg.get("table") or {}))
    sheet_cfg = dict(cfg.get("sheet") or {})
    if "rgba" in sheet_cfg:
        sheet_cfg["rgba"] = tuple(sheet_cfg["rgba"])
    with console.status("Building scene and letting it settle …"):
        rep = build_scene(
            items,
            out_file,
            table,
            cfg.get("sim_profile", "manipulation"),
            sheet=SheetSpec(**sheet_cfg),
            include_robots=bool(cfg.get("robots", False)),
            articulation=ArticulationSpec(**(cfg.get("articulation") or {})),
        )
    (out_file.parent / "scene_report.json").write_text(
        json.dumps(dataclasses_asdict(rep), indent=2)
    )

    t = Table(title=f"Scene: {out_file}")
    for c in ("Asset", "Placed on", "Position (x y z)", "Size (x y z) m"):
        t.add_column(c)
    for it in rep.items:
        size = [hi - lo for lo, hi in zip(it["bbox_min"], it["bbox_max"], strict=False)]
        t.add_row(
            it["id"],
            it["surface"],
            " ".join(f"{v:.2f}" for v in it["pos"]),
            " ".join(f"{v:.2f}" for v in size),
        )
    console.print(t)
    if rep.sheet:
        console.print(f"Floor sheet {rep.sheet['length']:.2f} × {rep.sheet['width']:.2f} m")
    else:
        console.print(
            f"Table {rep.table['length']:.2f} × {rep.table['width']:.2f} m, "
            f"top at {rep.table['height']:.2f} m"
        )
    if rep.skipped_robots:
        console.print(
            f"[dim]Robots/grippers left out (set robots: true in the scene config): "
            f"{', '.join(rep.skipped_robots)}[/]"
        )
    if rep.passive_motors:
        console.print(
            f"[dim]{len(rep.passive_motors)} object motors made drag-friendly "
            f"({', '.join(rep.passive_motors)}): Control panel sliders move them, and a drag "
            "overpowers them (articulation: motors: in the scene config)[/]"
        )
    if rep.momentary_joints:
        console.print(
            f"[dim]{len(rep.momentary_joints)} buttons/triggers/switches spring back when let "
            f"go: {', '.join(rep.momentary_joints)}[/]"
        )
    if rep.held_joints:
        console.print(
            f"[dim]{rep.held_joints} hinges/slides given enough friction to hold their part "
            "against its own weight (lids and flaps stay at any angle)[/]"
        )
    if rep.softened_joints:
        console.print(
            f"[dim]{rep.softened_joints} unactuated joints given light damping so dragged parts "
            "stay put (articulation: in the scene config)[/]"
        )
    for s in rep.settle:
        if s.get("error"):
            console.print(
                f"  [red]✗ {s['error']}[/]\n    The scene was not checked. Run "
                "`srmj batch configs/subset.yaml` and look at that asset's warnings."
            )
            continue
        mark = "[green]✓[/]" if s["ok"] else "[yellow]![/]"
        console.print(
            f"  {mark} {s['body']}: settled {s['drop_m'] * 1000:.0f} mm down, "
            f"slid {s['slide_m'] * 1000:.0f} mm"
        )
    console.print(f"View it: [cyan]srmj view {out_file}[/]")
    console.print(
        "[dim]In the viewer: double-click a part to select it, then\n"
        "  Ctrl + right-drag        lift / move it (vertical plane facing you)\n"
        "  Ctrl + Shift + right-drag  slide it along the floor\n"
        "  Ctrl + left-drag         twist it  ·  doors, lids and knobs: drag the part itself\n"
        "  Space pauses · Backspace resets[/]"
    )
    if open_viewer:
        from simready_mujoco.sim.load import view as _view

        _view(out_file)


def dataclasses_asdict(obj):
    import dataclasses

    return dataclasses.asdict(obj)


@app.command()
def parts(
    target: str = typer.Argument(
        ..., help="USD file, an asset id from the config, or 'all' for every asset"
    ),
    config: Path = typer.Option(Path("configs/subset.yaml"), "--config", "-c"),
    depth: int = typer.Option(4, "--depth", "-d", help="How deep to list the prim tree"),
    runtime: str = typer.Option("mujoco", "--runtime", "-r", help=RUNTIME_HELP),
) -> None:
    """List an asset's parts with their size and position, to write `articulation:` specs.

    Positions are metres in the asset's own frame, the frame `axis:` and `at:` refer to.
    """
    from simready_mujoco.usd.stage import open_stage

    if Path(target).exists():
        stage, _ = open_stage(target, runtime=runtime)
        _print_parts(Path(target).name, stage, depth, articulation=None)
        return

    from simready_mujoco.fetch import resolve_asset

    cfg = yaml.safe_load(config.read_text()) if config.exists() else {}
    entries = [e for e in cfg.get("assets", []) if target in ("all", e["id"])]
    if not entries:
        raise typer.BadParameter(f"'{target}' is neither a file nor an asset id in {config}")
    for entry in entries:
        usd = resolve_asset(entry).usd
        stage, _ = open_stage(
            usd,
            runtime=entry.get("runtime", cfg.get("runtime", runtime)),
            variants=entry.get("variants") or {},
        )
        _print_parts(entry["id"], stage, depth, articulation=entry.get("articulation"))


def _print_parts(title: str, stage, depth: int, articulation) -> None:
    from rich.table import Table

    from simready_mujoco.usd.articulate import describe_joints, describe_parts

    rows = describe_parts(stage, max_depth=depth)
    t = Table(title=f"{title}  (metres, asset frame)", title_justify="left")
    t.add_column("part", no_wrap=True)
    t.add_column("type", no_wrap=True)
    t.add_column("role", no_wrap=True)
    t.add_column("center x y z", no_wrap=True)
    t.add_column("size x y z", no_wrap=True)
    fmt = lambda v: " ".join(f"{c:6.3f}" for c in v) if v else ""  # noqa: E731
    for r in rows:
        name = "  " * r["depth"] + r["path"].split("/")[-1]
        t.add_row(name, r["type"], r["role"], fmt(r.get("center")), fmt(r.get("size")))
    from rich.console import Console

    Console(width=max(console.width, 120)).print(t)  # full names even in narrow terminals
    joints = describe_joints(stage)
    if joints:
        jt = Table(title="joints", title_justify="left")
        for col in ("joint", "type", "connects", "axis", "limits"):
            jt.add_column(col, no_wrap=True)
        for j in joints:
            jt.add_row(j["joint"], j["type"], j["connects"], j["axis"], j["limits"])
        Console(width=max(console.width, 120)).print(jt)
    if articulation:
        console.print(f"[green]{len(articulation)} joint(s) added from the config[/]\n")
    elif joints:
        console.print(f"[green]{len(joints)} joint(s) authored in the asset[/]\n")
    else:
        console.print(
            "[yellow]No joints authored.[/] To make a part move, add to this asset in the config:\n"
            "  articulation:\n"
            "    - {part: <name>, joint: hinge, axis: z, at: [-x, +y], range: [0, 110]}\n"
            "[dim]See docs/articulation.md[/]\n"
        )


@app.command()
def view(mjcf_file: Path = typer.Argument(..., exists=True, help="MJCF .xml file")) -> None:
    """Open an MJCF in the interactive MuJoCo viewer."""
    from simready_mujoco.sim.load import view as _view

    _view(mjcf_file)


if __name__ == "__main__":
    app()
