"""Load an MJCF in MuJoCo, summarise it, and open the viewer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco


@dataclass
class ModelSummary:
    file: str
    bodies: int
    joints: int
    actuators: int
    geoms: int
    equality_constraints: int
    total_mass_kg: float
    timestep: float
    settles: bool  # stepped 1 s without NaNs/instability

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_mjcf(mjcf_file: str | Path, settle_seconds: float = 1.0) -> ModelSummary:
    """Compile the model and step it briefly to check it's numerically stable."""
    model = mujoco.MjModel.from_xml_path(str(mjcf_file))
    data = mujoco.MjData(model)
    settles = True
    for _ in range(int(settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            settles = False
            break
    return ModelSummary(
        file=str(mjcf_file),
        bodies=model.nbody - 1,  # exclude world
        joints=model.njnt,
        actuators=model.nu,
        geoms=model.ngeom,
        equality_constraints=model.neq,
        total_mass_kg=float(sum(model.body_mass)),
        timestep=float(model.opt.timestep),
        settles=settles,
    )


def view(mjcf_file: str | Path) -> None:
    """Open the interactive MuJoCo viewer (needs a display)."""
    from simready_mujoco.sim.viewer import view as _view  # noqa: PLC0415

    _view(mjcf_file)
