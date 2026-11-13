"""Catch broken models before they reach the viewer: bad mass properties and divergence.

MuJoCo resets the simulation when it meets NaN/Inf, so a broken asset silently "rests" at its
start pose and placement checks pass. These checks name the body responsible instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

BAD_STATE = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)


@dataclass
class Divergence:
    time: float
    dof: int
    body: str
    what: str  # "qacc" | "qvel" | "qpos"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"simulation diverged at t={self.time:.3f} s on body {self.body} ({self.what})"


def mass_problems(m: mujoco.MjModel) -> list[str]:
    """Bodies whose mass properties are non-finite or non-positive."""
    out = []
    for b in range(1, m.nbody):
        name = m.body(b).name or f"body{b}"
        vals = np.concatenate(
            [[m.body_mass[b]], m.body_inertia[b], m.body_ipos[b], m.body_iquat[b]]
        )
        if not np.all(np.isfinite(vals)):
            out.append(f"{name}: non-finite mass properties")
        elif m.body_dofnum[b] and (m.body_mass[b] <= 0 or np.any(m.body_inertia[b] <= 0)):
            out.append(f"{name}: zero mass or inertia on a moving body")
    return out


def first_divergence(m: mujoco.MjModel, seconds: float = 1.0) -> Divergence | None:
    """Step a fresh copy of the model; return where it first produced NaN/Inf, if anywhere."""
    d = mujoco.MjData(m)
    for _ in range(max(1, int(seconds / m.opt.timestep))):
        before = d.qacc.copy(), d.qvel.copy()
        mujoco.mj_step(m, d)
        if any(d.warning[w].number for w in BAD_STATE):
            return _locate(m, d, before)
    return None


def _locate(m: mujoco.MjModel, d: mujoco.MjData, before) -> Divergence:
    for what, arr in (("qacc", d.qacc), ("qvel", d.qvel), ("qacc", before[0])):
        bad = np.flatnonzero(~np.isfinite(arr) | (np.abs(arr) > 1e10))
        if bad.size:
            dof = int(bad[0])
            return Divergence(float(d.time), dof, m.body(int(m.dof_bodyid[dof])).name, what)
    w = next(w for w in BAD_STATE if d.warning[w].number)
    dof = int(d.warning[w].lastinfo)
    body = m.body(int(m.dof_bodyid[min(dof, m.nv - 1)])).name if m.nv else "?"
    return Divergence(float(d.time), dof, body, w.name.removeprefix("mjWARN_BAD").lower())
