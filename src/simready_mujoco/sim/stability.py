"""Check whether position actuators are too stiff for the model's timestep.

PhysX integrates joint drives implicitly, so assets authored for it can use stiff drives on
light links. MuJoCo's integrators treat actuator *stiffness* explicitly, which is only stable
when the drive's natural frequency w = sqrt(kp / I_eff) satisfies roughly w*dt < 1 (we warn
above 0.5). The fix is joint `armature` (reflected rotor inertia, which real gearboxes have),
a smaller timestep, or lower gains: this module reports, it never changes the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

LIMIT = 0.5


@dataclass
class StiffDrive:
    actuator: str
    joint: str
    kp: float
    effective_inertia: float
    omega_dt: float
    armature_needed: float  # extra armature that brings omega*dt down to LIMIT/2

    def to_dict(self) -> dict:
        return asdict(self)


def _full_mass_matrix(m: mujoco.MjModel, d: mujoco.MjData, out: np.ndarray) -> None:
    """Dense joint-space inertia matrix; mj_fullM's signature changed in MuJoCo 4."""
    try:
        mujoco.mj_fullM(m, d, out)  # MuJoCo >= 4: (model, data, dst)
    except TypeError:
        mujoco.mj_fullM(m, out, d.qM)  # MuJoCo 3: (model, dst, qM)


def check_drive_stability(mjcf: str | Path) -> list[StiffDrive]:
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    full = np.zeros((m.nv, m.nv))
    _full_mass_matrix(m, d, full)
    dt = m.opt.timestep
    out = []
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        kp = -float(m.actuator_biasprm[a, 1])  # affine bias: -kp * q
        if kp <= 0:
            continue
        j = int(m.actuator_trnid[a, 0])
        dof = int(m.jnt_dofadr[j])
        inertia = float(full[dof, dof])  # includes armature
        if inertia <= 0:
            continue
        w_dt = math.sqrt(kp / inertia) * dt
        if w_dt > LIMIT:
            target = (LIMIT / 2 / dt) ** 2  # kp / I_needed
            out.append(
                StiffDrive(
                    m.actuator(a).name,
                    m.joint(j).name,
                    kp,
                    inertia,
                    w_dt,
                    max(kp / target - inertia, 0.0),
                )
            )
    return out
