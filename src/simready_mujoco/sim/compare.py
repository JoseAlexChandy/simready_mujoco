"""Compare two MJCF models of the same mechanism (e.g. our conversion vs MuJoCo Menagerie).

Joints are matched by order along the kinematic tree (hinge/slide only), since names differ
between sources. Actuators are matched by the joint they drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class JointRow:
    name_a: str
    name_b: str
    range_a: tuple[float, float] | None
    range_b: tuple[float, float] | None
    armature: tuple[float, float]
    damping: tuple[float, float]
    frictionloss: tuple[float, float]
    child_mass: tuple[float, float]  # mass of the body this joint moves


@dataclass
class ActuatorRow:
    joint: str
    kp: tuple[float, float]
    kv: tuple[float, float]
    forcerange: tuple[tuple[float, float] | None, tuple[float, float] | None]


@dataclass
class Comparison:
    file_a: str
    file_b: str
    total_mass: tuple[float, float]
    n_joints: tuple[int, int]
    n_actuators: tuple[int, int]
    joints: list[JointRow] = field(default_factory=list)
    actuators: list[ActuatorRow] = field(default_factory=list)
    hold_error: tuple[float | None, float | None] = (None, None)  # max |q - target| after hold
    notes: list[str] = field(default_factory=list)


def _scalar_joints(m: mujoco.MjModel) -> list[int]:
    return [
        j
        for j in range(m.njnt)
        if int(m.jnt_type[j])
        in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    ]


def _actuator_for_joint(m: mujoco.MjModel, j: int) -> int | None:
    for a in range(m.nu):
        if (
            int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and m.actuator_trnid[a, 0] == j
        ):
            return a
    return None


def _pd(m: mujoco.MjModel, a: int) -> tuple[float, float]:
    """Effective position gain and damping of an affine actuator: f = kp*(u - q) - kv*qdot."""
    return float(m.actuator_gainprm[a, 0]), float(-m.actuator_biasprm[a, 2])


def _range(m, j):
    return tuple(float(x) for x in m.jnt_range[j]) if m.jnt_limited[j] else None


def _frange(m, a):
    return tuple(float(x) for x in m.actuator_forcerange[a]) if m.actuator_forcelimited[a] else None


def hold_test(m: mujoco.MjModel, target: np.ndarray, seconds: float = 2.0) -> float | None:
    """Command `target` (one value per actuated scalar joint, in joint order), start there,
    simulate, and return the max joint error at the end. None if nothing is actuated."""
    joints = _scalar_joints(m)
    d = mujoco.MjData(m)
    acts = []
    for k, j in enumerate(joints[: len(target)]):
        d.qpos[m.jnt_qposadr[j]] = target[k]
        a = _actuator_for_joint(m, j)
        if a is not None:
            d.ctrl[a] = target[k]
            acts.append((j, k))
    if not acts:
        return None
    mujoco.mj_forward(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(max(abs(d.qpos[m.jnt_qposadr[j]] - target[k]) for j, k in acts))


def compare(file_a: str | Path, file_b: str | Path, hold_seconds: float = 2.0) -> Comparison:
    ma = mujoco.MjModel.from_xml_path(str(file_a))
    mb = mujoco.MjModel.from_xml_path(str(file_b))
    ja, jb = _scalar_joints(ma), _scalar_joints(mb)
    c = Comparison(
        str(file_a),
        str(file_b),
        (float(ma.body_mass.sum()), float(mb.body_mass.sum())),
        (len(ja), len(jb)),
        (ma.nu, mb.nu),
    )
    if len(ja) != len(jb):
        c.notes.append(
            f"joint count differs ({len(ja)} vs {len(jb)}); comparing the first {min(len(ja), len(jb))}"
        )

    for x, y in zip(ja, jb, strict=False):
        c.joints.append(
            JointRow(
                ma.joint(x).name,
                mb.joint(y).name,
                _range(ma, x),
                _range(mb, y),
                (
                    float(ma.dof_armature[ma.jnt_dofadr[x]]),
                    float(mb.dof_armature[mb.jnt_dofadr[y]]),
                ),
                (float(ma.dof_damping[ma.jnt_dofadr[x]]), float(mb.dof_damping[mb.jnt_dofadr[y]])),
                (
                    float(ma.dof_frictionloss[ma.jnt_dofadr[x]]),
                    float(mb.dof_frictionloss[mb.jnt_dofadr[y]]),
                ),
                (float(ma.body_mass[ma.jnt_bodyid[x]]), float(mb.body_mass[mb.jnt_bodyid[y]])),
            )
        )
        aa, ab = _actuator_for_joint(ma, x), _actuator_for_joint(mb, y)
        if aa is not None or ab is not None:
            pa = _pd(ma, aa) if aa is not None else (float("nan"),) * 2
            pb = _pd(mb, ab) if ab is not None else (float("nan"),) * 2
            c.actuators.append(
                ActuatorRow(
                    ma.joint(x).name,
                    (pa[0], pb[0]),
                    (pa[1], pb[1]),
                    (
                        _frange(ma, aa) if aa is not None else None,
                        _frange(mb, ab) if ab is not None else None,
                    ),
                )
            )

    # Hold test: B's first keyframe if it has one, else the midpoint of B's joint ranges.
    n = min(len(ja), len(jb))
    if mb.nkey > 0:
        target = np.array([mb.key_qpos[0, mb.jnt_qposadr[j]] for j in jb[:n]])
        c.notes.append(f"hold target: keyframe '{mb.key(0).name}' of B")
    else:
        target = np.array([np.mean(mb.jnt_range[j]) if mb.jnt_limited[j] else 0.0 for j in jb[:n]])
        c.notes.append("hold target: middle of B's joint ranges")
    c.hold_error = (hold_test(ma, target, hold_seconds), hold_test(mb, target, hold_seconds))
    return c
