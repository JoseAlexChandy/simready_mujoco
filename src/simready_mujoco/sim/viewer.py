"""MuJoCo viewer with keyboard placement of objects.

Double-click an object to select it, then:

    Move mode (default)              Rotate mode (F9 toggles)
    Home / End     away / towards    Home / End     tilt forward / back
    Insert / Delete  left / right    Insert / Delete  turn left / right
    Page Up / Down   up / down       Page Up / Down   roll left / right

    F10  step size (fine 5 mm / 1 deg, normal 2 cm / 5 deg, coarse 10 cm / 15 deg)
    F12  restore the object's original orientation (position kept)

Directions follow the camera: "away" is the way the camera looks, flattened onto the floor.
Rotations turn the object about its centre of mass. Works paused or running.

Only objects that are free to move (not fixed to the world) can be placed; selecting a
door or a lid moves the whole object it belongs to. Mouse dragging, the Control panel
sliders and every other viewer feature work as before.

Why these keys: letters toggle the viewer's display options, 0-5 toggle geom groups, and
the arrow keys step the simulation history while paused (which would undo a move).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# GLFW key codes
KEY_INSERT, KEY_DELETE = 260, 261
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269
KEY_F9, KEY_F10, KEY_F12 = 298, 299, 301

STEPS = (("fine", 0.005, 1.0), ("normal", 0.02, 5.0), ("coarse", 0.10, 15.0))

HELP = """Keyboard placement: double-click an object, then
  Home/End  Insert/Delete  PgUp/PgDn   move (away/towards, left/right, up/down)
  F9        switch move <-> rotate (tilt, turn, roll on the same keys)
  F10       step size        F12  restore original orientation"""


@dataclass
class _Target:
    root: int  # body id of the object's free-floating root
    qadr: int  # qpos address of its free joint (pos xyz, quat wxyz)
    vadr: int  # qvel address (6 dofs)


class KeyboardMover:
    """Moves the selected object's free root body from keyboard input.

    `pert` and `cam` are the viewer's MjvPerturb and MjvCamera (selection and view
    direction). Methods mutate `d` in place and call mj_forward.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, pert=None, cam=None, echo=print):
        self.m, self.d = m, d
        self.pert, self.cam = pert, cam
        self.rotate = False
        self.step = 1
        self.echo = echo
        self.qpos0 = m.qpos0.copy()
        self.status = "Keyboard placement: double-click an object"

    # ------------------------------------------------------------------ selection
    def target(self) -> _Target | None:
        body = int(self.pert.select) if self.pert is not None else 0
        if body <= 0:
            return None
        m = self.m
        while m.body_parentid[body] != 0:
            body = int(m.body_parentid[body])
        for j in range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return _Target(body, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        return None

    # ------------------------------------------------------------------ directions
    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal (forward, right) unit vectors of the current view."""
        az = math.radians(float(self.cam.azimuth)) if self.cam is not None else 90.0
        forward = np.array([math.cos(az), math.sin(az), 0.0])
        right = np.array([math.sin(az), -math.cos(az), 0.0])
        return forward, right

    # ------------------------------------------------------------------ actions
    def on_key(self, key: int) -> None:
        if key == KEY_F9:
            self.rotate = not self.rotate
            self._say(f"mode: {'rotate' if self.rotate else 'move'}")
            return
        if key == KEY_F10:
            self.step = (self.step + 1) % len(STEPS)
            name, dist, ang = STEPS[self.step]
            self._say(f"step: {name} ({dist * 100:g} cm / {ang:g} deg)")
            return
        moves = (KEY_HOME, KEY_END, KEY_INSERT, KEY_DELETE, KEY_PAGE_UP, KEY_PAGE_DOWN)
        if key not in (*moves, KEY_F12):
            return
        t = self.target()
        if t is None:
            body = int(self.pert.select) if self.pert is not None else 0
            self._say(
                "double-click an object first"
                if body <= 0
                else f"{self.m.body(body).name} is fixed to the world and can't be moved"
            )
            return
        _, dist, ang = STEPS[self.step]
        forward, right = self._axes()
        up = np.array([0.0, 0.0, 1.0])
        if key == KEY_F12:
            self._set_quat(t, self.qpos0[t.qadr + 3 : t.qadr + 7])
        elif not self.rotate:
            delta = {
                KEY_HOME: forward,
                KEY_END: -forward,
                KEY_DELETE: right,
                KEY_INSERT: -right,
                KEY_PAGE_UP: up,
                KEY_PAGE_DOWN: -up,
            }[key] * dist
            self.d.qpos[t.qadr : t.qadr + 3] += delta
        else:
            axis, sign = {
                KEY_HOME: (right, -1.0),  # tilt the top away from the camera
                KEY_END: (right, 1.0),
                KEY_INSERT: (up, 1.0),  # turn left (counter-clockwise from above)
                KEY_DELETE: (up, -1.0),
                KEY_PAGE_UP: (forward, -1.0),  # roll left
                KEY_PAGE_DOWN: (forward, 1.0),
            }[key]
            self._rotate(t, axis, sign * math.radians(ang))
        self.d.qvel[t.vadr : t.vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._report(t)

    def _rotate(self, t: _Target, axis: np.ndarray, angle: float) -> None:
        """Rotate the object about its centre of mass by `angle` around world `axis`."""
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, angle)
        com = np.array(self.d.subtree_com[t.root])
        pos = self.d.qpos[t.qadr : t.qadr + 3]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, dq)
        self.d.qpos[t.qadr : t.qadr + 3] = com + rot.reshape(3, 3) @ (pos - com)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, dq, self.d.qpos[t.qadr + 3 : t.qadr + 7])
        mujoco.mju_normalize4(q)
        self.d.qpos[t.qadr + 3 : t.qadr + 7] = q

    def _set_quat(self, t: _Target, quat: np.ndarray) -> None:
        """Set the orientation, keeping the centre of mass where it is."""
        mujoco.mj_forward(self.m, self.d)
        com = np.array(self.d.subtree_com[t.root])
        self.d.qpos[t.qadr + 3 : t.qadr + 7] = quat
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[t.qadr : t.qadr + 3] += com - self.d.subtree_com[t.root]

    def _report(self, t: _Target) -> None:
        p = self.d.qpos[t.qadr : t.qadr + 3]
        q = self.d.qpos[t.qadr + 3 : t.qadr + 7]
        # yaw / pitch / roll (Z-Y-X) for display
        w, x, y, z = q
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        name = self.m.body(t.root).name.split("/")[0]
        self._say(
            f"{name}: pos {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} m  "
            f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  roll {roll:+.0f} deg"
        )

    def _say(self, text: str) -> None:
        mode = "rotate" if self.rotate else "move"
        step = STEPS[self.step][0]
        self.status = f"[{mode}, {step}] {text}"
        if self.echo:
            self.echo(self.status)


def view(mjcf_file: str | Path, keyboard: bool = True) -> None:
    """Open the MuJoCo viewer; with `keyboard`, add keyboard placement of objects."""
    import mujoco.viewer as mv  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(mjcf_file))
    d = mujoco.MjData(m)
    if not keyboard:
        mv.launch(m, d)
        return

    mover = KeyboardMover(m, d)
    original = getattr(mv, "_Simulate", None)
    launch_internal = getattr(mv, "_launch_internal", None)
    if original is None or launch_internal is None:  # viewer internals changed
        print("Keyboard placement unavailable with this MuJoCo version; opening plain viewer.")
        mv.launch(m, d)
        return

    def capture(*args, **kwargs):
        sim = original(*args, **kwargs)
        mover.cam, mover.pert = args[0], args[2]  # (cam, opt, pert, user_scn, ...)
        return sim

    capture.MAX_GEOM = getattr(original, "MAX_GEOM", 10000)
    print(HELP)
    mv._Simulate = capture
    try:
        launch_internal(m, d, run_physics_thread=True, key_callback=mover.on_key)
    finally:
        mv._Simulate = original
