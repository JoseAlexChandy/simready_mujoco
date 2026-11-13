"""MuJoCo viewer with keyboard placement of the selected object.

Double-click an object (or any of its parts) to select it, then:

    Arrow keys   move it along the floor (up = away from you, relative to the camera)
    W / S        raise / lower
    A / D        turn left / right
    Q / E        tilt forward / back
    Z / C        roll left / right
    R            restore its original orientation
    Shift + key  bigger steps (5 cm / 15 deg instead of 1 cm / 5 deg)

Hold a key to keep moving. Rotations turn the object about its centre of mass. Works
paused or running (pause with Space to place precisely).

While an object is selected, these keys go to the object instead of the viewer. With
nothing selected (double-click empty space), every key does what the viewer normally does:
letters toggle display options (W wireframe, S shadows, ...), arrows step the history while
paused. Ctrl/Alt combinations always go to the viewer. Mouse dragging, the Control panel
sliders and everything else work as before.

How: the viewer's GLFW key callback is wrapped (GLFW is shared with MuJoCo's viewer through
the `glfw` package), so keys meant for the object never reach the viewer's shortcuts. If
that is not possible on some platform, the keys are taken through the viewer's own key
callback instead, and the viewer's shortcuts then also fire.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# GLFW key codes
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_A, KEY_C, KEY_D, KEY_E, KEY_Q, KEY_R, KEY_S, KEY_W, KEY_Z = 65, 67, 68, 69, 81, 82, 83, 87, 90
MOD_SHIFT, MOD_CONTROL, MOD_ALT, MOD_SUPER = 1, 2, 4, 8
RELEASE, PRESS, REPEAT = 0, 1, 2

STEP = (0.01, 5.0)  # m, deg
BIG_STEP = (0.05, 15.0)

MOVE_KEYS = {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_W, KEY_S}
TURN_KEYS = {KEY_A, KEY_D, KEY_Q, KEY_E, KEY_Z, KEY_C}
OBJECT_KEYS = MOVE_KEYS | TURN_KEYS | {KEY_R}

HELP = """Keyboard placement: double-click an object, then
  arrows  move along the floor      W/S  up/down       A/D  turn
  Q/E     tilt                      Z/C  roll          R    original orientation
  Shift   bigger steps              (double-click empty space: keys go back to the viewer)"""


@dataclass
class _Target:
    root: int  # body id of the object's free-floating root
    qadr: int  # qpos address of its free joint (pos xyz, quat wxyz)
    vadr: int  # qvel address (6 dofs)


class KeyboardMover:
    """Moves the selected object's free root body from keyboard input.

    `pert` and `cam` are the viewer's MjvPerturb (selection) and MjvCamera (view
    direction). `handle(key, mods)` returns True when the key was used for the object.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, pert=None, cam=None, echo=print):
        self.m, self.d = m, d
        self.pert, self.cam = pert, cam
        self.echo = echo
        self.qpos0 = m.qpos0.copy()
        self.status = "Keyboard placement: double-click an object"
        self._warned: set[int] = set()

    # ------------------------------------------------------------------ selection
    def selected(self) -> int:
        return int(self.pert.select) if self.pert is not None else 0

    def target(self) -> _Target | None:
        body = self.selected()
        if body <= 0:
            return None
        m = self.m
        while m.body_parentid[body] != 0:
            body = int(m.body_parentid[body])
        for j in range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return _Target(body, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        return None

    def wants(self, key: int, mods: int = 0) -> bool:
        """Should this key go to the object (rather than to the viewer)?"""
        if key not in OBJECT_KEYS or mods & (MOD_CONTROL | MOD_ALT | MOD_SUPER):
            return False
        if self.target() is not None:
            return True
        body = self.selected()
        if body > 0 and body not in self._warned:  # selected, but fixed to the world
            self._warned.add(body)
            self._say(f"{self.m.body(body).name} is fixed to the world; keys go to the viewer")
        return False

    # ------------------------------------------------------------------ directions
    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal (forward, right) unit vectors of the current view."""
        az = math.radians(float(self.cam.azimuth) if self.cam is not None else 90.0)
        return np.array([math.cos(az), math.sin(az), 0.0]), np.array(
            [math.sin(az), -math.cos(az), 0.0]
        )

    # ------------------------------------------------------------------ actions
    def handle(self, key: int, mods: int = 0) -> bool:
        if not self.wants(key, mods):
            return False
        t = self.target()
        dist, ang = BIG_STEP if mods & MOD_SHIFT else STEP
        forward, right = self._axes()
        up = np.array([0.0, 0.0, 1.0])
        if key == KEY_R:
            self._set_quat(t, self.qpos0[t.qadr + 3 : t.qadr + 7])
        elif key in MOVE_KEYS:
            delta = {
                KEY_UP: forward,
                KEY_DOWN: -forward,
                KEY_RIGHT: right,
                KEY_LEFT: -right,
                KEY_W: up,
                KEY_S: -up,
            }[key]
            self.d.qpos[t.qadr : t.qadr + 3] += delta * dist
        else:
            axis, sign = {
                KEY_A: (up, 1.0),  # turn left (counter-clockwise seen from above)
                KEY_D: (up, -1.0),
                KEY_Q: (right, -1.0),  # tilt the top away from you
                KEY_E: (right, 1.0),
                KEY_Z: (forward, -1.0),  # roll left
                KEY_C: (forward, 1.0),
            }[key]
            self._rotate(t, axis, sign * math.radians(ang))
        self.d.qvel[t.vadr : t.vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._report(t)
        return True

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
        w, x, y, z = self.d.qpos[t.qadr + 3 : t.qadr + 7]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        name = self.m.body(t.root).name.split("/")[0]
        self._say(
            f"{name}: pos {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} m  "
            f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  roll {roll:+.0f} deg"
        )

    def _say(self, text: str) -> None:
        self.status = text
        if self.echo:
            self.echo(text)


# ---------------------------------------------------------------------------- GLFW hook

_KEYFUN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
)


class _KeyHook:
    """Sits in front of the viewer's GLFW key callback: keys for the selected object are
    handled here (under the viewer's lock) and never reach the viewer's shortcuts."""

    def __init__(self, mover: KeyboardMover):
        self.mover = mover
        self.sim = None
        self.prev = None
        self._fn = _KEYFUN(self._on_key)  # keep a reference: GLFW holds a raw pointer

    def install(self) -> bool:
        import glfw  # noqa: PLC0415  (the same GLFW library MuJoCo's viewer uses)

        lib = glfw._glfw
        lib.glfwGetCurrentContext.restype = ctypes.c_void_p
        lib.glfwSetKeyCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.glfwSetKeyCallback.restype = ctypes.c_void_p
        window = lib.glfwGetCurrentContext()
        if not window:
            return False
        prev = lib.glfwSetKeyCallback(window, ctypes.cast(self._fn, ctypes.c_void_p))
        self.prev = _KEYFUN(prev) if prev else None
        return True

    def _on_key(self, window, key, scancode, action, mods) -> None:
        try:
            if action in (PRESS, REPEAT) and self.mover.wants(key, mods):
                with self.sim.lock():
                    self.mover.handle(key, mods)
                return
            if action == RELEASE and key in OBJECT_KEYS and self.mover.target() is not None:
                return  # the press was ours; keep the release away from the viewer too
        except Exception as e:  # noqa: BLE001  (never take the viewer down)
            print(f"keyboard placement: {e}")
        if self.prev is not None:
            self.prev(window, key, scancode, action, mods)


def view(mjcf_file: str | Path, keyboard: bool = True) -> None:
    """Open the MuJoCo viewer; with `keyboard`, add keyboard placement of objects."""
    import mujoco.viewer as mv  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(mjcf_file))
    d = mujoco.MjData(m)
    original = getattr(mv, "_Simulate", None)
    launch_internal = getattr(mv, "_launch_internal", None)
    if not keyboard or original is None or launch_internal is None:
        if keyboard:
            print("Keyboard placement unavailable with this MuJoCo version; plain viewer.")
        mv.launch(m, d)
        return

    mover = KeyboardMover(m, d)
    hook = _KeyHook(mover)
    fallback = {"on": False}

    def capture(*args, **kwargs):
        sim = original(*args, **kwargs)  # creates the window on this (the UI) thread
        mover.cam, mover.pert = args[0], args[2]  # (cam, opt, pert, user_scn, ...)
        hook.sim = sim
        try:
            fallback["on"] = not hook.install()
        except Exception:  # noqa: BLE001
            fallback["on"] = True
        if fallback["on"]:
            print("Note: keys also trigger the viewer's own shortcuts on this platform.")
        return sim

    def key_callback(key: int) -> None:  # only used if the GLFW hook could not be installed
        if fallback["on"]:
            mover.handle(key, 0)

    capture.MAX_GEOM = getattr(original, "MAX_GEOM", 10000)
    print(HELP)
    mv._Simulate = capture
    try:
        launch_internal(m, d, run_physics_thread=True, key_callback=key_callback)
    finally:
        mv._Simulate = original
