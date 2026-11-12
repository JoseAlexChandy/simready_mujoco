# 0003 — Native backend: semantic choices

- **Date:** 2026-11-12
- **Status:** Accepted

Decisions made while building the native USD → MJCF backend, with the reason for each.

1. **Asset values beat presets.** `mjc:option:*` / `mjc:flag:*` authored in the asset are kept;
   `sim_profiles.yaml` only fills what the asset leaves unset. The asset author tuned for it.
2. **Schema defaults are MuJoCo's defaults.** usd-core has no mjcPhysics plugin, so unauthored
   attributes are absent. We write only authored attributes and let MuJoCo apply its own
   defaults, which the mjcPhysics schema mirrors (e.g. `gainprm=1`, `biastype=none`).
3. **Joint reference angle.** When a body is authored away from its joint's zero, we keep the
   authored pose and set MJCF `ref`, so ranges, actuator targets and mimic couplings stay in
   USD's absolute joint coordinates.
4. **Mimic couplings are re-based.** MuJoCo's joint equality is relative to each joint's `ref`
   (`y − y0 = a0 + a1(x − x0)`); USD couplings are absolute, so `a0 = c0 − y0 + c1·x0`.
5. **PhysX mimic sign.** PhysX defines `q + g·q_ref + o = 0`, but NVIDIA's own tests treat
   direction as a warning, so assets may use either sign. We use the PhysX convention unless
   the other one is the only one that fits the follower's joint limits (then we warn).
6. **Collision filtering follows USD.** Bodies joined by a joint don't collide unless
   `physics:collisionEnabled` is true; we emit explicit `<exclude>` pairs because MuJoCo's
   parent/child filter doesn't apply when the parent is welded to the world.
7. **Report, don't silently fix, stiff drives.** PhysX drives are implicit; MuJoCo's position
   stiffness is not. We compute ω·dt per actuated joint and suggest armature when > 0.5.
8. **Meshes as binary STL**, deduplicated per (source mesh, scale), for speed on 300 MB assets.
