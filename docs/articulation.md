# Making parts move: `articulation:` specs

Some SimReady assets author their joints (the refrigerator, file cabinet and faucet). In the
MuJoCo viewer you can grab their doors, drawers and levers and move them.

Many props don't. A microwave's door, a laptop's lid or an air fryer's basket is often its own
mesh inside **one** rigid body, so the whole object moves as a block. For those, the asset's
entry in `configs/subset.yaml` lists the parts that should move. The converter then adds a
hinge or slider for each one, on the USD session layer, so the asset files are never edited.

## 1. Find the part

```bash
srmj parts microwave_oven        # one asset (id from the config, or a USD path)
srmj parts all                   # every asset in the config
```

This prints the prim tree with each part's **centre and size in metres, in the asset's own
frame**. The `axis:` and `at:` settings below use the same frame. The footer says whether the
asset already has joints.

Pick the prim that holds the moving geometry. Prefer the Xform that groups it (`door`) over a
single mesh inside it, so that every mesh of the door moves together.

## 2. Describe how it moves

```yaml
  - id: microwave_oven
    path: ...
    articulation:
      - part: door              # name (if unique) or path relative to the default prim
        joint: hinge            # hinge | slide
        axis: -z                # hinge: rotation axis; + direction opens (right-hand rule)
        at: [-x, -y]            # hinge line runs along the part's bounding-box edge at
                                #   min x, min y  (+x = max, x = centre, omit = centre)
        range: [0, 110]         # degrees (hinge) or metres (slide), 0 = as modelled
      - part: button
        joint: slide
        axis: +y                # slide: direction it moves when pressed / opened
        range: [0, 0.005]
        spring: 200             # springs back to 0: N/m (slide) or N·m/rad (hinge)
        damping: 1
```

| key | meaning |
|---|---|
| `part` | prim that moves (it and everything under it) |
| `joint` | `hinge` (revolute) or `slide` (prismatic) |
| `axis` | `x`, `y` or `z` in the asset frame, with an optional sign |
| `at` | hinge only. For each of the two axes other than the hinge axis, `-x`/`+x` puts the hinge at the part's min/max on that axis; centre if omitted |
| `range` | limits. `0` is the pose in the file (normally closed) |
| `spring` | optional stiffness pulling the part back to 0: buttons, spring-loaded lids |
| `damping` | optional |
| `mass` | optional kg. By default the part takes a share of the object's authored mass in proportion to its hull volume (at most half), so the total stays the same |
| `colliders` | optional extra prims that move with the part (for collision meshes stored outside the part's subtree) |
| `parent` | optional; default is the nearest rigid body above the part |
| `name` | optional joint name; default `<part>_<joint>` |

Quick rules for picking `axis` and `at`:

- **Side-hinged door** (microwave, fridge): `axis: ±z`. `at:` names the hinge side (`-x` or
  `+x`) and the front face (usually `-y` or `+y`). If the door swings into the body, flip the
  axis sign.
- **Lid** (laptop, air fryer top): `axis: ±x`. `at:` names the back edge (`-y`/`+y`) and the
  bottom of the lid (`-z`).
- **Drawer / basket**: `joint: slide`, with `axis:` pointing out of the front.
- **Push button**: a short `slide` into the body, with a `spring`.
- **Rotary knob**: `hinge` through its centre (omit `at:`), with no `range` if it turns freely.

## 3. Rebuild

```bash
srmj batch configs/subset.yaml --only microwave_oven
make scene
```

`conversion_report.json` lists each added joint with its anchor, axis and mass under
`added_joints`.

## How it works

`usd/articulate.py` applies `RigidBodyAPI` to the part and authors a `RevoluteJoint` or
`PrismaticJoint` between it and its parent. The joint's frame sits on the requested
bounding-box edge, and `collisionEnabled = false` stops the door colliding with the carcass it
sits in. Spring and damping are written as `mjc:stiffness` / `mjc:damping`. From there the
normal pipeline handles the joint like any authored joint, so the same spec would also work in
Isaac Sim if the session layer were exported.
