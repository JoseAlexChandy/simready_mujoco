#!/usr/bin/env bash
# Validate the native converter on a known-good model:
#   MuJoCo Menagerie MJCF --(NVIDIA mujoco-usd-converter)--> USD --(srmj, native)--> MJCF
# then compare the result with the original (masses, joints, actuators, hold-pose test).
#
# Usage: scripts/roundtrip_menagerie.sh [menagerie model, default franka_fr3/fr3.xml]
# Env:   MENAGERIE  path to a mujoco_menagerie checkout (cloned if missing)
#        WORK       scratch folder (default /tmp/srmj_roundtrip)
set -euo pipefail

MODEL="${1:-franka_fr3/fr3.xml}"
MENAGERIE="${MENAGERIE:-$HOME/mujoco_menagerie}"
WORK="${WORK:-/tmp/srmj_roundtrip}"
SRMJ="${SRMJ:-srmj}"
NAME="$(basename "$MODEL" .xml)"

if [ ! -d "$MENAGERIE" ]; then
  echo "Cloning MuJoCo Menagerie into $MENAGERIE"
  git clone -q --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
fi

# NVIDIA's converter ships its own OpenUSD build, so keep it in a separate venv.
if [ ! -x "$WORK/venv/bin/mujoco_usd_converter" ]; then
  echo "Installing mujoco-usd-converter into $WORK/venv"
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q mujoco-usd-converter
fi

rm -rf "$WORK/usd" "$WORK/mjcf"
echo "1/3 MJCF -> USD"
env -u PYTHONPATH "$WORK/venv/bin/mujoco_usd_converter" "$MENAGERIE/$MODEL" "$WORK/usd" 2>&1 | grep -v "^\[Warning\]" || true
echo "2/3 USD -> MJCF (srmj native)"
env -u PYTHONPATH "$SRMJ" convert "$WORK/usd/$NAME.usda" -o "$WORK/mjcf"
echo "3/3 compare with the original"
env -u PYTHONPATH "$SRMJ" compare "$WORK/mjcf/$NAME/$NAME.xml" "$MENAGERIE/$MODEL"
