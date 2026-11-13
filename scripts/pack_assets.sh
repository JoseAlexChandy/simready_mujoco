#!/usr/bin/env bash
# Pack the converted outputs and the downloaded assets for a GitHub Release.
#
#   scripts/pack_assets.sh [DATA_DIRS...]     # default data dir: data/koyo
#
# Writes dist/:
#   outputs.tar.gz.partNN     converted MJCF + meshes + textures + scene (outputs/)
#   data.tar.gz.partNN        downloaded assets (data/koyo by default; SimReady-by-link assets
#                             are not included: `srmj batch` downloads them again)
#   ATTRIBUTION.md            per-asset licence and source (CC-BY requires it)
#   SHA256SUMS                checksums, checked by scripts/fetch_assets.sh
# Parts are at most 1900 MB (GitHub release files must be under 2 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
data_dirs=("${@:-data/koyo}")
part_size="${PART_SIZE:-1900m}"

if grep -q "path: $PWD/" configs/subset.yaml 2>/dev/null; then
  echo "configs/subset.yaml has absolute paths ($PWD/...). Make them portable first:" >&2
  echo "  sed -i \"s#path: \$PWD/#path: #\" configs/subset.yaml" >&2
  exit 1
fi
[[ -d outputs ]] || { echo "No outputs/ yet: run srmj batch first." >&2; exit 1; }

gz=gzip; command -v pigz >/dev/null && gz=pigz
rm -rf dist && mkdir dist

echo "Packing outputs/ ($(du -sh outputs | cut -f1)) ..."
tar -cf - outputs | $gz -6 | split -b "$part_size" -d -a 2 - dist/outputs.tar.gz.part

existing=()
for d in "${data_dirs[@]}"; do [[ -d "$d" ]] && existing+=("$d"); done
if ((${#existing[@]})); then
  echo "Packing ${existing[*]} ($(du -shc "${existing[@]}" | tail -1 | cut -f1)) ..."
  tar -cf - "${existing[@]}" | $gz -6 | split -b "$part_size" -d -a 2 - dist/data.tar.gz.part
fi

python scripts/attribution.py configs/subset.yaml > dist/ATTRIBUTION.md
(cd dist && sha256sum -- * > SHA256SUMS)
echo
ls -lh dist
echo
echo "Next: make release   (uploads dist/ as a GitHub Release with gh)"
