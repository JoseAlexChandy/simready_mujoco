#!/usr/bin/env bash
# Unzip downloaded asset archives into data/<target>/<name>/, one folder per zip.
#
#   scripts/unpack.sh ~/Downloads                 # every .zip there -> data/downloads/
#   scripts/unpack.sh ~/Downloads data/koyo       # into data/koyo/
#   scripts/unpack.sh ~/Downloads/D00270.zip      # just one
#
# Zips already unpacked are skipped. Then: srmj add data/<target>
set -euo pipefail
src="${1:?usage: scripts/unpack.sh ZIP_OR_FOLDER [TARGET_DIR]}"
dest="${2:-data/downloads}"
mkdir -p "$dest"
if [[ -d "$src" ]]; then zips=("$src"/*.zip); else zips=("$src"); fi
n=0
for z in "${zips[@]}"; do
  [[ -f "$z" ]] || continue
  name="$(basename "$z" .zip)"
  if [[ -d "$dest/$name" ]]; then echo "  skip $name (already unpacked)"; continue; fi
  unzip -q -o "$z" -d "$dest/$name"
  echo "  $name"
  n=$((n + 1))
done
echo "Unpacked $n archive(s) into $dest. Next: srmj add $dest"
