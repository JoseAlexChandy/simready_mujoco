#!/usr/bin/env bash
# Download the packed assets from this repo's GitHub Releases and unpack them in place,
# so a fresh clone has outputs/ (converted models, scene) and data/ (downloaded assets).
#
#   scripts/fetch_assets.sh                # newest release whose tag starts with "assets-"
#   scripts/fetch_assets.sh assets-2026-11-13
#   scripts/fetch_assets.sh --from-dir dist   # unpack local files instead (no download)
#
# Needs only curl, python3, sha256sum and tar. Existing files are overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
dl=.assets_download

if [[ "${1:-}" == "--from-dir" ]]; then
  dl="${2:?--from-dir needs a folder}"
else
  remote=$(git remote get-url origin)
  repo=$(sed -E 's#^.*github\.com[:/]([^/]+/[^/]+)$#\1#; s#\.git$##' <<<"$remote")
  api="https://api.github.com/repos/$repo"
  tag="${1:-}"
  if [[ -z "$tag" ]]; then
    tag=$(curl -fsSL "$api/releases" | python3 -c '
import json, sys
tags = [r["tag_name"] for r in json.load(sys.stdin) if r["tag_name"].startswith("assets-")]
print(tags[0] if tags else "")')
    [[ -n "$tag" ]] || { echo "No assets-* release in $repo." >&2; exit 1; }
  fi
  echo "Fetching $tag from $repo ..."
  mkdir -p "$dl"
  curl -fsSL "$api/releases/tags/$tag" | python3 -c '
import json, sys
for a in json.load(sys.stdin)["assets"]:
    print(a["name"], a["browser_download_url"])' | while read -r name url; do
    if [[ -f "$dl/$name" ]] && grep -q " $name\$" "$dl/SHA256SUMS" 2>/dev/null \
        && (cd "$dl" && grep " $name\$" SHA256SUMS | sha256sum -c --status); then
      echo "  have $name"
      continue
    fi
    echo "  $name"
    curl -fL --retry 3 -o "$dl/$name" "$url"
  done
fi

echo "Checking checksums ..."
(cd "$dl" && sha256sum -c --quiet SHA256SUMS)
for kind in outputs data; do
  parts=("$dl"/"$kind".tar.gz.part*)
  [[ -e "${parts[0]}" ]] || continue
  echo "Unpacking $kind ..."
  cat "${parts[@]}" | tar -xzf -
done
[[ -f "$dl/ATTRIBUTION.md" ]] && cp "$dl/ATTRIBUTION.md" outputs/ATTRIBUTION.md
echo "Done. Try: srmj view outputs/scene/scene.xml"
