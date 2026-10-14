#!/usr/bin/env bash
# Build with the exact GitHub Pages action image used by the reported CI run.
# Docker is needed only for this optional compatibility check.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_path=${1:-/workspace/site-tools/pages-site}
mkdir -p "$output_path"
output_path=$(realpath "$output_path")
case "$output_path/" in "$repo_root/"*) echo 'Choose a build destination outside the checkout.' >&2; exit 1;; esac
docker run --rm --user "$(id -u):$(id -g)" \
  -e GITHUB_WORKSPACE=/github/workspace \
  -e INPUT_SOURCE=. -e INPUT_DESTINATION=../output \
  -e GITHUB_REPOSITORY=aravinthen/aravinthen.github.io \
  --mount "type=bind,src=$repo_root,dst=/github/workspace,readonly" \
  --mount "type=bind,src=$output_path,dst=/github/output" \
  ghcr.io/actions/jekyll-build-pages@sha256:6791ebfd912185ed59bfb5fb102664fa872496b79f87ff8b9cfba292a7345041
