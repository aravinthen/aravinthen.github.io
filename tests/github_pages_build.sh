#!/usr/bin/env bash
# Optional exact-production-image validation. Native Bundler remains recommended.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_path=${1:-$(mktemp -d "${TMPDIR:-/tmp}/notebook-pages.XXXXXX")}
mkdir -p "$output_path"
output_path=$(cd "$output_path" && pwd)
case "$output_path/" in "$repo_root/"*) echo 'Choose a destination outside the checkout.' >&2; exit 1;; esac
source_path=${SOURCE_DIR:-$repo_root}
source_path=$(cd "$source_path" && pwd)
case "$output_path/" in "$source_path/"*) echo 'Choose a destination outside the source.' >&2; exit 1;; esac
docker run --rm --user "$(id -u):$(id -g)" \
  -e GITHUB_WORKSPACE=/github/workspace -e INPUT_SOURCE=. -e INPUT_DESTINATION=../output \
  -e GITHUB_REPOSITORY=aravinthen/aravinthen.github.io \
  --mount "type=bind,src=$source_path,dst=/github/workspace,readonly" \
  --mount "type=bind,src=$output_path,dst=/github/output" \
  "${PAGES_IMAGE:-ghcr.io/actions/jekyll-build-pages@sha256:6791ebfd912185ed59bfb5fb102664fa872496b79f87ff8b9cfba292a7345041}"
printf 'Build output: %s\n' "$output_path"
