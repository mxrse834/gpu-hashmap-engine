#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 ]]; then
    set -- "$repo_root/test/file_a.bin" "$repo_root/test/file_b.bin"
fi

exec compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    "$repo_root/gpu-hashmap" "$@"
