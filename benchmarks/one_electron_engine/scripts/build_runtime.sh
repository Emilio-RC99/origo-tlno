#!/usr/bin/env bash
# Build only the shared C ABI through the same CMake configuration as the tools.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build.sh" --target tlno_onee_capi "$@"
