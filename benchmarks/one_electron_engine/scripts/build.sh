#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD="${BUILD:-$ROOT/benchmarks/one_electron_engine/build}"
case "${SIMD:-baseline}" in
    baseline|sse2) ONEE_ISA=baseline ;;
    avx2|avx512|native) ONEE_ISA="$SIMD" ;;
    auto) ONEE_ISA=native ;;
    *) echo "SIMD must be baseline, sse2, avx2, avx512 or native" >&2; exit 2 ;;
esac

OPTIONS=(-S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
    -DTLNO_SIMD="$ONEE_ISA" -DTLNO_BUILD_ONEE_TOOLS=ON)
if [[ -n "${CXX:-}" ]]; then OPTIONS+=(-DCMAKE_CXX_COMPILER="$CXX"); fi
if [[ -n "${EXTRA_CXXFLAGS:-}" ]]; then
    OPTIONS+=(-DCMAKE_CXX_FLAGS="$EXTRA_CXXFLAGS")
fi
cmake "${OPTIONS[@]}"
cmake --build "$BUILD" --parallel "${JOBS:-4}" "$@"
printf 'Built one-electron workflow in %s\n' "$BUILD"