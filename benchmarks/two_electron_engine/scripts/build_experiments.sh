#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 TARGET [avx2|avx512] [BUILD_DIR]"
    echo "example: CXX=g++ JOBS=2 bash $0 rys_native_lane_ablation avx2"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
target=$1
isa=${2:-avx2}
case "$isa" in
    avx2|avx512) ;;
    *) echo "ISA must be avx2 or avx512" >&2; exit 2 ;;
esac
build_dir=$(realpath -m "${3:-$repo_root/build-experiments-$isa}")

# CMake owns the source lists and flags; these settings retain the D1 build profile.
options=(-S "$repo_root" -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=OFF
    -DTLNO_SIMD="$isa" -DTLNO_TUNE=native
    -DTLNO_FAST_MATH=ON -DTLNO_ENABLE_LTO=ON -DTLNO_LTO_JOBS=2
    -DTLNO_BUILD_ONEE_TOOLS=OFF -DTLNO_BUILD_TWOE_TOOLS=ON
    -DTLNO_BUILD_EXPERIMENTS=ON)
if [[ -n ${CXX:-} ]]; then
    options+=(-DCMAKE_CXX_COMPILER="$CXX")
fi
cmake "${options[@]}"
cmake --build "$build_dir" --config Release --target "$target" --parallel "${JOBS:-2}"

# Keep the sidecar consumed by the Python runners, using the actual CMake command.
python3 - "$repo_root" "$build_dir" "$target" <<'PY'
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

root, build = map(Path, sys.argv[1:3])
target = sys.argv[3]
binary = build / target
if not binary.is_file():
    raise SystemExit(f"missing executable: {binary}")
entries = json.loads((build / "compile_commands.json").read_text())
entry = next(e for e in entries if f"CMakeFiles/{target}.dir/" in e["command"])
command = shlex.split(entry["command"])
cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines()
             if "=" in line and not line.startswith(("#", "//")))
compiler = next(value for key, value in cache.items() if key.startswith("CMAKE_CXX_COMPILER:"))
flags = command[1:]
for option in ("-o", "-c"):
    index = flags.index(option)
    del flags[index:index + 2]

metadata = {
    "purpose": f"Two-electron experiment: {target}",
    "tree_root": str(root),
    "compiler": compiler,
    "compiler_version": subprocess.check_output(
        [compiler, "-dumpfullversion", "-dumpversion"], text=True).strip(),
    "compile_flags": shlex.join(flags),
    "compile_command": entry["command"],
    "compile_commands": str(build / "compile_commands.json"),
}
for label, path in (
    ("app", Path(entry["file"])),
    ("rys_source", root / "integrals/two_electron/rys/rys_engine.ipp"),
    ("shell_contract", root / "core/shell/shell_contract.cpp"),
    ("shell_math", root / "core/shell/shell_math.cpp"),
    ("binary", binary),
):
    metadata[f"{label}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
provenance = Path(str(binary) + ".provenance.txt")
provenance.write_text("".join(f"{key}={value}\n" for key, value in metadata.items()))
print(f"binary={binary}")
print(f"provenance={provenance}")
PY
