To build the one-electron engine:
    Rebuild with: "JOBS=2 benchmarks/one_electron_engine/scripts/build.sh".
    Run:          "ctest --test-dir benchmarks/one_electron_engine/build --output-on-failure".

First commands for the two-electron engine timing consumer
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j2
    ctest --test-dir build --output-on-failure

Build two-electron experiments with the shared CMake wrapper (from the repository root):

```bash
CXX=/home/emilio/.local/gcc-13.3.0/bin/g++ \
bash benchmarks/two_electron_engine/scripts/build_experiments.sh \
    rys_native_lane_ablation avx2
```

```bash
CXX=/usr/bin/g++ JOBS=2 \
bash benchmarks/two_electron_engine/scripts/build_experiments.sh \
    rys_native_lane_ablation avx2
```

The arguments are the CMake executable target, ISA (`avx2` by default, or `avx512`),
and an optional build directory. The default directory is `build-experiments-<ISA>`.
`JOBS` controls build parallelism (default 2). The wrapper uses GCC, Release, C++17,
native CPU tuning, fast math, and `-flto=2`; CMake applies these settings to the
executable and its dependencies. Release also adds CMake's usual `-DNDEBUG`.
The binary's `.provenance.txt` records its compiler, actual compile command, and
current source/binary hashes for the Python runner.

Run D1 separately (the output directory must be new):

```bash
python3 benchmarks/two_electron_engine/scripts/native_lane_ablation_55.py \
    --binary build-experiments-avx2/rys_native_lane_ablation \
    --expected-lanes 4 --nprim 3 --warmup 2 --repeats 7 \
    --out-dir benchmarks/two_electron_engine/results/native-avx2
```

For AVX-512, build with `avx512`, use the binary in `build-experiments-avx512`,
set `--expected-lanes 8`, and choose a separate results directory.
New experiment targets belong in `benchmarks/two_electron_engine/CMakeLists.txt`
under `TLNO_BUILD_EXPERIMENTS`; they use this same build script. D1 is currently
registered; `rys_route_ablation.cpp` still needs its namespace/API port completed.
