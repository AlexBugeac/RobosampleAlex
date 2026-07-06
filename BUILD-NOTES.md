# Build ergonomics — field notes & fixes

Written 2026-07-06 while building `disasm` from a fresh worktree. Every item below is a
**real failure I hit in order**, with the workaround I used and the fix that would stop the
next person hitting it. A newcomer following `CMakePresets.json` as documented does **not**
get a working build today — it takes three undocumented overrides. Priority = how many users
it blocks × how surprising it is.

## The three configure-blockers (a fresh `cmake --preset cuda-tests` fails on each in turn)

### 1. CUDA compiler not found — `"The CUDA compiler identification is unknown"`  [BLOCKER]
`cmake --preset cuda-tests` aborts at `enable_language(CUDA)` because neither the preset nor
the environment points CMake at `nvcc`.
- **Workaround used:** `CUDACXX=/opt/cuda/bin/nvcc cmake --preset cuda-tests -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`.
- **Fix:** in the `cuda` base preset, set `"CMAKE_CUDA_COMPILER": "$env{CUDACXX}"` with a
  sensible fallback, or resolve it from `CUDAToolkit_ROOT`/`find_program(nvcc)` *before*
  `enable_language(CUDA)` and set it. One line in the preset removes the first wall.

### 2. BLAS not found — `"Could NOT find BLAS (missing: BLAS_LIBRARIES)"`  [BLOCKER]
`set(BLA_VENDOR OpenBLAS)` + `find_package(BLAS REQUIRED)` fails inside a conda env because
conda ships `libopenblas.so.0` / `libopenblasp-r0.3.30.so` but **not** the bare dev symlink
`libopenblas.so` that FindBLAS(OpenBLAS) looks for.
- **Workaround used:** `-DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so.0 -DLAPACK_LIBRARIES=…`.
- **Fix (pick one):** (a) drop the hard `BLA_VENDOR OpenBLAS` and let FindBLAS locate generic
  BLAS; (b) add a fallback: if `BLAS_LIBRARIES` is empty and `$CONDA_PREFIX/lib/libopenblas.so.0`
  exists, use it; or (c) add a `find_library(OpenBLAS NAMES openblas libopenblas.so.0)` shim.
  This same wall exists on the `refactor` branch too — it's a recurring conda-env footgun.

### 3. `cuda.h: No such file or directory`  [BLOCKER — this one is a genuine CMake bug]
`src/OpenMMContext.cpp` → `openmm/platforms/cuda/include/CudaContext.h` → `#include <cuda.h>`.
The compile line has ~35 `-I` OpenMM include dirs and the conda
`targets/x86_64-linux/include`, but **not** the system CUDA include holding `cuda.h`
(`/opt/cuda/include`). The conda cudatoolkit include ships the runtime headers, not the
driver `cuda.h`.
- **Workaround used:** `CPATH=/opt/cuda/include ninja` (compiler-level; bypasses CMake).
  Note `-DCMAKE_CXX_FLAGS="-I/opt/cuda/include"` **did not survive `cmake --preset`** — the
  preset's cache handling dropped it, which is itself a surprising trap.
- **Fix:** the CUDA-platform object target must carry `${CUDAToolkit_INCLUDE_DIRS}` in its
  `target_include_directories`. `find_package(CUDAToolkit)` already runs — just consume its
  include dirs on the OpenMM-CUDA sources. This is the right fix; `CPATH` is a band-aid.

## Secondary friction

### 4. `ccache` is `REQUIRED` for a pure speed optimization  [avoidable hard-fail]
`find_program(CCACHE ccache REQUIRED)` hard-fails configure if ccache is absent — but ccache
only *speeds up* rebuilds. **Fix:** `find_program(CCACHE ccache)`; set the compiler launcher
only `if(CCACHE)`, else warn. Never block a first build on a cache tool.

### 5. `-march=native` / `CUDA_ARCHITECTURES=native` host-lock the binary  [portability trap]
Fine on a single box; but a `.so` built on the RTX 5080 host emits AVX-512/sm_120 that faults
on an older lab machine. **Fix:** ship a `*-portable` preset with `-march=x86-64-v3` and an
explicit arch list (`75;86;89;120`); keep `native` as an opt-in "local-fast" preset.

### 6. Conda `LD_LIBRARY_PATH` shadows system libs during `ninja`  [documented-elsewhere gotcha]
Building needs `env -u LD_LIBRARY_PATH` (conda's `libreadline`/`libstdc++` shadow the system
ones and break `ninja`/`/bin/sh`). Undocumented in-tree. **Fix:** document it, or set a clean
build environment in the preset, or fix the RPATH so it isn't needed.

### 7. No memory guardrail — a naive `ninja` can OOM a busy workstation  [operational]
A full `-jN` build of the OpenMM object tree peaks multiple GB/TU; on a machine also running
GPU sims it can invoke the OOM-killer against the *sims*. I ran under
`systemd-run --user --scope -p MemoryMax=3500M nice -j2`. **Fix:** document a memory-capped
build recipe (and/or a `--mem-cap` flag in a bootstrap script), and default the docs to a
conservative `-j`.

### 8. The `cuda-tests` preset silently enables ASan+UBSan  [expectation mismatch]
`-fsanitize=address,undefined -fno-sanitize-recover=all` is on by default in `cuda-tests`.
Great for catching real RobotEngine bugs (keep it for CI), but it's heavier and can interact
badly with CUDA at runtime; a user just wanting to *run* the validation suite may not expect
an instrumented build. **Fix:** label it clearly and/or add a `cuda-tests-fast` (no sanitizer)
sibling for plain running vs. bug-hunting.

## The single highest-leverage fix: a `bootstrap.sh`

All of 1–7 are tribal knowledge. One script encapsulating them turns "three false starts +
undocumented flags" into `./bootstrap.sh`:

```sh
#!/usr/bin/env bash
set -euo pipefail
# 0. preflight (a mini 'doctor')
: "${CONDA_PREFIX:?activate the robosample conda env first}"
command -v nvcc >/dev/null || { echo "nvcc not on PATH (need CUDA toolkit)"; exit 1; }
test -f /opt/cuda/include/cuda.h || echo "warn: cuda.h not at /opt/cuda/include; set CUDA_HOME"
command -v ccache >/dev/null || echo "note: ccache absent (slower rebuilds)"
# 1-3. the discovery overrides
export CUDACXX="${CUDACXX:-$(command -v nvcc)}"
export CPATH="${CUDA_HOME:-/opt/cuda}/include${CPATH:+:$CPATH}"
BLAS="$CONDA_PREFIX/lib/libopenblas.so.0"
# 6. clean env + 7. memory-capped build
env -u LD_LIBRARY_PATH cmake --preset "${1:-cuda-tests}" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-native}" \
    -DBLAS_LIBRARIES="$BLAS" -DLAPACK_LIBRARIES="$BLAS"
systemd-run --user --scope -p MemoryMax="${MEMCAP:-3500M}" \
    nice env -u LD_LIBRARY_PATH CPATH="$CPATH" ninja -C "build/${1:-cuda-tests}"
```

That one file would have saved the three false starts I just had. **Recommended order to
actually fix the tree:** #3 (cuda.h include — real bug) → #1 (CUDA compiler in preset) →
#2 (BLAS fallback) → #4 (ccache optional) → ship `bootstrap.sh` → #5/#8 (portable + fast
preset variants). Items #1–#4 are ~10 lines of CMake between them and remove every hard wall.
</content>

### 11. Undeclared `attrs` Python dependency (disasm)
`python/robosample/topology.py` does `from attr import dataclass` but `attrs` is not in `envs/*.yaml` or `pyproject`. Fresh env → `ModuleNotFoundError: No module named 'attr'`. Fix: add `attrs` to the conda env spec + pyproject deps.
