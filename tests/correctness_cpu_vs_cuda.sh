#!/usr/bin/env bash
# Cross-platform CORRECTNESS check (not timing): does robosample sample the SAME
# Boltzmann distribution on the CUDA build as on the CPU build, despite CUDA's
# non-deterministic force reductions? Runs the butane torsion benchmark (seed 42,
# well-sampled) on: CPU build, then CUDA build twice with the SAME seed.
#   |cudaA - cudaB|  = CUDA run-to-run non-determinism magnitude
#   |cuda  - cpu|    = platform distributional difference
#   both vs OpenMM oracle (~0.80 anti) = absolute correctness
# A stochastic sampler is never bit-identical; the pass criterion is statistical
# agreement (all within sampling error of each other and the oracle).
set -u
cd /home/alexb/Robosample_disasm
PKG=python/robosample/robo_bindings.cpython-312-x86_64-linux-gnu.so
CUDA_SO="$(pwd)/build/cuda-release/robo_bindings.cpython-312-x86_64-linux-gnu.so"
CPU_SO="$(pwd)/build/cpu-release/robo_bindings.cpython-312-x86_64-linux-gnu.so"

run() { env -u LD_LIBRARY_PATH python3 tests/robosample_butane_run.py 2>&1 \
        | grep -E "robo_bindings.so|ANTI fraction|gauche split|VERDICT|frames:"; }

echo "===== [1/3] CPU build (deterministic reference) ====="
ln -sf "$CPU_SO" "$PKG"; run
echo; echo "===== [2/3] CUDA build — run A (seed 42) ====="
ln -sf "$CUDA_SO" "$PKG"; run
echo; echo "===== [3/3] CUDA build — run B (seed 42, same seed => shows non-determinism) ====="
run
ln -sf "$CUDA_SO" "$PKG"   # restore default
echo; echo "===== restored default CUDA symlink ====="
