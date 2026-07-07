#!/usr/bin/env bash
# CORRECTNESS of robosample's ACTUAL multi-world MIXED Gibbs sampler across platforms.
# Runs the 2-world (Cartesian + torsional) Gibbs protocol on alanine-dipeptide on:
# CPU build, then CUDA build twice (same seed). Compares Ramachandran basin
# populations => does the Gibbs sampler sample the same distribution on CPU vs CUDA,
# and is CUDA run-to-run deterministic?
set -u
cd /home/alexb/Robosample_disasm
PKG=python/robosample/robo_bindings.cpython-312-x86_64-linux-gnu.so
CUDA_SO="$(pwd)/build/cuda-release/robo_bindings.cpython-312-x86_64-linux-gnu.so"
CPU_SO="$(pwd)/build/cpu-release/robo_bindings.cpython-312-x86_64-linux-gnu.so"
ROUNDS="${1:-5000}"

run() { env -u LD_LIBRARY_PATH python3 tests/gibbs_world_run.py "$ROUNDS" 2>&1 \
        | grep -E "robo_bindings.so|MIXED-GIBBS basins|Error|Traceback"; }

echo "===== [1/3] CPU build — 2-world MIXED Gibbs ====="
ln -sf "$CPU_SO" "$PKG"; run
echo; echo "===== [2/3] CUDA build — MIXED Gibbs run A (seed 42) ====="
ln -sf "$CUDA_SO" "$PKG"; run
echo; echo "===== [3/3] CUDA build — MIXED Gibbs run B (seed 42, same => non-determinism) ====="
run
ln -sf "$CUDA_SO" "$PKG"   # restore default
echo; echo "===== restored default CUDA symlink ====="
