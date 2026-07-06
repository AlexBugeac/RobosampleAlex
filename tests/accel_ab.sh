#!/bin/bash
# accel Phase-B: CPU-vs-CUDA robosample wall-clock A/B (GENERIC ala-dipeptide, NO E2).
# Swaps the python/robosample robo_bindings.so between the two builds and times each.
set -e
cd /home/alexb/Robosample_disasm
PKG=python/robosample/robo_bindings.cpython-312-x86_64-linux-gnu.so
CUDA_SO=$(pwd)/build/cuda-release/robo_bindings.cpython-312-x86_64-linux-gnu.so
CPU_SO=$(pwd)/build/cpu-release/robo_bindings.cpython-312-x86_64-linux-gnu.so
PROD=${1:-500}
run(){ env -u LD_LIBRARY_PATH PYTHONPATH=$(pwd)/python \
  OPENMM_CUDA_COMPILER=/opt/cuda/bin/nvcc CUDA_ROOT=/opt/cuda ASAN_OPTIONS=detect_leaks=0 \
  python3 tests/timed_run.py "$PROD" 2>&1 | grep -E "robo_bindings|TIMED|CUDA platform|Reference|CPU|Error"; }
echo "===== CUDA build ====="; ln -sf "$CUDA_SO" "$PKG"; run
echo "===== CPU build  ====="; ln -sf "$CPU_SO"  "$PKG"; run
ln -sf "$CUDA_SO" "$PKG"; echo "===== restored CUDA symlink (default) ====="
