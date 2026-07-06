#!/usr/bin/env python3
"""Minimal timed robosample run for the accel CPU-vs-CUDA A/B (accel-engine Phase B).

Runs a small GENERIC system (ala-dipeptide, torsional world, fixed budget) and prints the
production wall-clock + which OpenMM platform the linked robo_bindings.so uses. Run it twice
via accel_ab.sh (CUDA symlink, then CPU .so) to test the research claim that for small systems
the per-step CPU<->GPU round-trip makes CPU faster than CUDA. GENERIC — NO E2.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0")
REPO = Path("/home/alexb/Robosample_disasm")
sys.path.insert(0, str(REPO / "python"))
import tempfile, robosample
from robosample import robo_bindings as rb

PRMTOP = REPO / "examples" / "ala-dipeptide.prmtop"
RST7 = next((REPO/"examples"/f"ala-dipeptide{e}" for e in (".rst7",".inpcrd") if (REPO/"examples"/f"ala-dipeptide{e}").exists()), None)
PROD = int(sys.argv[1]) if len(sys.argv) > 1 else 500

def main():
    os.chdir(tempfile.mkdtemp(prefix="robo_timed_", dir="/tmp"))
    print(f"robo_bindings.so -> {os.path.realpath(rb.__file__)}")
    ctx = robosample.Context("timed", 1, robosample.AmberDihedralClassifier())
    ctx.load_amber(str(PRMTOP), str(RST7)); ctx.set_enforce_periodic_box(False)
    df = ctx.standard_dihedral_bonds
    bonds = df[df["dihedral_type"].isin(["phi","psi"])] if "dihedral_type" in df.columns else df
    sele = ctx.build_flexibilities(bonds, rb.JointType.Torsion, False)
    ctx.add_cartesian_world().add_sampler(timeStep=0.001, mdSteps=20,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=False)
    ctx.add_robotic_world(sele).add_sampler(timeStep=0.004, mdSteps=50,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
    ctx.initialize([300.0])
    t0 = time.perf_counter()
    ctx.run_rex(20, PROD, PROD + 1, False)   # write_freq > prod: no DCD writes, pure sampling wall-clock
    dt = time.perf_counter() - t0
    print(f"TIMED: prod_rounds={PROD}  wall={dt:.2f}s  ms/round={1000*dt/PROD:.2f}")

if __name__ == "__main__":
    raise SystemExit(main())
