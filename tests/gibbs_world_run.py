#!/usr/bin/env python3
"""Run robosample's ACTUAL multi-world MIXED Gibbs protocol on alanine-dipeptide
and print the Ramachandran basin populations for whichever build is linked.

This is the sampler that matters: a fully-flexible Cartesian world (Fixman off)
*mixed* with a torsional CDHMC world (Fixman on) — Gibbs sampling. Used by the
CPU-vs-CUDA harness (gibbs_cpu_vs_cuda.sh) to check the Gibbs sampler samples the
same distribution on both platforms. GENERIC (ala-dipeptide). Seed fixed for the
CUDA run-to-run determinism check.

Usage: python3 tests/gibbs_world_run.py [rounds]
"""
import os, sys, tempfile
from pathlib import Path
import numpy as np

os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0")
REPO = Path("/home/alexb/Robosample_disasm")
sys.path.insert(0, str(REPO / "python"))
SCRATCH = Path("/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/"
               "6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad")
PRMTOP = REPO / "examples" / "ala-dipeptide.prmtop"
RST7 = next((REPO / "examples" / f"ala-dipeptide{e}"
             for e in (".rst7", ".inpcrd") if (REPO / "examples" / f"ala-dipeptide{e}").exists()), None)
T = 300.0
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = 42

import parmed as pmd
import mdtraj as md
import robosample
from robosample import robo_bindings as rb


def basins(phi, psi):
    d, e = np.degrees(phi), np.degrees(psi)
    return dict(alphaR=float(np.mean((d < 0) & (e > -100) & (e < 50))),
                beta_C7=float(np.mean((d < 0) & ((e > 50) | (e < -100)))),
                alphaL=float(np.mean((d > 0) & (e > -50) & (e < 100))))


def main():
    print(f"robo_bindings.so -> {os.path.realpath(rb.__file__)}")
    outdir = Path(tempfile.mkdtemp(prefix="gibbs_", dir=str(SCRATCH))); os.chdir(outdir)
    base = "gibbs_ala"
    ctx = robosample.Context(base, SEED, robosample.AmberDihedralClassifier())
    ctx.load_amber(str(PRMTOP), str(RST7), use_gbsa_obc2=True)
    ctx.set_enforce_periodic_box(False)
    g = ctx.prmtop_to_global_index
    struct = pmd.load_file(str(PRMTOP)); pairs = []
    for r in struct.residues:
        nm = {a.name: a.idx for a in r.atoms}
        if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]),  int(g[nm["CA"]])))
        if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
    sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
    # === the MIXED Gibbs protocol: two worlds alternated by run_rex ===
    ctx.add_cartesian_world().add_sampler(                       # fully-flexible (Fixman off)
        timeStep=0.001, mdSteps=20, acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings,
        use_nuts=False, use_fixman=False)
    ctx.add_robotic_world(sele).add_sampler(                     # torsional CDHMC (Fixman on)
        timeStep=0.004, mdSteps=50, acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings,
        use_nuts=False, use_fixman=True)
    ctx.initialize([T])
    ctx.run_rex(300, ROUNDS, 1, False)
    traj = md.load(base + ".0.dcd", top=str(PRMTOP))
    phi = md.compute_phi(traj)[1][:, 0]; psi = md.compute_psi(traj)[1][:, 0]
    b = basins(phi, psi)
    print(f"MIXED-GIBBS basins (2 worlds: Cartesian + torsional): "
          f"alphaR={b['alphaR']:.3f} beta_C7={b['beta_C7']:.3f} alphaL={b['alphaL']:.3f}  "
          f"({len(phi)} frames, {ROUNDS} rounds, seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
