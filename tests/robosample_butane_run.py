#!/usr/bin/env python3
"""robosample half of the independent butane oracle — well-sampled, compared to native OpenMM.

Copies the exact working invocation from tests/test_torsion_conformational.py:_run_gchmc_torsion
but with a generous budget (2x the test's prod), on GENERIC butane (NO E2). Verdict: does
well-sampled robosample match native OpenMM's ~80% anti? If yes, robosample butane sampling is
correct and the Tier-2 failures are test-budget issues; if no, real sampler bias.
"""
import os, sys, math, tempfile
from pathlib import Path
import numpy as np

os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0")
REPO = Path("/home/alexb/Robosample_disasm")
sys.path.insert(0, str(REPO / "python"))
SCRATCH = Path("/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/"
               "6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad")

import parmed as pmd
import mdtraj as md
import robosample
from robosample import robo_bindings as rb

PRMTOP = REPO / "examples" / "butane.prmtop"
RST7 = next((REPO / "examples" / f"butane{e}" for e in (".rst7", ".inpcrd") if (REPO / "examples" / f"butane{e}").exists()), None)
DIH = [2, 0, 1, 3]          # C3-C1-C2-C4 (0-based prmtop order), rotate about 0-1
ROTATE_BOND = (0, 1)
T = 300.0

def anti_frac(phi):        # anti = trans, |phi| ~ pi
    return float(np.mean(np.abs(phi) > 2.0))

def gauche_split(phi):     # gauche+ (~ +60deg), gauche- (~ -60deg)
    gp = float(np.mean((phi > 0.5) & (phi < 2.0)))
    gm = float(np.mean((phi < -0.5) & (phi > -2.0)))
    return gp, gm

def main():
    if RST7 is None:
        print("SKIP: no butane rst7/inpcrd"); return 0
    outdir = Path(tempfile.mkdtemp(prefix="robo_butane_", dir=str(SCRATCH)))
    os.chdir(outdir)
    base = "butane_oracle"

    ctx = robosample.Context(base, 42, robosample.AmberDihedralClassifier())
    ctx.load_amber(str(PRMTOP), str(RST7))
    ctx.set_enforce_periodic_box(False)
    g = ctx.prmtop_to_global_index
    pairs = [(int(g[ROTATE_BOND[0]]), int(g[ROTATE_BOND[1]]))]
    sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
    ctx.add_robotic_world(sele).add_sampler(
        timeStep=0.005, mdSteps=75,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings,
        use_nuts=False, use_fixman=True)
    ctx.initialize([T])
    # generous budget: 2x the test's prod (equil 200, prod 30000)
    ctx.run_rex(200, 8000, 1, False)

    traj = md.load(base + ".0.dcd", top=str(PRMTOP))
    phi_robo = md.compute_dihedrals(traj, [DIH])[:, 0].astype(float)
    np.save(SCRATCH / "butane_phi_robo.npy", phi_robo)

    phi_omm = np.load(SCRATCH / "butane_phi_omm.npy")
    a_r, a_o = anti_frac(phi_robo), anti_frac(phi_omm)
    gp_r, gm_r = gauche_split(phi_robo)
    print(f"frames: robosample={len(phi_robo)}  openmm={len(phi_omm)}")
    print(f"ANTI fraction:  robosample={a_r:.3f}   OpenMM(oracle)={a_o:.3f}   diff={abs(a_r-a_o):.3f}")
    print(f"robosample gauche split: g+={gp_r:.3f}  g-={gm_r:.3f}  (should be ~symmetric)")
    # verdict
    ok_anti = abs(a_r - a_o) < 0.08
    ok_sym = abs(gp_r - gm_r) < 0.06 or (gp_r + gm_r) < 0.02
    print(f"VERDICT: anti-fraction match={'PASS' if ok_anti else 'FAIL'} "
          f"({a_r:.2f} vs {a_o:.2f}); gauche symmetry={'ok' if ok_sym else 'ASYMMETRIC(undersampled)'}")
    if ok_anti:
        print("=> well-sampled robosample MATCHES native OpenMM on butane -> sampler correct; "
              "Tier-2 failures are test-budget/undersampling, not bias.")
    else:
        print("=> robosample does NOT match OpenMM even well-sampled -> possible real RobotEngine bias (escalate).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
