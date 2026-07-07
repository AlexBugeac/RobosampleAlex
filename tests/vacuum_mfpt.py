"""VALIDATE the Robosample headline claim (Spiridon 2020 Table 4): Ramachandran-mixed
dynamics crosses the alanine alphaL / phi=0 barrier ~10x fewer MD-STEPS than fully-flexible.
This is robosample-vs-robosample (fully-flexible is the baseline, NOT OpenMM). Vacuum, so
alphaL is a genuine rare event (the paper's regime). Metric = integrator steps to first
phi=0 crossing, starting inside alphaL/C7ax. Paper (units of steps/200, alphaL row):
  fully-flexible ~491,  mixed-TD ~49  =>  ~10x.  We report raw steps and the ratio.

Paper-tuned params (2020 Table, optimized): flexible 108 x 1.87 fs; Rama torsional 11 x 44.73 fs.
Cycle = 1 fully-flexible world + 10 CDHMC torsional worlds (2017/2020 alanine protocol).
env: ROBO_CUDA_KINEMATICS=1"""
import os, sys, time, tempfile
import numpy as np, parmed as pmd, mdtraj as md
import robosample
from robosample import robo_bindings as rb

R = "/home/alexb/Robosample_disasm"; PRM = R+"/examples/ala-dipeptide.prmtop"
RST = R+"/examples/ala-dipeptide-c7ax.rst7"   # start INSIDE alphaL (phi~+60)
SC = "/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad"
TOP = md.load_prmtop(PRM); T = 300.0; SEEDS = [1, 2, 3]
FLEX_MD, FLEX_DT = 108, 0.00187        # fully-flexible world: 108 x 1.87 fs
TORS_MD, TORS_DT = 11, 0.04473         # Rama torsional world: 11 x 44.73 fs
N_TORS = 10                            # 10 CDHMC : 1 flexible (paper alanine cycle)
CAP_FLEX, CAP_MIXED = 4000, 600        # max rounds before censoring a crossing

def build(seed, mixed):
    outdir = tempfile.mkdtemp(prefix="mfpt_", dir=SC); os.chdir(outdir)
    ctx = robosample.Context("mfpt", seed, robosample.AmberDihedralClassifier())
    ctx.load_amber(PRM, RST, use_gbsa_obc2=False)   # VACUUM
    ctx.set_enforce_periodic_box(False)
    # fully-flexible world (always present; = baseline, and provides ergodicity in mixed)
    ctx.add_cartesian_world().add_sampler(timeStep=FLEX_DT, mdSteps=FLEX_MD,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=False)
    steps_per_round = FLEX_MD
    if mixed:
        g = ctx.prmtop_to_global_index; st = pmd.load_file(PRM); pairs = []
        for r in st.residues:
            nm = {a.name: a.idx for a in r.atoms}
            if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]), int(g[nm["CA"]])))
            if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
        sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)  # Ramachandran (phi,psi only)
        for _ in range(N_TORS):
            ctx.add_robotic_world(sele).add_sampler(timeStep=TORS_DT, mdSteps=TORS_MD,
                acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
        steps_per_round += N_TORS * TORS_MD
    ctx.initialize([T])
    return ctx, outdir, steps_per_round

def first_cross_round(dcd):
    """first production frame where phi leaves alphaL and crosses to phi<-20 (the phi=0 barrier)."""
    phi = np.degrees(md.compute_phi(md.load(dcd, top=PRM))[1][:, 0])
    below = np.where(phi < -20)[0]
    return (int(below[0]) if len(below) else None), phi

def run(seed, mixed, cap):
    ctx, outdir, spr = build(seed, mixed)
    t0 = time.perf_counter(); ctx.run_rex(1, cap, 1, False); wall = time.perf_counter()-t0
    fr, phi = first_cross_round(os.path.join(outdir, "mfpt.0.dcd"))
    if fr is None:
        return None, cap*spr, wall, phi, spr  # censored
    return fr, fr*spr, wall, phi, spr

def main():
    print(f"ROBO_CUDA_KINEMATICS={os.environ.get('ROBO_CUDA_KINEMATICS','unset')}  VACUUM  start=alphaL(phi~+60)")
    print(f"metric = integrator steps to first phi=0 crossing (alphaL escape). paper/200: flex~491 mixed~49\n")
    res = {}
    for label, mixed, cap in [("fully-flexible", False, CAP_FLEX), ("Rama-mixed", True, CAP_MIXED)]:
        steps_list = []
        for s in SEEDS:
            fr, steps, wall, phi, spr = run(s, mixed, cap)
            cen = " (CENSORED, no crossing)" if fr is None else ""
            print(f"  {label:14s} seed{s}: steps_per_round={spr}  first_cross_round={fr}  "
                  f"MD_steps={steps}  (/200={steps/200:.1f})  wall={wall:.0f}s{cen}", flush=True)
            steps_list.append(steps)
        res[label] = np.array(steps_list, float)
        print(f"  -> {label}: MD-steps mean={res[label].mean():.0f}  /200={res[label].mean()/200:.1f}  "
              f"(paper/200: {'491' if not mixed else '49'})\n", flush=True)
    ratio = res["fully-flexible"].mean() / res["Rama-mixed"].mean()
    print(f"=== VALIDATION VERDICT ===")
    print(f"  flexible / mixed MD-step ratio = {ratio:.1f}x   (paper: ~10x)")
    print(f"  {'PASS: order-of-magnitude speedup reproduced' if ratio >= 5 else 'below paper ~10x — see notes'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
