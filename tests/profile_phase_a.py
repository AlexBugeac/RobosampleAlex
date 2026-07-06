#!/usr/bin/env python3
"""accel-engine Phase-A coarse profiler: where does robosample's per-round time go?

Splits a robosample GCHMC round into (a) OpenMM force-server time and (b) everything
else (articulated-body dynamics + Fixman metric + accept/reject). Method: time the full
robosample run, then time an equivalent native-OpenMM force-eval loop for the same
system+step budget on the same platform; the difference approximates the dynamics/Fixman
overhead. Coarse but actionable — tells us whether to attack the round-trip (Pillar 3) or
the CPU-serial dynamics (Pillar 2) first. GENERIC (ala-dipeptide) — NO E2.
"""
import os, sys, time, tempfile
from pathlib import Path
os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0")
REPO = Path("/home/alexb/Robosample_disasm")
sys.path.insert(0, str(REPO / "python"))
import parmed as pmd
import robosample
from robosample import robo_bindings as rb

PRMTOP = REPO / "examples" / "ala-dipeptide.prmtop"
RST7 = next((REPO / "examples" / f"ala-dipeptide{e}" for e in (".rst7", ".inpcrd")
             if (REPO / "examples" / f"ala-dipeptide{e}").exists()), None)
T = 300.0
ROUNDS = 400
MD_STEPS = 50   # robotic-world sampler mdSteps (per round the force server is hit ~this many times)


def robosample_round_time():
    outdir = Path(tempfile.mkdtemp(prefix="robo_prof_", dir="/tmp")); os.chdir(outdir)
    ctx = robosample.Context("prof", 1, robosample.AmberDihedralClassifier())
    ctx.load_amber(str(PRMTOP), str(RST7), use_gbsa_obc2=True); ctx.set_enforce_periodic_box(False)
    g = ctx.prmtop_to_global_index
    struct = pmd.load_file(str(PRMTOP)); pairs = []
    for r in struct.residues:
        nm = {a.name: a.idx for a in r.atoms}
        if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]), int(g[nm["CA"]])))
        if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
    sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
    ctx.add_robotic_world(sele).add_sampler(timeStep=0.004, mdSteps=MD_STEPS,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
    ctx.initialize([T])
    t0 = time.perf_counter(); ctx.run_rex(20, ROUNDS, ROUNDS + 1, False)
    return (time.perf_counter() - t0) / ROUNDS


def openmm_force_time():
    """Native-OpenMM force-eval loop, same system/platform, ROUNDS*MD_STEPS force calls."""
    import openmm as mm, openmm.app as app, openmm.unit as u
    prm = app.AmberPrmtopFile(str(PRMTOP))
    system = prm.createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2, constraints=app.HBonds)
    integ = mm.VerletIntegrator(0.004 * u.picoseconds)
    ctx = mm.Context(system, integ)
    ctx.setPositions(app.AmberInpcrdFile(str(RST7)).positions)
    n = ROUNDS * MD_STEPS
    # warm up
    for _ in range(50): ctx.getState(getForces=True)
    t0 = time.perf_counter()
    for _ in range(n):
        integ.step(1); ctx.getState(getForces=True, getEnergy=True)
    return (time.perf_counter() - t0) / ROUNDS   # force-server time attributable per round


def main():
    if RST7 is None:
        print("SKIP: no ala rst7/inpcrd"); return 0
    print(f"=== Phase-A coarse profile: ala-dipeptide, {ROUNDS} rounds, mdSteps={MD_STEPS} ===")
    t_round = robosample_round_time()
    print(f"robosample full round:        {1000*t_round:.2f} ms/round")
    t_force = openmm_force_time()
    print(f"OpenMM force-server (~{MD_STEPS}/rnd): {1000*t_force:.2f} ms/round-equivalent")
    t_dyn = max(0.0, t_round - t_force)
    frac_f = 100 * t_force / t_round if t_round else 0
    print(f"est. dynamics+Fixman+accept:  {1000*t_dyn:.2f} ms/round  ({100-frac_f:.0f}%)")
    print(f"est. force-server fraction:   {frac_f:.0f}%")
    print("INTERPRETATION: force-dominated → attack the round-trip (Pillar 3: CPU platform / on-device loop). "
          "dynamics-dominated → attack CPU-serial ABA+Fixman (Pillar 2: batched-replica GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
