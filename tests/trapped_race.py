"""Audit-compliant race on the TRAPPED transition: the alanine alphaL crossing (phi=0),
which plain MD barely samples (L-alanine's phi>0 region is kinetically hard). This is
robosample's actual win condition. Metric = phi=0 CROSSINGS per WALL-SECOND (effective
sampling rate of the trapped mode), not raw frames. robosample (mixed, FIXED Cartesian
HMC, CUDA kinematics ON) vs OpenMM MD; multiple seeds; from the beta basin.

Run with ROBO_CUDA_KINEMATICS=1 in env. GENERIC (ala, OBC2)."""
import os, sys, time, tempfile
import numpy as np, parmed as pmd, mdtraj as md
import openmm as mm, openmm.app as app, openmm.unit as u
import robosample
from robosample import robo_bindings as rb

R = "/home/alexb/Robosample_disasm"; PRM = R+"/examples/ala-dipeptide.prmtop"
RST = R+"/examples/ala-dipeptide-c7ax.rst7"  # START IN C7ax/alphaL (phi~+60): race the ESCAPE across phi=0
SC = "/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad"
T = 300.0; TOP = md.load_prmtop(PRM); SEEDS = [1, 2]
N_TORS = 4          # torsional worlds per Gibbs sweep (paper 10:1; trimmed for wall-clock)
EQUIL = 1           # minimal equilibration so production begins still inside C7ax
PROD = 120          # production rounds (frames scored for phi=0 crossings)
TORS_MDSTEPS = 500  # torsional move length; escape barrier from C7ax is lower than forward 8 kcal/mol
TORS_DT = 0.015
CART_MDSTEPS = 40; CART_DT = 0.0015
# force-evals per method = one per MD step integrated:
ROBO_FEVALS = (EQUIL + PROD) * (N_TORS * TORS_MDSTEPS + CART_MDSTEPS)


def phi_of(xyz):  # residue-1 phi in degrees
    return np.degrees(md.compute_phi(md.Trajectory(xyz, TOP))[1][:, 0])


def crossings(phi_deg, dead=20.0):
    """phi=0 crossings with a deadband: count transitions phi<-dead <-> phi>+dead."""
    sign = np.zeros(len(phi_deg), int)
    sign[phi_deg > dead] = 1; sign[phi_deg < -dead] = -1
    s = sign[sign != 0]
    return int(np.sum(np.abs(np.diff(s)) == 2)) if len(s) > 1 else 0


def aL_frac(phi_deg):  # left-handed = phi>0
    return float(np.mean(phi_deg > 0))


def robo(seed):
    outdir = tempfile.mkdtemp(prefix="trap_", dir=SC); os.chdir(outdir)
    ctx = robosample.Context("trap", seed, robosample.AmberDihedralClassifier())
    ctx.load_amber(PRM, RST, use_gbsa_obc2=True); ctx.set_enforce_periodic_box(False)
    g = ctx.prmtop_to_global_index; st = pmd.load_file(PRM); pairs = []
    for r in st.residues:
        nm = {a.name: a.idx for a in r.atoms}
        if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]), int(g[nm["CA"]])))
        if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
    sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
    # N_TORS torsional : 1 Cartesian (paper 10:1 ratio, trimmed), LONG barrier-crossing moves, FIXED Cartesian
    for _ in range(N_TORS):
        ctx.add_robotic_world(sele).add_sampler(timeStep=TORS_DT, mdSteps=TORS_MDSTEPS,
            acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
    ctx.add_cartesian_world().add_sampler(timeStep=CART_DT, mdSteps=CART_MDSTEPS,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=False)
    ctx.initialize([T])
    t0 = time.perf_counter(); ctx.run_rex(EQUIL, PROD, 1, False); dt = time.perf_counter()-t0
    phi = phi_of(md.load(os.path.join(outdir, "trap.0.dcd"), top=PRM).xyz)
    return dt, crossings(phi), aL_frac(phi), ROBO_FEVALS


def openmm(seed, budget_s):
    system = app.AmberPrmtopFile(PRM).createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(T*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(app.AmberInpcrdFile(RST).positions); ctx.setVelocitiesToTemperature(T*u.kelvin, seed)
    t0 = time.perf_counter(); frames = []; nsteps = 0
    while time.perf_counter()-t0 < budget_s:
        integ.step(250); nsteps += 250  # 0.5 ps between samples; step in C, no python throttle
        frames.append(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer))
    dt = time.perf_counter()-t0
    phi = phi_of(np.array(frames))  # batch phi at the end
    return dt, crossings(phi), aL_frac(phi), nsteps  # nsteps = force-evals


def main():
    print(f"ROBO_CUDA_KINEMATICS={os.environ.get('ROBO_CUDA_KINEMATICS','unset')}")
    print(f"\nTRAPPED-MODE RACE: start in C7ax/alphaL (phi~+60), race the ESCAPE across phi=0.")
    print(f"metric = phi=0 crossings per wall-second AND per force-evaluation. aL_frac<1 => escaped.\n")
    rr, oo = [], []
    for s in SEEDS:
        dt, cx, aL, fe = robo(s)
        print(f"  robosample seed{s}: wall={dt:.0f}s  crossings={cx}  aL={aL:.3f}  fevals={fe:.2e}"
              f"  -> {cx/dt:.3f} cross/s  {1e6*cx/fe:.2f} cross/Mfeval", flush=True)
        rr.append((cx, dt, aL, cx/dt, 1e6*cx/fe))
        dto, cxo, aLo, feo = openmm(s, dt)  # match wall-clock
        print(f"  openmm     seed{s}: wall={dto:.0f}s  crossings={cxo}  aL={aLo:.3f}  fevals={feo:.2e}"
              f"  -> {cxo/dto:.3f} cross/s  {1e6*cxo/feo:.2f} cross/Mfeval", flush=True)
        oo.append((cxo, dto, aLo, cxo/dto, 1e6*cxo/feo))
    rcs, ocs = np.mean([x[3] for x in rr]), np.mean([x[3] for x in oo])
    rcf, ocf = np.mean([x[4] for x in rr]), np.mean([x[4] for x in oo])
    print(f"\n=== VERDICT (mean over {len(SEEDS)} seeds) ===")
    print(f"  robosample: {rcs:.3f} cross/wall-s   {rcf:.2f} cross/Mfeval   (aL {np.mean([x[2] for x in rr]):.3f})")
    print(f"  openmm    : {ocs:.3f} cross/wall-s   {ocf:.2f} cross/Mfeval   (aL {np.mean([x[2] for x in oo]):.3f})")
    def cmp(r, o, unit):
        if r > o and o > 0: print(f"  -> robosample {r/o:.1f}x faster per {unit}")
        elif o > r and r > 0: print(f"  -> OpenMM {o/r:.1f}x faster per {unit}")
        elif r > 0 and o == 0: print(f"  -> robosample crosses, OpenMM ZERO per {unit} (robosample wins)")
        elif o > 0 and r == 0: print(f"  -> OpenMM crosses, robosample ZERO per {unit} (robosample loses)")
        else: print(f"  -> both ZERO per {unit} (inconclusive)")
    cmp(rcs, ocs, "wall-second"); cmp(rcf, ocf, "force-evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
