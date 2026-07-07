"""Rule out the tuning artifact: can robosample escape C7ax/alphaL at 300 K if given its
BEST fair shot -- long torsional moves (mdSteps up to session's ~1000-1500 crossing threshold)
+ a much beefier Cartesian world to supply the bond-angle give the frozen-torsion path lacks?
Same start (C7ax, phi~+60), same 300 K, no REX/bias. If phi still never crosses 0 -> the
loss to OpenMM (96 vs 0) is fundamental-here, not tuning. If it crosses -> tuning.

argv: [TORS_MDSTEPS=1500] [CART_MDSTEPS=400] [PROD=80]   env: ROBO_CUDA_KINEMATICS=1"""
import os, sys, time, tempfile
import numpy as np, parmed as pmd, mdtraj as md
import robosample
from robosample import robo_bindings as rb

R = "/home/alexb/Robosample_disasm"; PRM = R+"/examples/ala-dipeptide.prmtop"
RST = R+"/examples/ala-dipeptide-c7ax.rst7"
SC = "/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad"
TOP = md.load_prmtop(PRM); T = 300.0
TMD = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
CMD = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PROD = int(sys.argv[3]) if len(sys.argv) > 3 else 80
N_TORS = 4; EQUIL = 1

def crossings(phi, dead=20.0):
    s = np.zeros(len(phi), int); s[phi > dead] = 1; s[phi < -dead] = -1; s = s[s != 0]
    return int(np.sum(np.abs(np.diff(s)) == 2)) if len(s) > 1 else 0

print(f"CUDA_KIN={os.environ.get('ROBO_CUDA_KINEMATICS','unset')}  TORS_MDSTEPS={TMD}  CART_MDSTEPS={CMD}  PROD={PROD}")
outdir = tempfile.mkdtemp(prefix="esc_", dir=SC); os.chdir(outdir)
ctx = robosample.Context("esc", 1, robosample.AmberDihedralClassifier())
ctx.load_amber(PRM, RST, use_gbsa_obc2=True); ctx.set_enforce_periodic_box(False)
g = ctx.prmtop_to_global_index; st = pmd.load_file(PRM); pairs = []
for r in st.residues:
    nm = {a.name: a.idx for a in r.atoms}
    if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]), int(g[nm["CA"]])))
    if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
for _ in range(N_TORS):
    ctx.add_robotic_world(sele).add_sampler(timeStep=0.015, mdSteps=TMD,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
# BEEFY cartesian world: many steps at a real MD dt -> supplies angle relaxation for the TS
ctx.add_cartesian_world().add_sampler(timeStep=0.002, mdSteps=CMD,
    acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=False)
ctx.initialize([T])
t0 = time.perf_counter(); ctx.run_rex(EQUIL, PROD, 1, False); dt = time.perf_counter()-t0
phi = np.degrees(md.compute_phi(md.load("esc.0.dcd", top=PRM))[1][:, 0])
cx = crossings(phi); aL = float(np.mean(phi > 0))
print(f"wall={dt:.0f}s  frames={len(phi)}  phi min={phi.min():.0f} max={phi.max():.0f}  "
      f"crossings={cx}  aL_frac={aL:.3f}  escaped={'YES' if phi.min()<-20 else 'NO'}")
print("VERDICT: " + ("crossed -> earlier 0 was a TUNING artifact"
      if cx > 0 else "STILL 0 with long moves + beefy Cartesian -> loss to MD is fundamental-here, not tuning"))
