"""Timed torsional robosample run (ala) to A/B the fused CUDA-kinematics path.
Run twice: ROBO_CUDA_KINEMATICS=0 vs =1 (separate processes; the OpenMMContext is a
singleton so the env is read once per process). Prints ms/round + aR (correctness)."""
import os, sys, time, tempfile
import numpy as np, parmed as pmd, mdtraj as md
import robosample
from robosample import robo_bindings as rb
R = "/home/alexb/Robosample_disasm"; PRM = R+"/examples/ala-dipeptide.prmtop"; RST = R+"/examples/ala-dipeptide.rst7"
SC = "/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
os.chdir(tempfile.mkdtemp(prefix="tr_", dir=SC))
ctx = robosample.Context("tr", 42, robosample.AmberDihedralClassifier())
ctx.load_amber(PRM, RST, use_gbsa_obc2=True); ctx.set_enforce_periodic_box(False)
g = ctx.prmtop_to_global_index; st = pmd.load_file(PRM); pairs = []
for r in st.residues:
    nm = {a.name: a.idx for a in r.atoms}
    if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]), int(g[nm["CA"]])))
    if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))
sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
ctx.add_robotic_world(sele).add_sampler(timeStep=0.02, mdSteps=200,
    acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
ctx.initialize([300.0])
t0 = time.perf_counter(); ctx.run_rex(30, ROUNDS, 1, False); dt = time.perf_counter()-t0
tr = md.load("tr.0.dcd", top=PRM); phi = np.degrees(md.compute_phi(tr)[1][:, 0]); psi = np.degrees(md.compute_psi(tr)[1][:, 0])
aR = float(np.mean((phi < 0) & (psi > -100) & (psi < 50)))
print(f"CUDA_KIN={os.environ.get('ROBO_CUDA_KINEMATICS','unset')}  {ROUNDS} rounds  "
      f"wall={dt:.1f}s  ms/round={1000*dt/ROUNDS:.1f}  aR={aR:.3f}")
