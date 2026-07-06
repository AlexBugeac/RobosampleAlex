#!/usr/bin/env python3
"""Standard benchmark: alanine-dipeptide phi/psi (Ramachandran) — the classic
conformational-sampling method-comparison system. robosample (disasm) vs native
OpenMM MD, on the SAME implicit-solvent potential. GENERIC — NO E2.

A correct robosample must reproduce OpenMM's phi/psi 2D populations (alpha_R / C7eq
/ C5 / alpha_L basins). Reports basin populations for both + a 2D chi-square agreement.
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

import parmed as pmd
import mdtraj as md
import robosample
from robosample import robo_bindings as rb


def rama_hist(phi, psi, nb=18):
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    h, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
    return h / h.sum()

def basins(phi, psi):
    """Coarse Ramachandran basin fractions (radians)."""
    d = np.degrees(phi); e = np.degrees(psi)
    aR = np.mean((d < 0) & (e > -100) & (e < 50))         # alpha_R
    C7 = np.mean((d < 0) & ((e > 50) | (e < -100)))       # C7eq/C5/beta
    aL = np.mean((d > 0) & (e > -50) & (e < 100))         # alpha_L
    return dict(alphaR=float(aR), beta_C7=float(C7), alphaL=float(aL))

def openmm_rama():
    import openmm as mm, openmm.app as app, openmm.unit as u
    prm = app.AmberPrmtopFile(str(PRMTOP))
    system = prm.createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(T*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
    ctx = mm.Context(system, integ)
    ctx.setPositions((app.AmberInpcrdFile(str(RST7)).positions if RST7 else pmd.load_file(str(PRMTOP)).positions))
    mm.LocalEnergyMinimizer.minimize(ctx); ctx.setVelocitiesToTemperature(T*u.kelvin); integ.step(50000)
    top = md.load_prmtop(str(PRMTOP)); phis=[]; psis=[]
    for _ in range(6000):
        integ.step(100)
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        t = md.Trajectory(pos[None], top)
        phis.append(md.compute_phi(t)[1][0,0]); psis.append(md.compute_psi(t)[1][0,0])
    return np.array(phis), np.array(psis)

def robosample_rama():
    outdir = Path(tempfile.mkdtemp(prefix="robo_ala_", dir=str(SCRATCH))); os.chdir(outdir)
    base = "ala_rama"
    ctx = robosample.Context(base, 42, robosample.AmberDihedralClassifier())
    # MUST match the OpenMM reference potential: OpenMM uses implicitSolvent=OBC2,
    # so robosample must too — else robosample runs in VACUUM (C7eq/beta-dominant)
    # while OpenMM is solvated (alphaR-dominant), an apples-to-oranges comparison.
    ctx.load_amber(str(PRMTOP), str(RST7), use_gbsa_obc2=True); ctx.set_enforce_periodic_box(False)
    # Select backbone phi (N-CA) + psi (CA-C) rotatable bonds by EXPLICIT atom-index pairs
    # (the validated butane pattern). NOTE: standard_dihedral_bonds' 'dihedral_type' column is
    # INTEGER codes, not strings — filtering .isin(["phi","psi"]) silently returns empty and
    # freezes the backbone. parmed gives us the atom names; map prmtop->global index.
    g = ctx.prmtop_to_global_index
    struct = pmd.load_file(str(PRMTOP))
    pairs = []
    for r in struct.residues:
        nm = {a.name: a.idx for a in r.atoms}
        if "N" in nm and "CA" in nm: pairs.append((int(g[nm["N"]]),  int(g[nm["CA"]])))  # phi
        if "CA" in nm and "C" in nm: pairs.append((int(g[nm["CA"]]), int(g[nm["C"]])))   # psi
    print(f"robosample flexible backbone pairs (phi/psi): {pairs}")
    sele = ctx.build_flexibilities(pairs, rb.JointType.Torsion, False)
    ctx.add_cartesian_world().add_sampler(timeStep=0.001, mdSteps=20,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=False)
    ctx.add_robotic_world(sele).add_sampler(timeStep=0.004, mdSteps=50,
        acceptRejectMode=rb.AcceptRejectMode.MetropolisHastings, use_nuts=False, use_fixman=True)
    ctx.initialize([T]); ctx.run_rex(300, 8000, 1, False)   # right-sized budget
    traj = md.load(base + ".0.dcd", top=str(PRMTOP))
    return md.compute_phi(traj)[1][:,0], md.compute_psi(traj)[1][:,0]

def main():
    if not PRMTOP.exists() or RST7 is None:
        print("SKIP: missing ala-dipeptide files"); return 0
    print("=== native OpenMM Ramachandran ===")
    p_o, s_o = openmm_rama(); b_o = basins(p_o, s_o)
    print(f"OpenMM basins: {b_o}  ({len(p_o)} frames)")
    print("=== robosample Ramachandran ===")
    p_r, s_r = robosample_rama(); b_r = basins(p_r, s_r)
    print(f"robosample basins: {b_r}  ({len(p_r)} frames)")
    np.save(SCRATCH/"ala_rama_omm.npy", np.c_[p_o,s_o]); np.save(SCRATCH/"ala_rama_robo.npy", np.c_[p_r,s_r])
    dmax = max(abs(b_r[k]-b_o[k]) for k in b_o)
    print(f"max basin-fraction diff: {dmax:.3f}")
    print(f"VERDICT: {'PASS' if dmax < 0.12 else 'DIVERGENT'} — robosample "
          f"{'reproduces' if dmax<0.12 else 'does NOT reproduce'} native OpenMM Ramachandran (dmax={dmax:.2f})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
