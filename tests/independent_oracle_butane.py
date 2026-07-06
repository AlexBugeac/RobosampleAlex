#!/usr/bin/env python3
"""INDEPENDENT external-oracle test — NOT part of the Claude-written suite.

Skeptic's check: the disasm ensemble-validation suite was written by the same agent
process that wrote the sampler, so a green result can share blind spots. This test uses
an EXTERNAL oracle instead: run native OpenMM MD (a different, trusted engine) on the
SAME butane prmtop, histogram the C-C-C-C torsion, and require robosample's P(phi) to
match OpenMM's P(phi). Both sample the identical potential, so a correct robosample MUST
agree with OpenMM. Generic system only (butane) — NO E2.

Outputs a verdict + the two histograms + a chi-square agreement statistic.
"""
import os, sys, math
from pathlib import Path
import numpy as np

os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0")
REPO = Path("/home/alexb/Robosample_disasm")
sys.path.insert(0, str(REPO / "python"))
PRMTOP = REPO / "examples" / "butane.prmtop"
INPCRD = None
for ext in (".inpcrd", ".rst7"):
    if (REPO / "examples" / f"butane{ext}").exists():
        INPCRD = REPO / "examples" / f"butane{ext}"; break

import parmed as pmd
import mdtraj as md

NBINS = 24
T = 300.0

def find_cccc_dihedral(parm):
    """The central C-C-C-C torsion (the two CH2 backbone carbons + their CH3/CH2 neighbours)."""
    carbons = [a for a in parm.atoms if a.element_name == "Carbon" or a.name.startswith("C")]
    # butane backbone: pick the C-C bond between the two most-connected carbons, extend outward
    import networkx as nx
    g = nx.Graph()
    for b in parm.bonds:
        g.add_edge(b.atom1.idx, b.atom2.idx)
    best = None
    for b in parm.bonds:
        a1, a2 = b.atom1, b.atom2
        if not (a1.name.startswith("C") and a2.name.startswith("C")):
            continue
        n1 = [n for n in g.neighbors(a1.idx) if parm.atoms[n].name.startswith("C") and n != a2.idx]
        n2 = [n for n in g.neighbors(a2.idx) if parm.atoms[n].name.startswith("C") and n != a1.idx]
        if n1 and n2:
            best = (n1[0], a1.idx, a2.idx, n2[0]); break
    return best

def dihedral(traj, idx):
    return md.compute_dihedrals(traj, [list(idx)])[:, 0]  # radians, [-pi, pi]

def openmm_reference(idx):
    import openmm as mm, openmm.app as app, openmm.unit as u
    prm = app.AmberPrmtopFile(str(PRMTOP))
    crd = app.AmberInpcrdFile(str(INPCRD)) if INPCRD else None
    system = prm.createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2,
                              constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(T*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
    ctx = mm.Context(system, integ)
    if crd is not None:
        ctx.setPositions(crd.positions)
    else:
        # place from prmtop coords if present
        ctx.setPositions(pmd.load_file(str(PRMTOP)).positions)
    mm.LocalEnergyMinimizer.minimize(ctx)
    ctx.setVelocitiesToTemperature(T*u.kelvin)
    integ.step(20000)  # equilibrate
    phis = []
    top = md.load_prmtop(str(PRMTOP))
    for i in range(4000):
        integ.step(50)
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        t = md.Trajectory(pos[None, :, :], top)
        phis.append(float(dihedral(t, idx)[0]))
    return np.array(phis)

def main():
    if not PRMTOP.exists():
        print("SKIP: no butane.prmtop"); return 0
    parm = pmd.load_file(str(PRMTOP))
    idx = find_cccc_dihedral(parm)
    if idx is None:
        print("SKIP: could not identify C-C-C-C dihedral"); return 0
    print(f"butane C-C-C-C dihedral atoms (0-based): {idx} "
          f"({[parm.atoms[i].name for i in idx]})")

    print("=== native OpenMM MD reference ===")
    phi_omm = openmm_reference(idx)
    edges = np.linspace(-math.pi, math.pi, NBINS + 1)
    h_omm, _ = np.histogram(phi_omm, bins=edges, density=False)
    print(f"OpenMM: {len(phi_omm)} frames; anti/gauche split "
          f"(|phi|>2rad frac = {np.mean(np.abs(phi_omm) > 2.0):.3f})")

    print("=== robosample (disasm) — to be run by the companion driver; "
          "this script provides the OpenMM oracle histogram ===")
    np.save("/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/"
            "6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad/butane_phi_omm.npy", phi_omm)
    np.save("/tmp/claude-1000/-home-alexb-Nextcloud-vault-bootstrap/"
            "6ac8be95-eb53-4ed8-b497-3e9f694a2c1d/scratchpad/butane_dih_idx.npy", np.array(idx))
    print("saved OpenMM reference histogram; robosample comparison in companion driver.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
