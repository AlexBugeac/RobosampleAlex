"""Golden-ensemble regression test — guards against the REMC coordinate-reset bug.

The bug (fixed 2026-07-06, commit c9e3d46): Context::attemptREXSwap promoted stale WORK
coordinates (= the initial structure) to final on every accepted REMC swap, collapsing
the sampled ensemble onto the start conformation. With ~100% swap acceptance the effect
is stark and deterministic:

    buggy : mean backbone RMSD-to-initial ~ 0.10 A (ensemble pinned to start)
    fixed : mean backbone RMSD-to-initial ~ 0.74 A (free conformational drift)

This test runs a fixed-seed 2-replica REMC at 300/301 K (near-identical temperatures =>
~100% swap acceptance, the discriminating condition) and asserts the ensemble drifts
freely. It FAILS at ~0.10 A if the WORK-overwrite guard is ever removed again.

Requires the CUDA (or CPU) robo_bindings build. Skipped if the module or examples are
unavailable so the rest of the suite still collects.
"""
import os
import glob
from pathlib import Path

import pytest

os.environ.setdefault("OPENMM_CUDA_COMPILER", "/opt/cuda/bin/nvcc")
os.environ.setdefault("CUDA_ROOT", "/opt/cuda")

_REPO = Path(__file__).resolve().parents[1]
_EX = _REPO / "examples" / "ala-dipeptide"

robosample = pytest.importorskip("robosample")
mdtraj = pytest.importorskip("mdtraj")
parmed = pytest.importorskip("parmed")
from robosample import robo_bindings as rb  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (_EX.with_suffix(".prmtop").exists() and _EX.with_suffix(".rst7").exists()),
    reason="ala-dipeptide example files not found",
)

_TERMINAL_RES = {"ACE", "NME", "NMA", "NHE"}
_DRIFT_FLOOR_A = 0.5  # buggy ~0.10, fixed ~0.74 — 0.5 A cleanly separates the two


def _backbone_torsion_flexes(prmtop: str):
    parm = parmed.load_file(prmtop)
    flexes = []
    for bond in parm.bonds:
        a1, a2 = bond.atom1, bond.atom2
        if a1.residue.name in _TERMINAL_RES or a2.residue.name in _TERMINAL_RES:
            continue
        if frozenset({a1.name, a2.name}) in (frozenset({"N", "CA"}), frozenset({"CA", "C"})):
            f = rb.BondFlexibility()
            f.globalIndex1, f.globalIndex2, f.mobility = a1.idx, a2.idx, rb.BondMobility.Torsion
            flexes.append(f)
    return flexes


def test_remc_ensemble_drifts_under_accepted_swaps(tmp_path):
    """With ~100% accepted swaps the 300 K replica must still sample freely, not
    collapse onto the initial structure (the REMC coord-reset regression signature)."""
    prmtop = str(_EX.with_suffix(".prmtop"))
    rst7 = str(_EX.with_suffix(".rst7"))
    flexes = _backbone_torsion_flexes(prmtop)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx = robosample.Context(
            name="golden_remc", seed=11, prmtop=prmtop, inpcrd=rst7,
            write_freq=1, runType=rb.RunType.REMC, replicaSwapFreq=1,
        )
        ctx.add_cartesian_world().add_sampler(
            timeStep=0.001, mdSteps=30, boostMDSteps=30,
            useFixmanPotential=False, use_nuts=False)
        ctx.add_robotic_world(flexes).add_sampler(
            timeStep=0.002, mdSteps=20, boostMDSteps=20,
            useFixmanPotential=True, use_nuts=False)
        ctx.initialize([300.0, 301.0])   # near-identical -> ~100% swap acceptance
        ctx.run_rex(5, 100, 1, False)    # equil, prod, write_freq, write_to_stdio
        dcds = sorted(glob.glob("*repl0*.dcd"))
    finally:
        os.chdir(cwd)

    assert dcds, "no repl0 DCD written"
    traj = mdtraj.load(str(tmp_path / dcds[0]), top=prmtop)
    traj.superpose(traj, 0)
    rmsd_A = mdtraj.rmsd(traj, traj, 0) * 10.0  # nm -> Angstrom
    mean_rmsd = float(rmsd_A.mean())

    assert mean_rmsd > _DRIFT_FLOOR_A, (
        f"REMC ensemble is pinned near the initial structure "
        f"(mean RMSD-to-initial {mean_rmsd:.3f} A < {_DRIFT_FLOOR_A} A). "
        f"This is the REMC coordinate-reset regression: accepted swaps are overwriting "
        f"sampled coordinates with the stale WORK (initial) buffer. Check the runType "
        f"guard in Context::attemptREXSwap."
    )
