# Robosample — Improvement Roadmap (tool-focused)

Robosample as a general-purpose GCHMC internal-coordinate molecular-simulation engine —
improved on its own merits, for any user/system. Derived from the 2026-07-06 diagnosis
(`30-Resources/Methods/robosample-diagnosis.md`); the technical findings stand, re-prioritized
here by general engineering value rather than any single project's fit.

Owner: me = autonomous on the fork · alex = design/decision input. Each `src/` change re-runs
the P0 tests before commit (provenance gate). Checkboxes track progress. Suggested order at end.

---

## P0 — TRUST: make the engine verifiably correct
*A Monte Carlo engine no one can regression-test is not trustworthy. The REMC ensemble bug is
fixed; nothing yet guards it, and the sampler test suite is falsely green.*

- [ ] **P0.1 Golden-ensemble regression test** — fixed-seed 2-replica REMC on ala-dipeptide;
  assert mean backbone RMSD-to-initial > 0.5 Å over the last N frames. This is the exact metric
  that exposed the REMC bug (buggy 0.10 Å vs fixed 0.74 Å); it proves the fix and blocks its
  return. *[me · low · validate: passes on fixed build; would fail at 0.10 Å]* ← STARTING NOW
- [ ] **P0.2 Detailed-balance / Fixman-consistency test** — torsional-world vs Cartesian-world
  marginal energy histograms statistically indistinguishable (KS); plus a `mdSteps=0` +
  AlwaysAccept Boltzmann-invariance check. Catches the momentum bug + any KE-bookkeeping
  regression. *[me · low–med]*
- [ ] **P0.3 Un-break the falsely-green sampler tests** — `test_rigid`/`test_write_dcd`/
  `test_chamber` call non-existent `getDefaultBonds`/`add_torsional_world`; implement those as
  real thin wrappers over the working `add_robotic_world` path (also fixes the API, P3.1).
  *[me · low · validate: pytest collects + passes]*
- [ ] **P0.4 Momentum refresh** — make each within-round sample a valid HMC move: default
  `samplesperRound=1` immediately (correct, zero-risk), then implement per-move
  `perturbVelocities` + KE re-bookkeeping, gated behind P0.2. *[me · med]*
- [ ] **P0.5 Minimal CI** — GitHub Actions running the (now-real) pytest suite on the CPU preset;
  catches API drift + import breakage (the class of failure that killed all 3 sampler tests).
  *[me · low]*

## P1 — CAPABILITY: expand what the engine can actually do
*The gaps that most limit robosample's usefulness across systems.*

- [ ] **P1.1 Explicit solvent / PME** — biggest capability gap: only `NoCutoff`/`CutoffNonPeriodic`
  GB today (`OpenMM.cpp:636`), no periodic box / PME / barostat. Every solvated-system user needs
  this. Large: NonbondedForce PME + box + barostat in the BAT framework. (Note: the `disasm`
  branch already implements explicit solvent + NCMC — evaluate porting-vs-rebuilding.)
  *[alex design + me · high]*
- [ ] **P1.2 Re-enable glycan + nucleic dihedral classifiers** (`amber_dihedral.py:815`, code
  already written but commented out) — without it, glyco/NA torsions are "non-standard", never
  auto-selected, sugar rings cut arbitrarily. Restores automatic coverage for a whole class of
  biomolecules. Test on a Man9 glycan + RNA duplex. *[me · med]*
- [ ] **P1.3 CV-biased sampling (metadynamics/OPES) hook** — general enhanced-sampling: inject a
  bias force at the OpenMM segment (PLUMED plugin / CustomCVForce) so the existing Metropolis
  absorbs it. The general fix for the "no barrier lowering" limitation — lets robosample cross
  barriers on any user CV. *[alex/me design · hard]*
- [ ] **P1.4 REST2 / solute-scaling Hamiltonian-REMC** — per-replica λ-scaling of a chosen region
  via OpenMM force groups; defeats T-REMC's N-scaling. The swap kernel already exists. General
  capability for localized conformational problems. *[me · med]*
- [ ] **P1.5 Ring-pucker / backbone-angle sampling** — torsional-only freezes ring pucker (sugars,
  proline) and backbone bending. Add a detection + soft-closure or per-region Cartesian-relaxation
  world so these DOF are sampled. *[me · med–high]*

## P2 — PERFORMANCE
- [ ] **P2.1 Default small internal-coord systems to the CPU OpenMM platform** — CUDA is used as a
  per-MD-step force server and is latency-bound; often 2–10× slower than CPU for peptides. Add an
  atom-count threshold (CPU path already exists). *[me · low · validate: wall-clock CPU vs CUDA]*
- [ ] **P2.2 Fuse per-step force+energy device sync** (VERLET path) — one `getState(Forces|Energy)`
  instead of two. Halves device syncs. *[me · low]*
- [ ] **P2.3 Gate hot-loop stdout** (`HMCSampler.cpp:162` behind verbose) + **vectorize the O(n²)
  pandas setup** (`context.py:469`). *[me · trivial]*
- [ ] **P2.4 Avoid `realizeTopology()` per world/replica transfer** — move mobilizer frame/mass
  updates to Instance stage so a transfer is `realize(Instance)`+`realize(Position)`, not a full
  topology rebuild. Biggest large-system win; Simbody-internals surgery — spike + validate first.
  *[me · high]*
- [ ] **P2.5 Concurrent replicas** — batch small-system replicas on the GPU (one System with K
  non-interacting copies, or K contexts on CUDA streams) instead of the serial loop. *[me · high]*
- [ ] **P2.6 Wire the dead MTS integrator + force groups** (Cartesian/OMMVV path) — bonded inner,
  nonbonded every 2–4 steps; the r-RESPA integrator is already written. *[me · med]*

## P3 — SOFTWARE HEALTH
- [ ] **P3.1 Collapse the flexibility API to one correct public function** — kill the 5-way naming
  disagreement (`build_flexibilities`/`create_torsional_bonds`/`selectBonds`/…); one method that
  does the raw-prmtop→BAT conversion internally. *[me · low–med]* (folds into P0.3)
- [ ] **P3.2 Dead-code + stub triage** — remove/quarantine the 34–36% commented C++; convert the 13
  `Replica.cpp` `SimTK_ASSERT_ALWAYS(false)` abort-stubs to catchable exceptions; flag the
  commented-out live-call-outs (the exact bug pattern). *[me · med]*
- [ ] **P3.3 Split `Context.__init__`** — extract the ~700-line parse into a free
  `build_system_topology()` so it's unit-testable and the DSSP/parmed cost is opt-in. *[me · med]*
- [ ] **P3.4 Portability + versioning** — pin `-march=x86-64-v3` + explicit CUDA arch list (the
  `.so` is host-locked today); single `__version__` + git SHA stamped into the module and every
  DCD; downgrade `ccache` REQUIRED→optional. *[me · low]*
- [ ] **P3.5 Reduced-potential / MBAR + ESS emission** — write per-sample reduced potentials +
  autocorrelation so any run is analyzable/convergence-checkable (currently K−1 replicas of data
  are discarded). *[me · low]*

---

## Suggested order
P0.1 → P0.3/P3.1 → P0.2 → P0.4 → P2.3 → P2.1 → P0.5 → P1.2 → P3.4 → P2.2 → P3.5 → P1.4 →
(P1.3, P1.1, P2.4, P2.5, P2.6, P1.5, P3.2, P3.3 — larger, sequenced as capacity allows).
Start: **P0.1 now.**
</content>
