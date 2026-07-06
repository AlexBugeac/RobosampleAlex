# Robosample `disasm` — Work Plan

Working area: `/home/alexb/Robosample_disasm` (fork branch `disasm-work` off `origin/disasm`
`137a739`). Refactor work stays isolated on `refactor-updated-2026-07-06`.

## Framing — why "apply the refactor fixes" is a no-op here (verified)

disasm is a **ground-up rewrite that removes Simbody** (in-tree `RobotEngine`/`RobotState`),
so the refactor fixes don't transplant — they're already handled:

| Refactor fix | disasm state |
|---|---|
| REMC coordinate-reset guard | **Gone by construction** — single `replicaCoords_` array, swap is a clean `std::swap`, no `WORK` buffer |
| Momentum per-round | **Correct by construction** — per-move momentum draw |
| `validate()` filter / `setActiveForceGroup` / improper ×2 | Not bugs; restructured away |
| `prmtop_reader` divide-by-zero | **Already fixed** (`np.errstate` + zero-mask, `prmtop_reader.py:283,316`) |
| `write_freq=0` guard, `initialize()` guards, example/API fixes | Different API (Context takes `dihedral_classifier`; `JointType` not `BondMobility`) — re-assess per file, most already handled |

**Nothing to port.** The real value on disasm is finishing what it started: its validation suite.

## Current state (from `docs/specs/ensemble-validation/90-implementation-status.md`)

disasm has the **real ensemble-validation suite refactor never had**:
- **Tier 0** (C++ equipartition / kinetic invariants) — ✅ PASS (incl. slow T0.2)
- **Tier 1** (PE ladder, 7 rungs) — ✅ PASS 7/7 (~28 min)
- **Tier 2** (torsion conformational) — ⚠️ **12 pass / 4 fail** ← the production-readiness gate
- Stats self-checks 8/8, PE-parity fixtures 3/3 — ✅

The 4 Tier-2 failures are documented with leading diagnoses and a discriminating experiment.

---

## PHASE 0 — Build + reproduce the baseline  *(foundation; do first)*
- [ ] **0.1** Init submodules in the worktree (`git submodule update --init`) and configure the
  CUDA build (`ccache`, `-DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so.0`, memory-capped
  `systemd-run --scope -p MemoryMax=3500M ninja -j2` to protect running sims).
- [ ] **0.2** Run **Tier 0** (fast C++): `ctest -R 'Equipartition|EnsembleValidation'` → confirm PASS.
- [ ] **0.3** Reproduce **Tier 1** PE-ladder PASS (7/7) and the **4 Tier-2 failures** exactly, so we
  have a trustworthy baseline before changing anything.
- **Exit:** disasm builds, validation harness runs, baseline (12 pass / 4 fail) reproduced.

## PHASE 1 — Close the 4 Tier-2 failures  *(the gate to "disasm is trustworthy")*
Follow the devs' own resume order — **do NOT loosen bands until green** (that hollows out validation).
- [ ] **1.1 Fix the 2-butanol χ²=9055 (test-side, cheapest, do first).** `_chi2_gof_nd` weights the
  reference at the **bin centre** (`test_torsion_conformational.py:306`); with coarse 36° 2D bins near
  a torsion wall this is badly biased. Fix: evaluate `_reference_pe_grid` on a finer sub-grid and
  aggregate the mean Boltzmann weight per coarse bin (mirror `TestEnsembleValidation.cpp::
  gammaSubsampledWeights`). No sampler re-run needed to test the hypothesis.
- [ ] **1.2 Run the discriminating experiment:** ethane 3-fold equipopulation, ~20–25k prod rounds
  (~15–20 min). Does it converge to [1/3,1/3,1/3]?
  - **If YES** → all four are test-side (undersampling/binning). Proceed to 1.3.
  - **If NO** → a **real sampler bias** in the new RobotEngine — escalate as a genuine finding
    (this is the highest-value possible outcome: a correctness issue in the rewrite's core).
- [ ] **1.3** Raise N_eff for the population tests (`_T2_0_PROD_ROUNDS`, `_T2_0_MDSTEPS`,
  `_NATIVE_BASIN_N_SAMPLES`); calibrate the equality bands from the symmetry-exact ethane case and
  reuse for butane; re-run all Tier 2.
- **Exit:** Tier 2 → 16/16 green (or a documented, escalated real-bias finding). disasm is then a
  **validated** engine — something refactor never was.

## PHASE 2 — Capability: make the disasm-specific features production-grade
- [ ] **2.1 NCMC Metropolized-inner (Construction II).** disasm's endpoint-ΔH NCMC doesn't scale to
  explicit solvent (devs' own diagnosis: acceptance extensive in bath size). The coded-but-off
  `useMetropolizedInner` path is the fix — validate it recovers non-zero acceptance for 2-ala in
  TIP3P and gate it on. *[the marquee capability disasm exists for]*
- [ ] **2.2 Explicit-solvent validation** — extend the ensemble suite (a solvated Tier-1 rung) so
  explicit-solvent runs are certified, not just implemented.
- [ ] **2.3** Enhanced-sampling hooks (CV-bias/OPES, REST2) — same general value as on refactor,
  but built on the trusted RobotEngine.

## PHASE 3 — Software-health carryover  *(only what actually applies to disasm)*
- [ ] **3.1** CI running the Tier-0 (fast) + stats-self-check suite on push (disasm HAS real tests —
  wire them into GitHub Actions so regressions can't merge silently).
- [ ] **3.2** Version stamp (`__version__` + git SHA into the module + every DCD) — reproducibility.
- [ ] **3.3** Portability: pin `-march=x86-64-v3` + explicit CUDA arch list (avoid the host-locked
  `.so` across the lab fleet).
- [ ] **3.4** Audit disasm's own API/example health (the `run_*.py` drivers, `build_flexibilities`
  under the new `JointType` API) — fix any broken examples the way we did on refactor, IF broken.

---

## Suggested order
0.1 → 0.2 → 0.3 → **1.1 → 1.2 (discriminating)** → 1.3 → 3.1/3.2/3.3 → 2.1 (NCMC) → 2.2 → 2.3 → 3.4.
Phase 1 is the whole game: a green Tier-2 (or an escalated real finding) is what makes disasm
adoptable. Everything else is downstream of that.

## What this plan deliberately does NOT do
- Port refactor fixes (verified no-ops).
- Loosen validation bands to force green (would destroy the suite's value).
- Adopt disasm for production before Phase 1 closes.
</content>
