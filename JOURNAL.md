# Robosample Fork — Work Journal

Lab notebook for autonomous work on `AlexBugeac/RobosampleAlex` (fork of `spirilaurentiu/Robosample`).
Working branch: `refactor-updated-2026-07-06`. Machine: alexander (RTX 5080, CUDA 13.2).
Maintained by Claude (Opus 4.8) on Alex's behalf. Newest entries at top.

---

## 2026-07-06 — Session 1: upgrade, fixes, fork, deep-read

### Fork & remotes
- Fork exists as `AlexBugeac/RobosampleAlex` (parent `spirilaurentiu/Robosample`). Wired as git remote `fork`.
- Pushed working branch `refactor-updated-2026-07-06` to the fork.
- `origin` = upstream `spirilaurentiu/Robosample` (read-only reference).

### Branch state
`refactor-updated-2026-07-06` = upstream `origin/refactor` (`9bc3b82`) + local commits:
- `5d0ca26` — **preserve `add_robotic_world` prmtop→BAT `converted` fix** atop the merge. Upstream STILL ships this broken; without it, torsional worlds fail. This is load-bearing.
- `cb09f1a` — fix shipped `test_installation.py` (was `AttributeError` via the broken `build_flexibilities` path); guard `write_freq=0` (was an FPE core dump) → clear `ValueError`.
- `a5b3da6` — fix `run_2ala.py` per-world-temp example (same `build_flexibilities` bug); fix `prmtop_reader.py` divide-by-zero warning; add `initialize()` guards (no-worlds / empty-temps / non-positive temps).

### Upgrade notes (63b18a7 → 9bc3b82)
- New upstream build deps: **ccache** (installed into the `robosample` conda env) + explicit **BLAS/LAPACK** at cmake (`-DBLAS_LIBRARIES=$CONDA_ENV/lib/libopenblas.so.0`, same for LAPACK).
- Rebuilt CUDA `robo_bindings.so` (memory-capped 3.5 GB cgroup, `nice -j2`, to protect concurrent sims). 606 TUs, ccache cold.
- Brought in: per-world temperatures, BoundHMC sampler, CYX ring-closing fix.

### Verified this session (on ala-dipeptide, RTX 5080)
- New build runs end-to-end (2-world Cartesian+torsional, CUDA), identical correctness to pre-upgrade.
- Shipped `test_installation.py` now passes.
- `write_freq=0` → clean `ValueError`.
- **Per-world temperatures WORK**: World 0 @300 K + World 1 @1000 K (and 3000 K via run_2ala) in one job.
- REMC swap machinery active (`REX`/`REXdetails` diagnostics every round — the v1 zero-swap bug's path is present).
- `initialize()` guards fire correctly.
- CYX-fixed build loads the db411-424 topology (13 disulfides incl. long-range) and builds the prototype cleanly.

### Known-unfixed (needs maintainer review / broad GPU-scale testing)
- **`build_flexibilities` is broken for ALL callers**: returns a nested, pre-converted structure `add_robotic_world` rejects, AND is called with inconsistent arity (`(bonds,mobility,roll)` in examples vs `(bonds)` in `autoblock.py`/`run_ffar1.py`). Root fix = make it return a flat raw-prmtop-index list; touches GPCR-scale callers not cheaply validated. Candidate for a proper fix on this fork now that we can iterate freely.

### Deep read (6-agent fleet) — headline findings

**🔴 RUNTIME-CONFIRMED + FIXED: REMC ensemble corruption.** `Context::attemptREXSwap` promoted `WORK` coords→final on every accepted swap because the `if (runType == RENE||RENEMC||REBASONTOP)` guard was commented out (`src/Context.cpp:1392`). In plain REMC, `WORK` holds the *initial* structure (never refreshed), so accepted swaps reset both replicas to the start conformation.
- Verified: 300/301 K, 51 accepted swaps → repl0 RMSD-to-initial mean **0.10 Å, 67/105 frames pinned** (collapsed). No-swap control drifts to 0.72 Å.
- Fix: restored the guard. Re-verified: same 51 swaps → mean **0.74 Å, 1/105 pinned** (free drift). Fixed.
- Impact: past robosample REMC ensembles are biased hard toward the starting structure — explains why robosample REMC never gave real E2 conformational sampling (STATUS.md treated it as frame-finder only).

**Re-assessed after direct code reading + measurement (severity corrected):**
- **`validate()` ergodicity filter — DOWNGRADED to benign.** The *live* `EnergySnapshot::validate` (`EnergySnapshot.cpp:137`) uses raw ratios `|PE/refPE|>10` (`:165`) and `|KE/refKE|>100000` (`:177`), NOT the modified-relative `is_exploded` formula (that's the commented-out Category-B path). For real systems these essentially never fire (ala-dipeptide PE∈[−134,−39] → ratio max ~3.5; nothing swings KE 1e5×). Earlier "blocks barrier crossings" framing was an over-read; corrected. No fix warranted.
- **Momentum-refresh cadence — CONFIRMED (conditional).** Velocities drawn only in `reinitialize` (once per round, `World.cpp:4212`); `sampleIteration`'s `perturbVelocities` is commented out (`HMCSampler.cpp:3560`). For `samplesperRound>1` the within-round HMC moves reuse end-state velocities → non-canonical. Impact: **`run_e2_remc_2w.py` (samplesperRound=1) is UNAFFECTED**; **`run_e2_remc_v2.py` (samplesperRound=10) IS affected.** Zero-risk workaround: use `samplesperRound=1` + more rounds. Proper per-move-momentum fix is delicate (needs KE-bookkeeping consistency) — deferred to a validated change. disasm fixed this by construction (per-move).
- **`setActiveForceGroup` no-op — DOWNGRADED to missed optimization (not a correctness bug).** Verified: both current (`reinitialize:145`, "using OpenMM regardless of" world type) and proposed (`sampleIteration`, `evaluatePotentialEnergyFromPositionsCache`) energies come from OpenMM — NOT "DuMM current vs OpenMM proposed" as the regression audit claimed. With the stub dead, intra-rigid-body energy is included in BOTH; but a torsional move never changes intra-body geometry (Simbody holds bodies rigid) → those terms are constant and cancel in ΔH. The zeroing was an optimization (skip computing constant terms), not needed for correctness.
- **NMA-boost KE transposition** (`HMCSampler.cpp:3672-3673`, regression audit H2) — real, but ONLY affects `DistortOpt>0` (NMA/boost) runs; standard E2 REMC (DistortOpt=0) unaffected.
- **Factor-of-2**: improper-harmonic torsion missing ×2 (affects CHARMM impropers only); `FixmanTorqueExt` `cot` vs `2·cot` (self-flagged `// wrong`, only active on Free/Ball-root worlds — refactor enabled it, singularity had it disabled).

### Verified severity synthesis (after runtime + code verification of every "critical" agent claim)
| Finding | Raw agent severity | VERIFIED verdict | Affects Alex's E2? |
|---|---|---|---|
| REMC coord-reset | critical | **CONFIRMED critical, FIXED** (c9e3d46) | YES (2w/v2 both) — now fixed |
| Momentum once/round | critical | CONFIRMED, only `samplesperRound>1` | v2 only; workaround: use 1 |
| `validate()` ergodicity filter | critical | **BENIGN** (thresholds never fire; PE ratio max ~3.5 vs 10) | no |
| `setActiveForceGroup` / A1 mismatch | critical | **NOT a bug** (intra-rigid cancels; both OpenMM) | no |
| NMA-boost KE transpose | critical | real, `DistortOpt>0` only | no (standard runs) |
| Metropolis ratio, Fixman | — | verified IDENTICAL to singularity, correct | — |
**Net: one real ensemble-corrupting bug (REMC, fixed); one conditional (momentum, workaround); the rest benign/boost-only/optimization. The "verify before alarming" discipline overturned 3 of 5 critical claims (validate filter, setActiveForceGroup, AND — later — the improper-torsion ×2).**

### 4-specialist diagnostic panel (sampling · structural-bioinformatics · RSE · HPC)
Full diagnosis + prioritized roadmap → vault `30-Resources/Methods/robosample-diagnosis.md`. Headlines:
- **Sampling:** plain GCHMC+T-REMC gives NO barrier lowering → cannot cross the AS412 barrier at feasible cost (W1); T-REMC N-scaling wall (W2). Fixes that fit Alex's toolkit: REST2/solute-scaling H-REMC, CV-bias/OPES on the AS412 CV (his own d413-451/d418-527), MBAR+ESS.
- **Structural bioinformatics:** FATAL for E2 — no explicit solvent (GB-only, `OpenMM.cpp:636`); Alex's own data shows the glycan effect is invisible in implicit (Δ+0.06Å) vs explicit (Δ−2.10Å). Glycan/nucleic dihedral classifiers commented out (`amber_dihedral.py:815`). Torsional-only freezes ring pucker + backbone-angle.
- **RSE:** test suite is FALSELY GREEN (every sampler test calls non-existent methods → AttributeError; badge measures only math helpers); nothing guards against reintroducing the fixed bugs. Golden-ensemble + detailed-balance tests (~2 days) would've caught both.
- **HPC:** GPU used as a per-MD-step force server in the default VERLET path → latency-bound; for peptides CUDA is likely SLOWER than CPU (P1). `realizeTopology()` per world transfer (P2). Serial replicas, no batching (P4). Quick wins: default small systems to CPU (2-10×), fuse syncs, gate stdout, vectorize pandas.

**STRATEGIC CONCLUSION:** for the E2 glycoprotein problem specifically, robosample is NOT the right primary tool — explicit solvent + AS412 barrier-crossing are things Alex ALREADY has in his OpenMM+PLUMED/OPES pipeline (which produced his actual primary evidence). Robosample's viable role = fast GB/vacuum torsional conformer generator feeding explicit-solvent validation, AFTER re-enabling glycan classifiers. Frontier = `disasm` (explicit solvent + NCMC) once its Tier-2 validation is green.

**🟡 Structural:** only 4 Python modules are the real library; `build_flexibilities`/`create_torsional_bonds`/`selectBonds` all broken vs `add_robotic_world`; OpenMM fork ~stock 8.5, real patches in Simbody/Molmodel forks; sampler test coverage ≈ nil (`test_fixman_potential.py` 100% commented); classic `inp.*` format removed (Python API only).

Full per-subsystem analysis → vault architecture map `30-Resources/Methods/robosample-architecture.md`.

### Safety
- Pre-upgrade working `.so` + fix patches backed up: `scratchpad/robosample_working_backup_2026-07-06/`.
- All work isolated on the fork; upstream untouched; Alex's concurrent E2 sims never disturbed.
