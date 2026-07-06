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

**🟠 Code-confirmed, high-severity (runtime-test pending):**
- `EnergySnapshot::validate` hard-rejects `|PE/refPE|>10` *before* Metropolis (`EnergySnapshot.cpp:165`) — the codebase's own header documents this as a known ergodicity bug that blocks barrier crossings.
- Momenta resampled once per round, not per move (`World.cpp:4212`) — breaks HMC detailed balance when `samplesPerRound>1`.
- `OPENMM::setActiveForceGroup` fully commented out (`OpenMM.cpp:246`) — per-world rigidification never applied to OpenMM energies.
- Factor-of-2: improper-harmonic torsion missing ×2; `FixmanTorqueExt` `cot` vs `2·cot` (self-flagged `// wrong`).

**🟡 Structural:** only 4 Python modules are the real library; `build_flexibilities`/`create_torsional_bonds`/`selectBonds` all broken vs `add_robotic_world`; OpenMM fork ~stock 8.5, real patches in Simbody/Molmodel forks; sampler test coverage ≈ nil (`test_fixman_potential.py` 100% commented); classic `inp.*` format removed (Python API only).

Full per-subsystem analysis → vault architecture map `30-Resources/Methods/robosample-architecture.md`.

### Safety
- Pre-upgrade working `.so` + fix patches backed up: `scratchpad/robosample_working_backup_2026-07-06/`.
- All work isolated on the fork; upstream untouched; Alex's concurrent E2 sims never disturbed.
