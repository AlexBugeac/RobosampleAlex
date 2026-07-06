# Overnight autonomous campaign — 2026-07-06 → 07

**Started:** 2026-07-06 23:22 EEST · **HARD STOP:** 2026-07-07 07:22 EEST (8h).
**Operator:** Claude (Opus 4.8), autonomous, on Alex's behalf. Branch: `accel-engine`
(fork AlexBugeac/RobosampleAlex) — all night's work committed here, tagged `[disasm]`/`[accel]`;
branch reorganization is a morning cleanup.

## Two topics (balance both)
- **T1 — Improve + test + benchmark `disasm`:** finish Phase-0 validation baseline, write an
  INDEPENDENT external-oracle test (`exp(−βU)` + native-OpenMM cross-check, since the suite is
  Claude-written), fix the Tier-2 failures, and run disasm on the STANDARD method-comparison
  benchmarks (alanine dipeptide φ/ψ FES first, then deca-alanine / small peptides).
- **T2 — `accel-engine` branch work:** Phase A profile (ABA vs ln det M vs OpenMM round-trip),
  Phase B CPU-vs-CUDA A/B (the near-free small-system win), Phase C JAX batched-replica PoC
  scaffolding (adam/JaxSim, reproduce M/ln det M vs disasm reference).

## ⛔ HARD CONSTRAINT — NO E2
Do **NOT** involve the HCV E2 project in any of this work. Use ONLY standard, generic
method-comparison benchmark systems (alanine dipeptide, poly-alanine, butane/ethane/2-butanol,
chignolin, Trp-cage, and the repo's generic `examples/`). **No E2 prmtops / structures / CVs /
trajectories / db-mutant topologies.** The 3 running E2 sims are to be *protected* (memory-cap
builds) but never read, used, or referenced. All benchmarks + oracle tests use generic systems.

## Operating rules (autonomous)
1. **Wall-clock hard stop 07:22.** Check `date` each tick; past deadline → write morning summary, stop, no more wakeups.
2. **Self-paced.** Long builds/tests → detached `systemd-run --user` service (survives session); short work inline. Wake ~20–30 min to check + advance.
3. **Commit every self-contained result** to `accel-engine` with a one-line ledger entry below.
4. **Graceful degradation:** a failed build/test → log it, move to the next queue item; never stall the whole night on one blocker.
5. **Protect the 3 running sims:** memory-cap all builds (`MemoryMax=5G`); check RAM/sims before launching.
6. **Only skip/pause for ambiguous data-destroying or cross-user actions** (none expected).

## Queue (priority order; ✎=in progress ✅=done ✗=failed/skipped)
- [x] T1.a ✅ disasm API learned (`AmberDihedralClassifier`→`Context(name,seed,clf)`→`load_amber`→worlds→`run_rex`); `attrs` dep installed; module imports
- [x] T1.b ✅ Tier-1 PE-ladder 7/7 PASS (vs native OpenMM). Tier-2 4-fail reproduction pending.
- [ ] T1.c INDEPENDENT oracle test: sample known 1D torsion U(φ) → compare histogram to exact exp(−βU)/Z AND native OpenMM MD
- [ ] T1.d Fix 2-butanol Tier-2 (sub-grid Boltzmann weighting, `test_torsion_conformational.py:306`)
- [ ] T1.e Ethane equipopulation discriminating run (~20 min): test-side vs real RobotEngine bias?
- [ ] T1.f Benchmark: alanine dipeptide φ/ψ free-energy surface vs reference; then deca-alanine
- [ ] T1.g Fix remaining Tier-2 (raise N_eff, calibrate bands)
- [ ] T2.a Phase A — profile the disasm HMC loop (component wall-times)
- [ ] T2.b Phase B — CPU/Reference vs CUDA platform A/B wall-clock on a peptide
- [ ] T2.c Phase C — JAX + adam/JaxSim install; reproduce M / ln det M vs disasm for a small tree
- [ ] T2.d Fold profiling + A/B numbers into DESIGN.md + this log

## Ledger (newest first)
- 00:41 cpu-release robo_bindings build DONE (323/323) — accel Phase-B A/B now runnable (CPU .so in build/cpu-release). butane 8000-run ~done. Next: butane verdict, verify 2-butanol fix, then CPU-vs-CUDA wall-clock A/B (swap the python/robosample robo_bindings symlink per-platform between runs).
- 00:36 accel Phase-B finding: disasm OpenMM platform is COMPILE-TIME (USE_CUDA→always CUDA, no runtime switch) — so CPU-vs-CUDA A/B needs a separate CPU build. Launched cpu-release robo_bindings build in PARALLEL (cpubuild-svc, capped 4G) — CPU-bound, runs alongside GPU sampling. butane 8000-run still going.
- 00:29 butane 30k-round run killed at 19437 moves (over-long, monopolized GPU); relaunched at 8000 rounds for a faster verdict. GPU is single — runs are sequential; each robosample run ~7-10min.
- 00:24 ala-dipeptide Ramachandran benchmark script written (robosample phi/psi vs native OpenMM basins, 8000-round budget). Ready to launch when GPU frees. Butane run still going (~15.5k moves) — 30k budget was over-generous; using 8000 for later benchmarks.
- 00:18 2-butanol Tier-2 FIX implemented (additive): _subgrid_pmf helper (sub-grid-averaged bin PMF) + wired ONLY into the 2D 2-butanol fixture via existing ref_pe_grid arg (subdiv=4). _chi2_gof_nd + all 1D tests untouched; does NOT loosen bands. Syntax OK. UNVERIFIED (GPU busy w/ butane) — verify next tick via pytest -k 2butanol.
- 00:10 2-butanol Tier-2 bug FULLY diagnosed: _chi2_gof_nd (test_torsion_conformational.py:~305) uses w=exp(-beta*ref_pe_grid) at BIN CENTRE; for coarse 2D bins (36deg/axis, _N_BINS_2D=10) near a torsion wall the bin-AVERAGED Boltzmann weight != centre value -> chi2=9055 (test-side, not sampler). Fix = evaluate PE on a KxK finer sub-grid per coarse bin, average exp(-beta*U) per bin (mirror KE-Gamma gammaSubsampledWeights). Implementing next tick. butane run still going.
- 23:57 Tier-1 PE-ladder = 7/7 PASSED (robosample vs native OpenMM; benign CUDA-teardown noise). Baseline reproduced.
- 23:57 T2.2 lead CORRECTED (honest): the test DOES use native-OpenMM references (rigid-scan PE + Langevin MD), sound methodology — so the [0.33,0.33,0.34] is an UNDERSAMPLED native ref (both sides), not wrong methodology, matching the devs' 'stuck/undersampled' note. My well-sampled oracle (~80% anti) is the convergence target both should reach. Next: run robosample-butane well-sampled, confirm it hits ~80% anti.
- 23:51 T1.c PROGRESS: independent OpenMM oracle for butane C-C-C-C torsion = ~80% anti / 20% gauche (|phi|>2rad=0.798). LEAD: failing Tier-2 T2.2 has robosample anti~0.78 (MATCHES my oracle) vs a [0.33,0.33,0.34] equipop reference that is physically WRONG for butane -> likely a TEST-side bug, not sampler bias (verify next tick). Minor real issue: gauche+/- asymmetry 0.17 vs 0.05 (undersampling). Tier-1 still running.
- 23:3x T1.a done: API learned, attrs installed, import OK. Tier-1 launched (tier1-svc). E2-free constraint recorded.
- 23:22 campaign set up; robo_bindings.so built; deadline 07:22.
