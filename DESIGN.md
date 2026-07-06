# Robosample `accel-engine` — Acceleration Architecture

Branch: `accel-engine` (off the `disasm` Simbody-free RobotEngine base, on fork
`AlexBugeac/RobosampleAlex`). Goal: a GPU/AI-accelerated robosample **without sacrificing
Monte-Carlo correctness**. Living design — pillars filled as the 4-agent research lands.

## 0. The non-negotiable gate (applies to every change on this branch)
**MC-grade correctness > speed.** Any acceleration must preserve detailed balance /
asymptotically-exact Boltzmann sampling. "Fast but biased" is disqualified. Concretely:
- The **Fixman acceptance term** `½kT ln det M(q)` enters the accept/reject decision directly,
  so its numerics set correctness, not just efficiency → **fp64, validated to ~1e-10 vs a
  trusted reference**, never float32.
- Every acceleration is gated on: (i) the independent `exp(−βU)` configurational oracle +
  native-OpenMM cross-check (being built), and (ii) unchanged HMC acceptance statistics.
- **Robotics/game-grade physics is NOT MC-grade** — float32, contact-solver stability, and
  black-box GPU stepping are the wrong optimization target. (Research pillar 1 confirmed this.)

## 1. The bottlenecks we're attacking (from the HPC + sampling reviews)
1. **Per-step CPU↔GPU round-trip** — OpenMM called once per MD step as a force server;
   latency-bound (~20–60 µs/step). For small systems the GPU is *starved* → CUDA slower than CPU.
2. **CPU-serial articulated-body dynamics** — ABA + the Fixman `M`, `ln det M`, `M⁻¹` solve.
3. **Serial-replica REMC** — replicas run one-at-a-time on a single OpenMM context; no batching.
4. **No barrier-crossing enhancement** — plain HMC/REMC can't cross kJ barriers faster than MD
   (a sampling-method gap → the AI/enhanced-sampling target, research pillar 4).

## 2. Research pillars → design decisions

### Pillar 1 — NVIDIA GPU robotics/dynamics stack  ✅ RESEARCHED
**Reject as engine:** cuRobo/Warp-sim/Newton/MuJoCo-Warp/PhysX/Isaac — all float32, none expose
`M`/`ln det M`. **Adopt two fp64 primitives instead:**
- **`cuSolverDx`/cuSOLVER — fp64 batched Cholesky** of `M(q)` → gives BOTH the `M⁻¹` solve (ABA)
  AND `ln det M = 2·Σ ln Lᵢᵢ` in one factorization. MC-grade, C/CUDA-native, no framework lock-in.
  **→ lowest-risk first target (attacks bottleneck #2's precision-critical piece).**
- **NVIDIA Warp (kernel language, Apache-2.0, fp64 + autodiff)** — to author a custom batched
  CRBA+ABA if a full GPU engine is warranted; autodiff yields the Fixman-torque gradient for free.
  Prototype in Warp → port hot kernels to raw CUDA calling cuSolverDx.
- cuRobo = read-only reference (batched RNEA / sparse-Jacobian kernels).
- **Skeptical caveat baked in:** for small trees the ABA is tiny; GPU wins only via batching
  replicas + co-locating with force eval. PoC must prove batched-GPU beats CPU **at robosample's
  real REMC replica count (16–64)** before porting the whole engine.

### Pillar 2 — GPU articulated-body algorithms & batched-replica execution  ✅ RESEARCHED
- **DON'T port single-tree ABA to GPU** — a peptide tree (tens–hundreds of bodies, depth ~10–50)
  has an O(depth) critical path that runs in tens of µs on one CPU core, *below* GPU launch+transfer
  latency. Every GPU-dynamics library that shows speedups (GRiD, Brax, MJX) gets them from BATCHING
  many systems, not accelerating one small tree.
- **CORRECTION to Pillar 1 — the Fixman `ln det M` is already O(N), no Cholesky needed.** It's a
  byproduct of the articulated-body factorization `M = [I+HφK]D[I+HφK]*` → `ln det M = Σ ln D(i)`
  (~24% overhead), and Simbody *already* computes it at linear scaling (GNEIMO / Wagner-Jain-Vaidehi
  2013, PMC3835462). ⟹ **cuSolverDx Cholesky is NOT for the core path** — only for a batched-dense
  variant if we keep each replica's M dense-small.
- **THE win = batched-replica REMC** (~10–50× vs multicore-CPU REMC, honest). Same-topology replicas
  (differ only in T/λ) = divergence-free ideal batched kernel; exchange = a device-memory shuffle
  (Brax/MJX pattern). OpenMM can't do this natively (one-System-per-Context); NVIDIA MPS (~2× via
  process concurrency) is the stopgap, a fused kernel is strictly better.
- **Cleanest prototype path:** JAX with **adam** or **JaxSim** (both expose first-class
  `mass_matrix()`, `vmap`-batch, fwd+rev autodiff, fp64 via `jax_enable_x64`), reusing the OpenMM GPU
  energy path; benchmark vs serial Simbody at N=256–4096. Native fallback = **TDS** (templated
  C++/CUDA, trivial fp64) only if JAX fp64 throughput disappoints. Pinocchio = CPU fp64 ground-truth.
- **THE precision risk (real, undocumented → validate empirically):** RL engines are fp32-tuned; MC
  needs fp64 for the acceptance reduction or round-off corrupts detailed balance. On RTX 5080 fp64 is
  throttled (½–1/64 of fp32) → all-fp64 loses much of the speedup. **Design = mixed precision: fp32
  geometry/pair energies, fp64 reduction of the acceptance criterion.** No paper quantifies fp32
  detailed-balance error in molecular REMC — measure it on the PoC, don't assume.

### Pillar 3 — Alternative force/MD engines (kill the round-trip)  ✅ RESEARCHED
Root cause confirmed: OpenMM's per-step `getState(getForces)` sync collapses GPU util ~100%→4%
(OpenMM #2052); it's the *per-step-server architecture*, not small systems per se — continuous MD
hides it by running 1000s of steps on-device between syncs.
- **#1 near-free win — swap the OpenMM platform to CPU/Reference for small systems.** One-line
  change (`Platform.getPlatformByName('CPU'/'Reference')`). For small implicit-solvent peptides in
  this per-step pattern, CPU plausibly beats CUDA outright AND gives **deterministic fp64 energies
  ideal for MC acceptance** (Reference platform). **Measure this before any rewrite — may be the
  whole answer for the small-system regime.**
- **#2 low-effort GPU win — NVIDIA MPS** to run replicas/processes concurrently, amortizing the
  idle GPU without architecture change.
- **#3 real structural fix (large systems) — fuse the whole HMC loop on-device in JAX:** MJX
  (Featherstone-in-JAX, the existence proof) + jax-md forces + BlackJAX HMC, one `lax.scan`ned XLA
  program → no per-step host traffic. Cost: must implement AMBER/CHARMM+GBSA energy in JAX (no
  framework ships it). Apache-2.0.
- **REJECT ML potentials for speed:** MACE/ANI are **50–130× SLOWER** per force eval, lack implicit
  solvent, ~1 kcal/mol accuracy (flips populations), stability failures (Fu et al. "Forces are not
  Enough"), MACE-OFF weights non-commercial. Only ever an *accuracy* project, never a throughput fix.
- **No off-the-shelf AMBER+GBSA+on-device-loop engine exists** — the fused path is build-it.

### CONVERGENCE (Pillars 1 & 3 agree independently — high confidence):
> **Small systems: the GPU was never the right device for a per-step server — go CPU (near-free,
> fp64, MC-exact). Large systems: keep the *whole* loop on-device (Warp fp64 kernels or JAX/MJX
> fusion). MC-precision is the gate; robotics-fp32 and ML-potentials are both disqualified.
> MEASURE the per-component wall-time before writing any GPU code.**
**MEASURED 2026-07-07 (accel Phase-B A/B, ala-dipeptide ~22 atoms, 500 rounds, disasm):**
`CUDA build 71.3 ms/round vs CPU build 37.0 ms/round → CPU ~1.9x FASTER.` Empirically confirms the research: for small systems the per-step CPU↔GPU round-trip starves the GPU; the CPU/Reference platform is the near-free win. Actionable: default small internal-coord systems to a CPU build/platform.


### Pillar 4 — AI-accelerated sampling  ✅ RESEARCHED
**Exactness taxonomy (the filter):** A = MH-corrected proposal (exact by construction, degrades
*gracefully* to slow-but-correct); B = reweighting (asymptotically exact but ESS collapses → silently
*confidently-wrong* at finite N); C = enhanced sampling + reweight (real theorem, exact IF the CV
spans the slow modes); D = emulators (no correction, undetectable bias). Robosample already lives in A.
- **★ WINNER — learned CV + OPES/metadynamics as a new Gibbs "world," MBAR-reweighted (class C).**
  The ONLY approach that is simultaneously barrier-lowering, asymptotically exact (WT-metad
  convergence theorem, Dama-Parrinello-Voth PRL 2014), **production-mature on real proteins**, AND
  native to robosample's multi-world torsional design (a bias world = one more world in the Gibbs
  cycle; torsional CVs are what BAT expresses). **Alex already runs OPES** → smallest lift, highest
  rigor. CVs via mlcolvar (Deep-TICA/VAMPnets) → PLUMED. Caveat: CV completeness (diagnosable via
  hysteresis / VAMP score).
- **★ #2 — normalizing-flow / learned-HMC proposal inside robosample's Metropolis (class A: SNF,
  L2HMC, Timewarp-MCMC).** Best *architectural* fit (exact by construction — robosample already does
  the accept/reject). Proven risk: acceptance-rate collapse at scale (Timewarp ~1%) → **scope to a
  local flexible region (dozens of torsions), never the whole protein.** The right home for a novel
  methods contribution.
- **REJECT as samplers:** Boltzmann Generators standalone (ESS collapse beyond peptides; only viable
  as a *proposal* → merges into #2); diffusion emulators DiG/BioEmu/MDGen (class D — use only to
  *seed* candidate states, never as the sampler); ML surrogate potentials (orthogonal — speeds energy
  eval, doesn't lower barriers, changes the target).
- **Non-negotiable acceptance gate:** any AI result must report reweighted agreement with a trusted
  reference + an ESS / MH-acceptance rate. "Structures look native" ≠ "samples the distribution."

## The unified conclusion (all 4 pillars)
Three *tractable, high-value, MC-exact* wins — none of which is the original "port the whole engine to
a CUDA robot":
1. **Speed, small systems (near-free):** OpenMM CPU/Reference platform swap. [P1,P3]
2. **Speed, throughput (~10–50×):** batched-replica REMC — JAX (adam/JaxSim), fp64-reduced acceptance,
   mixed precision, reuse OpenMM energies. The genuine GPU win. [P2,P3]
3. **Sampling efficiency (barrier crossing, the real scientific ceiling):** learned-CV OPES world,
   reweighted. Exact + mature + builds on Alex's existing OPES. [P4]
**Explicitly NOT worth it:** single-tree GPU ABA (wrong regime), robotics engines (fp32, no `ln det M`),
ML potentials for speed (50–130× slower), cuSOLVER for the *core* `ln det M` (already O(N) in Simbody).
**The gate on everything: MC-grade precision — mixed precision, validate detailed balance empirically.**

## 3b. Phased build plan (finalized from all 4 pillars)
- **Phase A — PROFILE (no GPU code yet).** Instrument the RobotEngine HMC loop: ABA vs `ln det M` vs
  OpenMM force round-trip vs rest, at real system sizes & replica counts. Decides which win is real.
- **Phase B — CPU-platform A/B (near-free, MC-exact).** Small-system peptide: CPU/Reference vs CUDA
  wall-clock + confirm identical acceptance stats. Likely the whole small-system answer. [win #1]
- **Phase C — batched-replica REMC PoC** in JAX (adam/JaxSim), N=256–4096, fp64 acceptance reduction.
  Gate: reproduce Simbody `M`/`ln det M` to ~1e-10 AND beat serial CPU REMC at real replica counts AND
  detailed balance intact under mixed precision. [win #2]
- **Phase D — learned-CV OPES world** (mlcolvar CV → OPES bias world in the Gibbs cycle → MBAR).
  Gate: reweighted ΔG within ±0.5 kcal/mol of a reference + ≥10× transitions vs unbiased. [win #3]
- **Phase E (optional, methods contribution) — flow-proposal-in-Metropolis**, scoped to one loop.

## 3. Phased build plan (seeded; sequenced by ROI × risk × MC-safety)

**Phase A — Measure before building (the honest gate).** Instrument the disasm RobotEngine HMC
loop: for real systems at real replica counts, how much wall-time is ABA vs `ln det M` vs the
OpenMM force round-trip vs everything else? *If the round-trip dominates (expected for small
systems), Pillar 3 outranks Pillar 1.* No GPU code until this profile exists.

**Phase B — fp64 batched Cholesky PoC (Pillar 1, lowest risk).** cuSolverDx Cholesky of `M(q)` →
`ln det M` + solve. **Gate:** reproduce the RobotEngine's `M`/`ln det M` to ~1e-10 (fp64); then
show batched over 16–64 replicas it beats the CPU path at that count. If it doesn't beat CPU →
document and stop (correct outcome), don't port more.

**Phase C — (research-informed) either:** custom Warp fp64 CRBA/ABA batched over replicas (if
Phase A says dynamics is the bottleneck), OR an all-on-GPU force loop to kill the round-trip
(if Phase A says forces/round-trip dominate — Pillar 3).

**Phase D — AI sampling layer (Pillar 4), gated on Metropolis-correction** — ML proposals with an
exact accept/reject so mixing improves without breaking the distribution.

## 4. Validation (every phase)
Independent `exp(−βU)` oracle + native-OpenMM cross-check + the disasm Tier-0/1/2 suite must pass.
No change merges to `accel-engine` without its validation green + a JOURNAL entry.

---
*Seeded 2026-07-06 with Pillar 1 (NVIDIA) researched; Pillars 2–4 pending their agents.
Design finalizes when all four land; then Phase A (profiling) starts — measure before we build.*
</content>
