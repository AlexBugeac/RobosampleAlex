# Robosample validation notes — the Cartesian-world HMC acceptance bug (and fix)

_Investigation: does disasm robosample reproduce OpenMM (PE-right), run faster, and is it
correctly configured for its current state? This note records the root-cause bug found and
the fix applied. Author: Claude (Opus 4.8), on Alex's behalf._

## Goal (the three metrics that matter)
1. **PE-right** — the mixed GCHMC protocol must reproduce OpenMM's potential-energy
   distribution (a converged full-ensemble sampler MUST, by the ergodic theorem: same
   stationary distribution ⇒ same P(U)).
2. **Faster** — beat plain MD in wall-clock *on a system where MD is actually trapped*.
3. **Correctly configured for the current program** (not chasing the paper's exact numbers,
   since the code has changed since publication).

## The symptom chain (what led here)
- Mixed GCHMC on alanine-dipeptide (OBC2, 300 K) gave PE **mean ≈ −130 kJ/mol vs OpenMM
  ≈ −100** — a persistent ~30 kJ/mol offset that **no tuning moved** (Cartesian mdSteps
  20→300, world order, run length all left it at ~0.55–0.60 of thermal bond+angle energy).
- Config (αR↔β) trapping was a *separate* problem, fixed by long torsional moves
  (mdSteps≈1000 → barrier crossing). PE offset persisted even then.
- An adversarial 3-agent audit of a robosample-vs-OpenMM "convergence race" flagged the
  Cartesian world as running **broken at ~22% acceptance** (from `race.moves.csv`).

## ROOT CAUSE (confirmed in code + data)
`World::generateSample()`, Cartesian branch (`src/World.cpp`): the fully-flexible Cartesian
world runs a **symplectic Verlet** trajectory with thermal velocities, then accepted/rejected
using **potential energy only**:

```cpp
bridge_.setVelocitiesToTemperature(...);          // draws Maxwell-Boltzmann KE
bridge_.integrateTrajectoryOnDevice(...);         // Verlet MD
const double peNew = bridge_.calcPotentialEnergy();
state_.energy.ke = 0.0;                            // "device kinetic energy not pulled back"
if (metropolis(peOld, peNew)) { ... }             // <-- accept on ΔPE, NOT ΔH
```

That is **not** HMC. Correct HMC accepts on Δ**H** = Δ(PE+KE). Verlet ≈ conserves H, so
ΔPE ≈ −ΔKE; accepting on ΔPE alone **rejects exactly the PE-uphill moves whose KE dropped
to conserve H**. Consequences, all observed:
- ~15% Cartesian acceptance regardless of move length (measured in `moves.csv`: `world` col,
  `type=cartesian`, `accepted` col → 14–17%; torsional world 82–92%).
- `KE=0, fixman=0` on every Cartesian row (code literally sets `ke=0.0`).
- The move behaves as a **PE-downhill filter → low-PE-biased ensemble** → bonds/angles pinned
  near their minima → stiff DOF ~60% thermalized → PE stuck ~30 kJ/mol low.
- The Cartesian world is the component that thermalizes the stiff DOF and restores the full
  ensemble, so it was **structurally unable** to make PE right — hence the whole-session gap.

Integrator confirmed `VerletIntegrator` (`OpenMMContext.cpp:603`; MTS variant :599) — both
symplectic, so H-based HMC acceptance is correct.

## THE FIX (applied)
Accept on the full Hamiltonian, pulling KE back from the device (it was already available —
`OpenMMContext::integrateTrajectory` sets `kineticEnergy = state.getKineticEnergy()`; the
World code just discarded it). Four files:
1. `include/OpenMMContext.hpp` — declare `calcKineticEnergy()`.
2. `src/OpenMMContext.cpp` — define `calcKineticEnergy()` (fresh `getState(Energy)` → KE).
3. `include/ForceBridge.hpp` — `calcKineticEnergy()` passthrough.
4. `src/World.cpp` (Cartesian branch) — record `keOld` after the velocity draw and `keNew`
   after integration; `metropolis(peOld+keOld, peNew+keNew)`; set `state_.energy.ke=keNew`,
   `total=peNew+keNew`.

## Verification plan (the decisive test)
Rebuild (memory-capped, protecting the 3 running E2 sims), then re-run mixed GCHMC on
alanine-dipeptide and check:
- Cartesian acceptance jumps from ~15% toward ~90% (healthy HMC).
- Bond+angle energy reaches ~thermal (was ~0.6).
- **PE histogram slides onto OpenMM's** (the ~30 kJ/mol offset closes). ← metric #1.

## Open items
- **Faster:** the per-step CPU↔GPU round-trip is the throughput cost; the disasm
  `setCudaKinematics` fused path addresses it but is **not yet bound to Python** — needs a
  pybind binding + rebuild to test.
- **A valid speed race** requires a genuinely trapped transition (αL/φ=0 or a higher barrier),
  paper-style 10:1 ratio, ground-truth reference, effective-sample / per-force-eval metrics,
  multiple seeds — per the audit panel's recommendations.

## Methodological caveats recorded (lessons this session)
- The reject marker in the `[hmc]` log is lowercase `-> reject` (uppercase `-> ACCEPT`);
  grepping `REJECT` returns 0 and gives false "100% acceptance". Use `moves.csv` for
  per-world acceptance (it covers the Cartesian world, which emits no `[hmc]` lines).
- PE-histogram comparison is only valid for the MIXED (full-ensemble) protocol, not
  torsional-only (constrained ensemble genuinely has a different P(U)). For the mixed
  protocol it is a legitimate, stringent convergence test.

## Verification — acceptance level (PASSED)
After rebuild, cartesian-only alanine (dt=1.5fs, mdSteps=40):
- **Cartesian acceptance: 15% → 76%** (healthy HMC).
- **KE now recorded**: ~100–140 kJ/mol per move (was 0.0).
Next: confirm the PE histogram of the MIXED protocol closes onto OpenMM (metric #1).

## Verification — PE level (PASSED — metric #1 ACHIEVED)
Mixed GCHMC (torsional + fixed Cartesian) on alanine-dipeptide, 1500 rounds, after the fix:
- torsional acceptance 94%, **cartesian acceptance 72%** (was ~15%).
- **robosample PE: mean -104.0, std 15.4**  vs OpenMM mean ~-100, std ~15.1.
- Before the fix it was mean -130, std 8.8 (offset ~30 kJ/mol, too narrow).
=> The ~30 kJ/mol PE offset is CLOSED; mean AND width now match OpenMM. The mixed
protocol reproduces OpenMM's P(U). The entire PE gap was the PE-only acceptance bug.

Remaining: metric #2 (faster) needs the setCudaKinematics python binding + a properly
trapped system + the audit-compliant race design.

## Metric #2 (faster) — step 2a: fused CUDA kinematics
`ROBO_CUDA_KINEMATICS=1` enables the disasm fused GPU robot-kinematics path (env-controlled,
no binding/rebuild needed; `setCudaKinematics` only overrides the env default). Torsional
ala, mdSteps=200: OFF 342 ms/round -> ON 251 ms/round = ~1.36x, sampling unchanged. It kills
the per-atom host round-trip (the auditors' flagged bottleneck). Use it ON for the race.
Honest: still slow per-step vs OpenMM MD, so robosample only wins where MD is TRAPPED -- for
ala that is the alphaL/phi=0 crossing (MD barely visits alphaL). Race built on that.

## Metric #2 (faster) -- the full investigation and why the paper claim is NOT yet validated

### Step 1: robosample-vs-OpenMM barrier-crossing race (C7ax/alphaL escape) -- a CATEGORY ERROR
Built a fair race: start both engines INSIDE C7ax/alphaL (phi~+60, structure at
`examples/ala-dipeptide-c7ax.rst7`, built by `tests/make_c7ax_start.py`: restrain phi->+60
psi->-70, minimize, 50 ps relax -> phi=63 psi=-67), race the ESCAPE across phi=0. Metric =
crossings per wall-second AND per force-evaluation, 300 K, OBC2, CUDA kinematics ON.
Result (seed1, decisive): **OpenMM 96 crossings, robosample 0** -- MD reached the ~40% alphaL
equilibrium, robosample stayed trapped. OpenMM faster 192x/wall-s, 7.9x/force-eval.
Tuning check (`tests/robo_escape_tuned.py`, mdSteps=1500 + beefy Cartesian): robosample DID
cross once (genuine phi=0 vault +69->-84 into alphaR, NOT a +-180 wrap), confirming the 0 was
partly a tuning artifact -- but still 192x/8x behind MD. NOTE: run exited with CUDA teardown
errors (array-deletion on context destruction, AFTER the result printed -> GPU-cleanup bug in
the fused-kinematics shutdown path; benign to the result, worth filing).

**Why this was the wrong test:** read the actual papers (`references/papers/spiridon_2017_cdhmc_gibbs`,
`spiridon_2020_robosample` -- paper.md has full content, checks.md has regression numbers).
EVERY efficiency claim in both papers is MIXED-world vs FULLY-FLEXIBLE *within robosample*;
OpenMM only supplies forces and is NEVER the baseline. The headline (2020 Table 4) is alphaL
phi=0-crossing MFPT measured in MD-STEPS (per integrator step), Rama-mixed vs fully-flexible:
flex ~491, mixed-TD ~49 (units steps/200) => ~10x. 2017 sec 3.4: "per integrator step, MIXED
more efficient than UDHMC." So "is robosample faster than OpenMM in wall-clock" is a question
the paper never asks. The paper even predicts the 300 K trap: "torsional dynamics may be less
efficient at low temperature ... combination with replica exchange" (2017 future work).

### Step 2: the paper's ACTUAL claim (vacuum alphaL MFPT, mixed vs flexible) -- CENSORED/CENSORED
`tests/vacuum_mfpt.py`: vacuum, start in alphaL, measure integrator-steps to first phi=0
crossing, fully-flexible (108x1.87fs) vs Rama-mixed (1 flex + 10 torsional 11x44.73fs, the
paper's tuned params), 3 seeds. Paper target (Table 4 alphaL row /200): flex ~491, mixed ~49.
RESULT: **both censored** -- fully-flexible all 3 seeds hit 432k steps with 0 crossings
(phi explored 16-99, stuck in alphaL well, psi ranged fully -149..79 so sampler works);
Rama-mixed hit 130k steps (600 rounds) with 0 crossings (85% torsional acceptance, phi 49-90).
Cannot compute a ratio.

Three compounding reasons it did not reproduce (NOT a refutation of the paper):
1. **Wrong force field.** prmtop is **ff19SB** (has CMAP, XC atom type). Paper used ff12SB
   (2017) / ff14SB (2020), NO CMAP. CMAP reshapes the phi/psi landscape and alphaL depth.
   Proof it differs: all 3 flex seeds censored at 432k = 4.4x the paper's 98k-step MFPT;
   P(0 crossings)=e^-4.4 per seed, ~2e-6 for all three if the FF matched.
2. **Constraint solver fails at the paper's own tuned 44.73 fs step.** disasm velocity-corrector
   does NOT converge (relative change up to ~22 >> tol 1e-4, capped at 10 iters). Step still
   taken (trajectory-Metropolis keeps correctness) but torsional PROPOSALS are degraded -> phi
   never reaches the TS. Original Simbody presumably converged here. **Likely a disasm
   constraint-solver regression at large dt -- the most actionable branch finding.**
3. **Step budget 100x too small:** ~1e5 steps/run vs the paper's 1e7.

### What a real validation requires (not yet done)
- Rebuild alanine with the paper's FF (ff12SB or ff14SB, NO CMAP) so alphaL MFPT ~ the paper's.
- Fix/relax the disasm constraint tolerance OR retune the torsional step DOWN to where the
  solver converges, then re-tune to ~0.651 acceptance (paper's optimal).
- Run to ~1e7 steps/run, 3+ seeds; compare mixed vs fully-flexible MFPT in MD-steps -> expect ~10x.
- Cheaper correctness fixtures worth doing first: butane Shirts test (slope 0.13363; MIXED-TD
  got 0.13357; GAFF/AM1-BCC 300+450 K) and the 4-bead uniform-torsion test (Fixman on/off).

### Honest scorecard for the disasm branch this session
- **PE-correct (metric #1): VALIDATED** -- the KE/HMC-acceptance bug fix (above) is real and holds.
- **Throughput: 1.36x** fused CUDA kinematics (`ROBO_CUDA_KINEMATICS=1`) -- real.
- **Paper's sampling speedup: NOT validated, NOT refuted** -- setup mismatch (FF, solver-at-large-dt,
  step budget). Two concrete branch bugs surfaced: (a) constraint solver non-convergence at 44.73 fs;
  (b) CUDA array-teardown errors on context destruction in the fused-kinematics path.
- Test scripts: `tests/make_c7ax_start.py`, `tests/trapped_race.py`, `tests/robo_escape_tuned.py`,
  `tests/vacuum_mfpt.py`, `tests/timed_robo.py`.
