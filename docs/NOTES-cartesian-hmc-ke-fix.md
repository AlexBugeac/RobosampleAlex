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
