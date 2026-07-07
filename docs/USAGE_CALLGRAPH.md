# Usage-Oriented Call Graph Index

The code organized by **what you're trying to do**, not alphabetically. Each section
lists the entry points, the key files/functions, and links into the generated Doxygen
graphs. For the raw per-function call edges see
[`docs/generated/CALLGRAPH_INDEX.md`](generated/CALLGRAPH_INDEX.md); for the Python
surface see [`docs/generated/PYTHON_API_INDEX.md`](generated/PYTHON_API_INDEX.md).

Robosample has **no CLI `main()`** — the entry point is the Python module.

---

## Entry point — building and running a simulation

**Entry:** `import robosample; robosample.Context(baseName, seed)` → `Context`
(`src/Context.cpp`). Bindings: `src/PyBind11.cpp`.

Typical flow (each step links to the class it drives):

1. `Context::Context` — construct orchestrator
2. `Context::loadAmber` (`load_amber`) — parse topology → `SystemTopology`
3. `Context::build_flexibilities` — choose mobile joints → `Selection`
4. `Context::addRoboticWorld` / `addCartesianWorld` / `addDockingWorld` — build a `World`
5. `World::add_sampler` — attach an HMC sampler (`RobotIntegrator`)
6. `Context::initialize` — internal-coordinate velocity init
7. `Context::run_rex` — the replica-exchange / Gibbs production loop

Graphs: `docs/generated/doxygen/html/classContext.html` (collaboration graph),
and the call graphs on `Context::run_rex` and `Context::addRoboticWorld`.

---

## Input parsing — topology & coordinates

**Important files:** `src/Context.cpp` (loader entry), `include/TopologyElements.hpp`,
the `SystemTopology` bindings in `src/PyBind11.cpp` (`atoms_*`, `bonds_*`, `angles_*`).

**Key functions:** `Context::loadAmber` (AMBER `prmtop`/`rst7`, optional GBSA-OBC2);
the `SystemTopology` accessors expose per-atom (`atoms_x/…/charge/sigma/epsilon`),
per-bond (`bonds_i/j/equilibrium`), and per-angle fields. See the `fast-amber-loader`
and `loader-feature-matrix` specs in `docs/specs/`.

---

## Model construction — worlds, joints, flexibilities

**Important files:** `src/World.cpp` (the hub, `buildModel`), `src/RobotEngine.cpp`
(multibody tree + joints), `include/robot_math.hpp` (articulated-body math).

**Key functions:** `Context::build_flexibilities` (which bonds become `Torsion`/other
joints), `World::buildModel` (assembles the multibody model + force bridge),
`Context::addRoboticWorld` / `addCartesianWorld` / `addDockingWorld`. Each **World** is one
constraint set; mixing worlds is what makes the sampling ergodic (Gibbs).

Graphs: `classWorld.html` (collaboration graph — shows the `RobotEngine` + `OpenMMContext`
it owns), and `World::buildModel`'s call graph.

---

## Core computation — dynamics, forces, acceptance

**Important files:** `src/RobotEngine.cpp` (O(n) articulated-body dynamics + Fixman
mass-matrix determinant / torque), `src/OpenMMContext.cpp` + `include/ForceBridge.hpp`
(per-step force server), `include/RobotIntegrator.hpp` (HMC / velocity-Verlet),
`include/RobotState.hpp` (generalized-coordinate state).

**Key functions:** the sampler move (propagate constrained dynamics via `RobotEngine`,
pulling forces from `OpenMMContext` each step) and the Metropolis accept/reject with the
**Fixman** correction. **Invariant:** dynamics use the unmodified potential; Fixman enters
only acceptance (`docs/specs/fixman-idealized-chains-validation.md`,
`docs/specs/singular-dof-fixman.md`).

Graphs: `classRobotEngine.html`, `classOpenMMContext.html`, and the call graph on the
integrator step.

---

## Replica exchange / Gibbs driver

**Important files:** `src/Context.cpp` (`run_rex`), `src/World.cpp` (per-world move).

**Key functions:** `Context::run_rex` — the production loop: per round, run each world's
HMC move, mix worlds (Gibbs), attempt replica exchanges across the temperature ladder, and
write output at `writeFreq`. NCMC configuration: `configure_ncmc*`, `set_ncmc_*` bindings.

---

## Output writing — trajectories & reports

**Important files:** `src/World.cpp` / `namespace dcd` (DCD writer), reaction reporting in
`src/OpenMMContext.cpp` (`enable_reaction_reporter`, `is_reaction_reporter`).

**Key functions:** DCD trajectory writing during `run_rex`; force-group energy reporting via
`OpenMMContext::ForceGroupEnergy`; reaction sampling (`ReactionSample`). See
`docs/specs/reaction-force-monitoring.md` and `docs/specs/robotics-oracle-*.md`.

---

## Analysis helpers

**Important files:** `include/NMA.hpp` (normal-mode analysis; `set_nma_soft_mode_from_hessian`),
`include/robot_math.hpp` (Hessian/inertia math).

---

_Generated Doxygen HTML pages referenced above live under
`docs/generated/doxygen/html/` after running `python3 scripts/generate_docs.py`._
