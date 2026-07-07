# Robosample Architecture (disasm branch)

A concise map of the codebase for humans and coding agents. For the exhaustive,
auto-generated reference (every function, signature, call graph) see
[`docs/generated/`](docs/generated/) — start with
[`PYTHON_API_INDEX.md`](docs/generated/PYTHON_API_INDEX.md). Regenerate with
`python3 scripts/generate_docs.py`.

Robosample is a **rigid-body / internal-coordinate molecular simulator**: it samples
conformations with **Constrained-Dynamics Hamiltonian Monte Carlo (CDHMC)** used as a
**Gibbs move**, borrowing O(n) articulated-body ("robot mechanics") dynamics and using
**OpenMM as a per-step force server**. The theory lives in `references/papers/`
(esp. `spiridon_2020_robosample`, `spiridon_2017_cdhmc_gibbs`); this document is about
the *code*.

> **Scope note.** The repo vendors its dependencies in-tree — `openmm/` (~410k LOC),
> `pybind11/` (~47k), `units/`, `pcg-cpp/`. Robosample's own source is **~7k LOC in
> `src/` + `include/`**. All navigation below (and the doc pipeline) concerns *that*.

---

## 1. Execution flow (the pipeline)

The entry point is the **Python module**, not a CLI `main()` — `import robosample`
exposes the classes bound in `src/PyBind11.cpp`. A typical run:

```
import robosample
ctx = robosample.Context(baseName, seed)          # Context.cpp — orchestrator
ctx.load_amber(prmtop, rst7, use_gbsa_obc2=…)     # -> SystemTopology (atoms/bonds/angles)
sele = ctx.build_flexibilities(pairs, JointType)  # choose which bonds are mobile joints
ctx.add_cartesian_world()                         # World.cpp — one constraint set …
ctx.add_robotic_world(sele)                       #   … (torsional / docking / cartesian)
world.add_sampler(timeStep, mdSteps, …,           # RobotIntegrator — an HMC sampler
                  use_fixman=True)
ctx.initialize([temperatures])                    # internal-coordinate velocity init
ctx.run_rex(equil, prod, writeFreq, …)            # replica-exchange / Gibbs driver loop
```

Per production round, `run_rex` runs each **World's** HMC move — propagate constrained
dynamics on the *unmodified* potential via the **RobotEngine** (getting forces from the
**OpenMMContext** force server each step), then accept/reject with the **Fixman**
correction folded into the Metropolis criterion — then mixes worlds (Gibbs, to restore
ergodicity) and exchanges replicas, writing DCD trajectories.

This mirrors the method exactly: alternating constraint sets = Gibbs sampling; O(n)
articulated-body dynamics = the RobotEngine; Fixman only in acceptance = the CDHMC design.

---

## 2. Directory responsibilities

| Path | Responsibility |
|---|---|
| `src/` (~7k LOC) | Robosample's own C++ implementation. |
| `include/` | Public headers for the above. |
| `src/PyBind11.cpp` | **The entire Python API** — every `robosample.*` name is bound here. |
| `docs/specs/` | Hand-written design specs & validation invariants (see §5). |
| `docs/generated/` | Auto-generated API/call-graph indexes + Doxygen HTML/XML (see §6). |
| `references/papers/` | Literature the method is built on (theory, not code). |
| `tests/` | Validation suite (kinetic invariants, PE ladder, torsion benchmarks). |
| `examples/` | Example systems (butane, alanine-dipeptide, …). |
| `openmm/`, `pybind11/`, `units/`, `pcg-cpp/` | **Vendored dependencies** — not robosample source; excluded from docs. |

---

## 3. Major classes / modules (what owns what)

| Class (file) | Owns / does |
|---|---|
| **`Context`** (`Context.cpp`) | Top-level orchestrator. Holds the `SystemTopology` and the list of `World`s; exposes `load_amber`, `build_flexibilities`, `add_*_world`, `initialize`, `run_rex`. This is `robosample.Context`. |
| **`World`** (`World.cpp`, ~2.5k LOC — the hub) | One **constraint set / move type** (cartesian, torsional, docking). Builds its multibody model (`buildModel`), owns a `RobotEngine`, an `OpenMMContext`, and its sampler(s) via `add_sampler`. |
| **`RobotEngine`** (`RobotEngine.cpp`) | The **internal-coordinate (articulated-body) dynamics** — O(n) Featherstone/Jain kinematics + the Fixman mass-matrix determinant/torque. Uses the math in `robot_math.hpp`. |
| **`OpenMMContext`** (`OpenMMContext.cpp`, `ForceBridge.hpp`) | The **force server** — wraps OpenMM to return energies/forces per MD step, and force-group energies. |
| **`RobotState`** (`RobotState.hpp`) | Simulation state in generalized coordinates (positions/velocities/momenta). |
| **`RobotIntegrator`** (`RobotIntegrator.hpp`), **`MTSIntegrator`** | HMC / velocity-Verlet integrators over the robot dynamics; multiple-time-step variant. |
| **`SystemTopology`** / `TopologyElements.hpp` | The loaded AMBER topology (atoms/bonds/angles); exposed via the `atoms_*`/`bonds_*`/`angles_*` bindings. |
| **`ConstraintSet`** | Loop / distance constraints for non-tree topologies. |
| **`NMA`** (`NMA.hpp`) | Normal-mode analysis (soft modes; `set_nma_soft_mode_from_hessian`). |
| `Writer`, `namespace dcd` | Trajectory (DCD) output. |
| `robot_math.hpp` | Articulated-body math: `Mat33`, `Quat`, `ArticulatedInertia`, `PhiMatrix`, `Inertia`, `MassProperties`. |

Full per-class member lists + graphs: [`CLASS_INDEX.md`](docs/generated/CLASS_INDEX.md),
[`MODULE_INDEX.md`](docs/generated/MODULE_INDEX.md).

---

## 4. Where to add new functionality

| To add… | Touch |
|---|---|
| A new **joint type** | `RobotEngine` (kinematics) + the `JointType` enum + `Context::build_flexibilities`. |
| A new **world / move type** | `World` (`buildModel`) + `Context::add*World` + a binding in `PyBind11.cpp`. |
| A new **sampler / integrator** | `RobotIntegrator` (or `MTSIntegrator`) + `World::add_sampler`. |
| A new **force term / group** | `OpenMMContext` / `ForceBridge`. |
| New **trajectory / report output** | `Writer` / `namespace dcd`. |
| Exposing anything to **Python** | add a `.def(...)` in `src/PyBind11.cpp` (single file). |

---

## 5. Invariants & design constraints (see specs — not duplicated here)

The load-bearing correctness constraints are specified in `docs/specs/` — treat these as
authoritative before changing the dynamics or acceptance path:

- **Fixman metric correctness** (unbiased Boltzmann under constraints):
  `docs/specs/fixman-idealized-chains-validation.md`, `docs/specs/singular-dof-fixman.md`.
- **Ensemble validation** (Tier-0 kinetic/equipartition invariants; Tier-1 PE-distribution
  ladder vs OpenMM): `docs/specs/ensemble-validation/`.
- **Reaction-force / oracle correctness:** `docs/specs/robotics-oracle-*.md`,
  `docs/specs/reaction-force-monitoring.md`.
- **MC exactness rule (from the method):** dynamics run on the *unmodified* potential; the
  Fixman potential enters *only* the accept/reject step. Do not fold Fixman into the
  propagator.

---

## 6. Generated documentation (how to navigate deeper)

Run `python3 scripts/generate_docs.py`, then:

- [`docs/generated/PYTHON_API_INDEX.md`](docs/generated/PYTHON_API_INDEX.md) — the
  Python-callable surface → C++ target → docs. **Start here.**
- [`docs/generated/API_INDEX.md`](docs/generated/API_INDEX.md) — every function: signature,
  file:line, verbatim source comment, callees/callers, HTML link.
- [`docs/generated/CALLGRAPH_INDEX.md`](docs/generated/CALLGRAPH_INDEX.md) — cross-cutting
  call graph (who calls what).
- [`docs/USAGE_CALLGRAPH.md`](docs/USAGE_CALLGRAPH.md) — call graph organized **by task**.
- `docs/generated/doxygen/html/index.html` — full Doxygen site with SVG call/caller/class/
  include graphs and a cross-linked source browser.

> **Trust boundary for the call graphs.** Doxygen's call/caller graphs are *syntactic* —
> they resolve calls by name/signature, so **virtual dispatch, template instantiations, and
> function-pointer/`std::function` callbacks are under-represented**. Treat the graphs as a
> navigation aid, not a complete dynamic trace. For ground-truth call edges on the
> template-heavy core, a Clang-AST tool (`clang-uml` / `clang-doc`) over the existing
> `build/*/compile_commands.json` is the escalation path — intentionally left out of the
> default pipeline to keep it simple.
