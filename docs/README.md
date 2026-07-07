# Robosample documentation

Deterministic, usage-oriented API + call-graph docs for humans and coding agents.
**No AI/LLM is used** — everything is derived from code structure, signatures,
Doxygen call/reference relations, and comments already in the source.

## Quick start

```bash
python3 scripts/generate_docs.py        # Doxygen (HTML+XML+graphs) + Markdown indexes
# or:
cmake --build <builddir> --target docs  # same thing, via CMake
```

Then open `docs/generated/doxygen/html/index.html`, or read the Markdown indexes in
`docs/generated/` — **start with [`generated/PYTHON_API_INDEX.md`](generated/PYTHON_API_INDEX.md)**.

For orientation, read [`../ARCHITECTURE.md`](../ARCHITECTURE.md) and
[`USAGE_CALLGRAPH.md`](USAGE_CALLGRAPH.md) (call graph organized by task).

## What gets generated

| Output | Committed? | Contents |
|---|---|---|
| `generated/*.md` | **yes** (lightweight) | Curated indexes: API, call graph, module, class, Python API. |
| `generated/doxygen/html/` | no (~40 MB, git-ignored) | Full Doxygen site + SVG call/caller/class/include graphs + source browser. |
| `generated/doxygen/xml/` | no (git-ignored) | Machine-readable Doxygen XML (the indexes are built from this). |

Regenerate the Markdown indexes without re-running Doxygen:
`python3 scripts/generate_docs.py --indexes-only`.

## Dependencies

| Tool | Purpose | Install (Arch / Debian) |
|---|---|---|
| **Doxygen** ≥ 1.9 | parses C++, emits HTML + XML | `pacman -S doxygen` / `apt install doxygen` |
| **Graphviz** (`dot`) | call/caller/class/include graphs | `pacman -S graphviz` / `apt install graphviz` |
| **Python 3** | Markdown index generator (stdlib only) | already required by the project |

Optional (not used by the default pipeline): `clang-uml` / `clang-doc` over
`build/*/compile_commands.json` for AST-accurate call graphs on the template-heavy core
(see the trust-boundary note in `../ARCHITECTURE.md`).

## Files in this pipeline

| File | Role |
|---|---|
| `docs/Doxyfile` | Curated Doxygen config — **scoped to `src/` + `include/` only** (vendored `openmm/`, `pybind11/`, … are excluded so they don't swamp the graphs). |
| `scripts/generate_docs.py` | Orchestrator: runs Doxygen, then the index generator. |
| `scripts/gen_doc_indexes.py` | Deterministic Doxygen-XML → Markdown generator (stdlib `ElementTree`). |
| `ARCHITECTURE.md` | Hand-maintained architecture map. |
| `docs/USAGE_CALLGRAPH.md` | Hand-maintained task-oriented call-graph index. |

## Limitations (be aware)

- **Descriptions come from existing comments only.** This codebase has plentiful `//`
  comments but no Doxygen `///`/`@brief` blocks, so the index generator surfaces the
  **verbatim leading `//` comment** per function; where none exists it says *"no comment
  in source"* rather than inventing one. (~130 of ~516 functions currently carry a comment.)
  To make descriptions appear in the Doxygen **HTML** too, a one-time mechanical
  `//` → `///` conversion of leading comments would suffice (optional; not done here to
  avoid touching source).
- **Doxygen call graphs are syntactic** — virtual dispatch, template instantiations, and
  function-pointer/`std::function` callbacks are under-represented. Use the graphs for
  navigation, not as a complete dynamic trace.
