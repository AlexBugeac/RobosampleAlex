#!/usr/bin/env python3
"""Render a class-level architecture / call-dependency graph of Robosample as PNG.

Data-driven: aggregates the member-level call edges from the Doxygen XML up to
CLASS level (caller class -> callee class, weighted by number of call sites),
keeps the core robosample classes, colours them by role, and renders with dot.
Deterministic; no LLM.

Usage: python3 scripts/gen_arch_graph.py [--xml DIR] [--out PNG]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# role -> (fill, font colour); order also controls the rough top-to-bottom layout
ROLE = {
    "entry":   ("#eef2ff", "#3730a3"),
    "orchestr":("#4c6ef5", "#ffffff"),
    "model":   ("#5f3dc4", "#ffffff"),
    "dynamics":("#2b8a3e", "#ffffff"),
    "forces":  ("#e8590c", "#ffffff"),
    "state":   ("#495057", "#ffffff"),
    "io":      ("#9c36b5", "#ffffff"),
    "math":    ("#0c8599", "#ffffff"),
}
# curated core classes -> (role, label)
CORE = {
    "Context":       ("orchestr", "Context\\norchestrator"),
    "World":         ("model",    "World\\nconstraint set / hub"),
    "RobotEngine":   ("dynamics", "RobotEngine\\nO(n) articulated-body\\n+ Fixman metric"),
    "RobotIntegrator":("dynamics","RobotIntegrator\\nHMC / velocity-Verlet"),
    "MTSIntegrator": ("dynamics", "MTSIntegrator\\nmulti-time-step"),
    "OpenMMContext": ("forces",   "OpenMMContext\\nOpenMM force server"),
    "RobotState":    ("state",    "RobotState\\ngeneralized-coord state"),
    "SystemTopology":("state",    "SystemTopology\\nAMBER topology"),
    "ConstraintSet": ("state",    "ConstraintSet\\nloop constraints"),
    "NMA":           ("dynamics", "NMA\\nnormal modes"),
    "Writer":        ("io",       "Writer / dcd\\ntrajectory output"),
    # articulated-body math layer (robot_math.hpp) — the calls into these value-types
    # are what Clang's AST resolves but Doxygen's syntactic graph misses.
    "Mat33":         ("math",     "Mat33"),
    "SymMat33":      ("math",     "SymMat33"),
    "Quat":          ("math",     "Quat"),
    "Vec3":          ("math",     "Vec3"),
    "SpatialVec":    ("math",     "SpatialVec"),
    "Transform":     ("math",     "Transform"),
    "ArticulatedInertia": ("math","ArticulatedInertia"),
    "Inertia":       ("math",     "Inertia"),
}
# Architectural relationships that Doxygen's syntactic call graph does NOT resolve
# (inline/header/ownership). Rendered DASHED to distinguish from measured call edges.
# Grounded in ARCHITECTURE.md, not invented.
STRUCT_EDGES = [
    ("World", "RobotIntegrator"),   # World::add_sampler attaches the HMC sampler
    ("World", "MTSIntegrator"),     # alternative integrator
    ("RobotIntegrator", "RobotEngine"),   # sampler drives the articulated-body dynamics
    ("RobotIntegrator", "RobotState"),    # …updating generalized-coord state
    ("World", "Writer"),            # trajectory output during run_rex
    ("World", "NMA"),               # normal-mode / soft-mode analysis
]


def compound_names(xml_dir: Path):
    idx = ET.parse(xml_dir / "index.xml").getroot()
    return {c.get("refid"): c.findtext("name") for c in idx.findall("compound")}


def class_of_refid(refid: str, names: dict) -> str | None:
    """Map a member refid (classWorld_1a…) to its class short-name (World)."""
    if not refid:
        return None
    comp = refid.split("_1", 1)[0]
    full = names.get(comp)
    if not full:
        return None
    return full.split("::")[-1]


def build_edges(xml_dir: Path, names: dict):
    edges = defaultdict(int)
    for cf in xml_dir.glob("*.xml"):
        if cf.name in ("index.xml", "Doxyfile.xml"):
            continue
        cd = ET.parse(cf).getroot().find("compounddef")
        if cd is None or cd.get("kind") not in ("class", "struct"):
            continue
        caller_cls = (cd.findtext("compoundname") or "").split("::")[-1]
        if caller_cls not in CORE:
            continue
        for md in cd.findall(".//memberdef[@kind='function']"):
            for r in md.findall("references"):
                callee = class_of_refid(r.get("refid", ""), names)
                if callee in CORE and callee != caller_cls:
                    edges[(caller_cls, callee)] += 1
    return edges


ALWAYS = {"Context", "World", "RobotEngine", "OpenMMContext", "RobotState",
          "RobotIntegrator", "SystemTopology"}


def to_dot(edges, source="Doxygen XML") -> str:
    # only render CORE nodes that participate in an edge (plus the always-on spine),
    # so the math layer doesn't leave orphans in the Doxygen view.
    struct = [(a, b) for a, b in STRUCT_EDGES if (a, b) not in edges and a in CORE and b in CORE]
    used = set(ALWAYS)
    for (a, b) in list(edges) + struct:
        if a in CORE and b in CORE:
            used.add(a); used.add(b)
    out = ["digraph robosample {",
           '  rankdir=TB; bgcolor="white"; pad=0.4; nodesep=0.5; ranksep=0.75;',
           '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
           'fontsize=11, penwidth=0, margin="0.18,0.10"];',
           '  edge [color="#adb5bd", arrowsize=0.8, penwidth=1.2];',
           '  labelloc="t"; fontname="Helvetica-Bold"; fontsize=16;',
           '  label=<Robosample — class-level architecture<br/>'
           '<font point-size="10">solid = measured call edges (%s, width = # call sites) &#183; '
           'dashed = architectural ownership</font>>;' % source,
           "",
           '  py [label="import robosample\\nContext(...)", shape=note, '
           'style="filled", fillcolor="%s", fontcolor="%s", fontname="Helvetica-Bold"];'
           % ROLE["entry"]]
    for cls, (role, label) in CORE.items():
        if cls not in used:
            continue
        fill, font = ROLE[role]
        out.append(f'  {cls} [label="{label}", fillcolor="{fill}", fontcolor="{font}"];')
    out.append('  py -> Context [color="#3730a3", penwidth=2.0];')
    edges = {(a, b): w for (a, b), w in edges.items() if a in used and b in used}
    if edges:
        mx = max(edges.values())
        for (a, b), w in sorted(edges.items(), key=lambda kv: -kv[1]):
            pen = 1.0 + 3.5 * (w / mx)
            out.append(f'  {a} -> {b} [penwidth={pen:.1f}, label="{w}", '
                       f'fontsize=8, fontcolor="#868e96"];')
    # dashed architectural edges (skip any already present as a measured edge)
    for a, b in struct:
        if a in used and b in used:
            out.append(f'  {a} -> {b} [style=dashed, color="#ced4da", '
                       f'arrowsize=0.7, penwidth=1.1];')
    # legend
    out.append('  subgraph cluster_legend {')
    out.append('    label="roles"; fontsize=10; color="#dee2e6"; style="rounded";')
    for role, (fill, font) in ROLE.items():
        if role == "entry":
            continue
        out.append(f'    L_{role} [label="{role}", fillcolor="{fill}", '
                   f'fontcolor="{font}", fontsize=9, width=1.1, height=0.3];')
    order = [r for r in ROLE if r != "entry"]
    out.append("    " + " -> ".join(f"L_{r}" for r in order) + " [style=invis];")
    out.append("  }")
    out.append("}")
    return "\n".join(out)


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="docs/generated/doxygen/xml", type=Path)
    ap.add_argument("--out", default="docs/generated/architecture-graph.png", type=Path)
    ap.add_argument("--edges-json", default=None, type=Path,
                    help="use class edges from JSON (e.g. clang) instead of Doxygen XML")
    ap.add_argument("--source", default="Doxygen XML", help="label for the edge source")
    a = ap.parse_args()
    if a.edges_json:
        raw = json.load(open(a.edges_json))
        edges = {}
        for k, w in raw.items():
            x, y = k.split("->", 1)
            edges[(x, y)] = w
        dot = to_dot(edges, source=a.source)
    else:
        if not (a.xml / "index.xml").exists():
            sys.exit("run Doxygen first (python3 scripts/generate_docs.py)")
        names = compound_names(a.xml)
        edges = build_edges(a.xml, names)
        dot = to_dot(edges, source=a.source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dotfile = a.out.with_suffix(".dot")
    dotfile.write_text(dot)
    r = subprocess.run(["dot", "-Tpng", "-Gdpi=150", str(dotfile), "-o", str(a.out)])
    if r.returncode != 0:
        sys.exit("dot failed")
    print(f"wrote {a.out}  ({len(edges)} class-level edges over {len(CORE)} core classes)")


if __name__ == "__main__":
    main()
