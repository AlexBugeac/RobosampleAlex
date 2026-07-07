#!/usr/bin/env python3
"""AST-accurate call graph for Robosample via libclang (clang.cindex).

Unlike Doxygen's *syntactic* call graph, this parses the real Clang AST from
`compile_commands.json`, so it resolves calls through templates and (statically)
virtual dispatch that Doxygen misses. Deterministic; no LLM.

Outputs:
  docs/generated/CLANG_CALLGRAPH_INDEX.md   accurate per-function callees/callers
  docs/generated/clang-class-edges.json     class-level edges (for the PNG)
and prints a coverage comparison vs the Doxygen call edges.

"Live" call graph, two ways:
  1. This script — regenerated from the compiler each run (source of truth).
  2. clangd `--background-index` gives live *editor* call-hierarchy (see docs/README).

Usage: python3 scripts/gen_clang_callgraph.py [--cc build/cuda-release/compile_commands.json]
"""
from __future__ import annotations
import argparse
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import clang.cindex as CX

REPO = Path("/home/alexb/Robosample_disasm")
VENDORED = ("/openmm/", "/pybind11/", "/units/", "/pcg-cpp/", "/build/")


def load_libclang():
    for lib in ("/usr/lib/libclang.so.22.1.6", "/usr/lib/libclang.so",
                "/usr/lib/libclang.so.21.1"):
        try:
            CX.Config.set_library_file(lib)
            CX.Index.create()
            return lib
        except Exception:
            continue
    sys.exit("error: could not load a libclang.so")


def safe_kind(cur):
    """cur.kind, but tolerant of libclang/bindings version drift (unknown kind ids)."""
    try:
        return cur.kind
    except ValueError:
        return None


def is_robo(cur) -> bool:
    """True iff a cursor's location is in robosample's own (non-vendored) source."""
    f = cur.location.file
    if not f:
        return False
    p = f.name
    return str(REPO) in p and not any(v in p for v in VENDORED)


def qual(cur) -> str:
    """Class::method or free-function name for a decl cursor."""
    if cur is None:
        return ""
    parent = cur.semantic_parent
    if parent is not None and safe_kind(parent) in (
            CX.CursorKind.CLASS_DECL, CX.CursorKind.STRUCT_DECL,
            CX.CursorKind.CLASS_TEMPLATE):
        return f"{parent.spelling}::{cur.spelling}"
    return cur.spelling


def cls_of(cur) -> str | None:
    parent = cur.semantic_parent if cur else None
    if parent is not None and safe_kind(parent) in (
            CX.CursorKind.CLASS_DECL, CX.CursorKind.STRUCT_DECL,
            CX.CursorKind.CLASS_TEMPLATE):
        return parent.spelling
    return None


DEF_KINDS = (CX.CursorKind.CXX_METHOD, CX.CursorKind.FUNCTION_DECL,
             CX.CursorKind.CONSTRUCTOR, CX.CursorKind.DESTRUCTOR,
             CX.CursorKind.FUNCTION_TEMPLATE, CX.CursorKind.CONVERSION_FUNCTION)


def parse_args_for(entry) -> list[str]:
    toks = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    args, skip = [], False
    for t in toks[1:]:
        if skip:
            skip = False
            continue
        if t == "-c":
            continue
        if t == "-o":
            skip = True
            continue
        if t.endswith((".cpp", ".cc", ".o")):
            continue
        args.append(t)
    return args


def extract(index, entry, edges: dict, members: dict):
    """Walk one TU; record robosample-internal caller->callee edges (deduped by USR)."""
    try:
        tu = index.parse(entry["file"], args=parse_args_for(entry))
    except Exception:
        return
    if tu is None:
        return

    def visit(cur, enclosing):
        if safe_kind(cur) in DEF_KINDS and cur.is_definition() and is_robo(cur):
            enclosing = cur
            usr = cur.get_usr()
            if usr and usr not in members:
                loc = cur.location
                members[usr] = {
                    "name": qual(cur), "cls": cls_of(cur),
                    "file": str(Path(loc.file.name).relative_to(REPO)) if loc.file else "",
                    "line": loc.line,
                }
        if safe_kind(cur) == CX.CursorKind.CALL_EXPR and enclosing is not None:
            ref = cur.referenced
            if ref is not None and is_robo(ref) and safe_kind(ref) in DEF_KINDS:
                cu, ru = enclosing.get_usr(), ref.get_usr()
                if cu and ru and cu != ru:
                    edges.setdefault((cu, ru), 0)
                    edges[(cu, ru)] += 1
                    # ensure callee has a member record even if defined elsewhere
                    if ru not in members:
                        loc = ref.location
                        members[ru] = {
                            "name": qual(ref), "cls": cls_of(ref),
                            "file": str(Path(loc.file.name).relative_to(REPO)) if loc.file else "",
                            "line": loc.line,
                        }
        for ch in cur.get_children():
            visit(ch, enclosing)

    visit(tu.cursor, None)


def doxygen_class_edges(xml_dir: Path):
    """Set of (callerClass, calleeClass) edges Doxygen resolved (for coverage compare)."""
    import xml.etree.ElementTree as ET
    if not (xml_dir / "index.xml").exists():
        return set()
    names = {c.get("refid"): c.findtext("name")
             for c in ET.parse(xml_dir / "index.xml").getroot().findall("compound")}
    edges = set()
    for cf in xml_dir.glob("class*.xml"):
        cd = ET.parse(cf).getroot().find("compounddef")
        if cd is None:
            continue
        caller = (cd.findtext("compoundname") or "").split("::")[-1]
        for md in cd.findall(".//memberdef[@kind='function']"):
            for r in md.findall("references"):
                comp = (r.get("refid") or "").split("_1", 1)[0]
                callee = (names.get(comp) or "").split("::")[-1]
                if caller and callee:
                    edges.add((caller, callee))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default="build/cuda-release/compile_commands.json")
    ap.add_argument("--out", default="docs/generated", type=Path)
    a = ap.parse_args()
    lib = load_libclang()
    index = CX.Index.create()
    cc = json.load(open(REPO / a.cc))
    tus = [e for e in cc if (REPO / "src") == Path(e["file"]).parent]  # robosample TUs only
    print(f"libclang {lib.split('/')[-1]} — parsing {len(tus)} robosample translation units…")

    edges: dict = {}
    members: dict = {}
    for i, e in enumerate(tus, 1):
        print(f"  [{i}/{len(tus)}] {Path(e['file']).name}")
        extract(index, e, edges, members)

    # member-level callees/callers
    callees = defaultdict(list)
    callers = defaultdict(list)
    for (cu, ru), n in edges.items():
        callees[cu].append((ru, n))
        callers[ru].append((cu, n))

    # class-level edges
    cls_edges: dict = defaultdict(int)
    for (cu, ru), n in edges.items():
        ca, cb = members.get(cu, {}).get("cls"), members.get(ru, {}).get("cls")
        if ca and cb and ca != cb:
            cls_edges[(ca, cb)] += n

    # ---- write markdown ----
    lines = [f"# Clang (AST-accurate) Call Graph Index\n",
             f"_Auto-generated by `scripts/gen_clang_callgraph.py` via libclang from "
             f"`{a.cc}`. Resolves template + statically-virtual calls that Doxygen's "
             f"syntactic graph misses. {len(edges)} internal call edges over "
             f"{len(members)} functions._\n"]
    for usr, m in sorted(members.items(), key=lambda kv: (kv[1]["cls"] or "", kv[1]["name"])):
        ce, cr = callees.get(usr, []), callers.get(usr, [])
        if not ce and not cr:
            continue
        lines.append(f"\n### `{m['name']}`  \n`{m['file']}:{m['line']}`")
        if ce:
            lines.append("- **calls →** " + ", ".join(
                f"`{members.get(r,{}).get('name','?')}`" + (f"×{n}" if n > 1 else "")
                for r, n in sorted(ce, key=lambda x: -x[1])))
        if cr:
            lines.append("- **← called by** " + ", ".join(
                f"`{members.get(c,{}).get('name','?')}`" for c, _ in cr))
    (a.out / "CLANG_CALLGRAPH_INDEX.md").write_text("\n".join(lines))

    # ---- class edges json (for the PNG) ----
    (a.out / "clang-class-edges.json").write_text(
        json.dumps({f"{ca}->{cb}": n for (ca, cb), n in cls_edges.items()}, indent=0))

    # ---- coverage comparison vs Doxygen ----
    dox = doxygen_class_edges(REPO / "docs/generated/doxygen/xml")
    clang_set = set(cls_edges.keys())
    new = clang_set - dox
    print(f"\nclass-level call edges:  clang={len(clang_set)}  doxygen={len(dox)}")
    print(f"edges clang found that Doxygen missed: {len(new)}")
    if new:
        for ca, cb in sorted(new)[:12]:
            print(f"    + {ca} -> {cb}")
    print(f"\nwrote {a.out}/CLANG_CALLGRAPH_INDEX.md and clang-class-edges.json")


if __name__ == "__main__":
    main()
