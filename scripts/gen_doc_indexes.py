#!/usr/bin/env python3
"""Deterministic Doxygen-XML -> curated Markdown index generator for Robosample.

NO LLM / NO AI. Everything emitted here is derived mechanically from:
  * the Doxygen XML (signatures, file:line, call/caller relations, class/namespace
    ownership, HTML anchors), and
  * verbatim comments already present in the source (leading `//` blocks), and
  * the pybind11 binding table parsed from src/PyBind11.cpp.

It writes, under docs/generated/:
  API_INDEX.md        - every documented function: signature, file:line, verbatim
                        source comment, callees, callers, owning class/namespace,
                        link to the Doxygen HTML page.
  CALLGRAPH_INDEX.md  - the cross-cutting call graph: per function, who it calls and
                        who calls it (the view no off-the-shelf tool emits).
  MODULE_INDEX.md     - source files/classes as modules, with their members + links.
  CLASS_INDEX.md      - classes/structs with their public members.
  PYTHON_API_INDEX.md - the Python-callable surface (robosample.*) mapped to the
                        underlying C++ target and its documentation.

Purpose/preconditions/intent are NEVER invented: description text is only the
Doxygen brief/detailed (usually empty in this repo) plus the verbatim source
comment. Absence of a comment is reported as such.

Usage:  python3 scripts/gen_doc_indexes.py [--xml DIR] [--out DIR] [--repo DIR]
"""
from __future__ import annotations
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---- links -----------------------------------------------------------------
# HTML lives at docs/generated/doxygen/html/; the Markdown indexes at
# docs/generated/. So links are relative: doxygen/html/<page>.
HTML_REL = "doxygen/html"


def html_link(refid: str, is_member: bool) -> str:
    """Map a Doxygen refid to its generated HTML URL.

    Compound refid  -> <refid>.html
    Member refid    -> <compound>.html#<anchor>, split on the '_1' delimiter.
    """
    if not refid:
        return ""
    if is_member and "_1" in refid:
        compound, anchor = refid.split("_1", 1)
        return f"{HTML_REL}/{compound}.html#{anchor}"
    return f"{HTML_REL}/{refid}.html"


# ---- mixed-content description flattening ----------------------------------
def flatten(elem) -> str:
    """Flatten a Doxygen description element (mixed <para>/<ref>/...) to plain text."""
    if elem is None:
        return ""
    text = " ".join(t.strip() for t in elem.itertext() if t and t.strip())
    return re.sub(r"\s+", " ", text).strip()


# ---- verbatim leading source comment ---------------------------------------
_COMMENT_RE = re.compile(r"^\s*(//|/\*|\*|\*/)")


def leading_comment(repo: Path, path: str, line: int) -> str:
    """Return the contiguous comment block immediately above `line` (1-indexed),
    verbatim. Deterministic; never rewritten. Empty string if none."""
    if not path or line is None or line < 2:
        return ""
    f = repo / path
    try:
        src = f.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    i = line - 2  # 0-indexed line just above the declaration
    out = []
    # skip a single blank gap between comment and declaration
    if 0 <= i < len(src) and src[i].strip() == "":
        i -= 1
    while i >= 0 and _COMMENT_RE.match(src[i]):
        out.append(src[i].rstrip())
        i -= 1
    out.reverse()
    # strip comment markers but keep the text verbatim otherwise
    cleaned = []
    for ln in out:
        s = ln.strip()
        s = re.sub(r"^/\*+|\*+/$", "", s)
        s = re.sub(r"^//+!?|^\*+", "", s).strip()
        # drop pure banner/divider lines (runs of - = # * _ ~ etc.)
        if s and re.fullmatch(r"[-=#*_~<>/\\.\s]{3,}", s):
            continue
        if s:
            cleaned.append(s)
    return " ".join(cleaned).strip()


# ---- data model ------------------------------------------------------------
class Member:
    __slots__ = ("mid", "name", "fqn", "signature", "kind", "file", "line",
                 "brief", "detailed", "comment", "callees", "callers", "owner", "link")

    def __init__(self):
        self.callees = []   # list[(name, refid)]
        self.callers = []   # list[(name, refid)]


def parse(xml_dir: Path, repo: Path):
    members: dict[str, Member] = {}
    compounds = []  # (refid, kind, name)
    index = ET.parse(xml_dir / "index.xml").getroot()
    for comp in index.findall("compound"):
        compounds.append((comp.get("refid"), comp.get("kind"), comp.findtext("name")))

    for refid, kind, _name in compounds:
        if kind not in ("class", "struct", "namespace", "file"):
            continue
        cf = xml_dir / f"{refid}.xml"
        if not cf.exists():
            continue
        cd = ET.parse(cf).getroot().find("compounddef")
        if cd is None:
            continue
        owner = cd.findtext("compoundname") or ""
        for md in cd.findall(".//memberdef"):
            if md.get("kind") != "function":
                continue
            mid = md.get("id")
            if not mid or mid in members:
                continue
            m = Member()
            m.mid = mid
            m.kind = md.get("kind")
            m.name = md.findtext("name") or ""
            m.fqn = md.findtext("qualifiedname") or f"{owner}::{m.name}" if owner else m.name
            definition = (md.findtext("definition") or "").strip()
            args = (md.findtext("argsstring") or "").strip()
            m.signature = f"{definition}{args}".strip()
            loc = md.find("location")
            m.file = loc.get("file") if loc is not None else ""
            # prefer the definition body location (where the real comments live)
            bodyfile = loc.get("bodyfile") if loc is not None else None
            bodystart = loc.get("bodystart") if loc is not None else None
            m.line = int(loc.get("line")) if (loc is not None and loc.get("line")) else None
            m.brief = flatten(md.find("briefdescription"))
            m.detailed = flatten(md.find("detaileddescription"))
            # verbatim comment: try definition body first, then declaration
            m.comment = ""
            if bodyfile and bodystart:
                m.comment = leading_comment(repo, bodyfile, int(bodystart))
            if not m.comment and m.file and m.line:
                m.comment = leading_comment(repo, m.file, m.line)
            m.owner = owner
            m.link = html_link(mid, is_member=True)
            for r in md.findall("references"):
                m.callees.append((r.text or "", r.get("refid") or ""))
            for r in md.findall("referencedby"):
                m.callers.append((r.text or "", r.get("refid") or ""))
            members[mid] = m
    return members, compounds


# ---- pybind11 binding table ------------------------------------------------
_DEF_RE = re.compile(
    r'\.def(?P<variant>_property_readonly|_property|_readonly|_static|_readwrite)?'
    r'\(\s*"(?P<py>[A-Za-z_]\w*)"\s*(?:,\s*&?(?P<cpp>[A-Za-z_][\w:]*))?'
)
_CLASS_RE = re.compile(r'py::class_<\s*(?P<cpp>[A-Za-z_][\w:]*)[^>]*>\s*\(\s*\w+\s*,\s*"(?P<py>[A-Za-z_]\w*)"')


def parse_bindings(repo: Path):
    """Return (classes, methods) parsed deterministically from src/PyBind11.cpp.
    classes: list[(py_class, cpp_class)]; methods: list[(py_name, cpp_target, variant)]."""
    f = repo / "src" / "PyBind11.cpp"
    classes, methods = [], []
    if not f.exists():
        return classes, methods
    text = f.read_text(errors="replace")
    for mm in _CLASS_RE.finditer(text):
        classes.append((mm.group("py"), mm.group("cpp")))
    for mm in _DEF_RE.finditer(text):
        cpp = mm.group("cpp") or "(inline lambda)"
        methods.append((mm.group("py"), cpp, mm.group("variant") or "def"))
    return classes, methods


# ---- emitters --------------------------------------------------------------
def _hdr(title, note):
    return (f"# {title}\n\n"
            f"_Auto-generated by `scripts/gen_doc_indexes.py` from Doxygen XML — "
            f"do not edit by hand. {note}_\n\n")


def emit_api_index(members, out: Path):
    by_owner: dict[str, list[Member]] = {}
    for m in members.values():
        by_owner.setdefault(m.owner or "(global)", []).append(m)
    lines = [_hdr("API Index",
                  "Descriptions are the verbatim leading source comment (this repo has no "
                  "Doxygen `///` comments); absence is noted. Call/caller edges are Doxygen's "
                  "syntactic relations.")]
    lines.append(f"**{len(members)} functions across {len(by_owner)} classes/namespaces.**\n")
    for owner in sorted(by_owner):
        lines.append(f"\n## `{owner}`\n")
        for m in sorted(by_owner[owner], key=lambda x: x.name):
            lines.append(f"### `{m.name}`\n")
            lines.append(f"- **Signature:** `{m.signature}`")
            if m.file and m.line:
                lines.append(f"- **Defined:** `{m.file}:{m.line}`")
            desc = m.brief or m.detailed or m.comment
            if desc:
                src = "doxygen" if (m.brief or m.detailed) else "verbatim source comment"
                lines.append(f"- **Description ({src}):** {desc}")
            else:
                lines.append("- **Description:** _(no comment in source)_")
            if m.callees:
                names = ", ".join(f"`{n}`" for n, _ in m.callees[:12] if n)
                lines.append(f"- **Calls:** {names}" + (" …" if len(m.callees) > 12 else ""))
            if m.callers:
                names = ", ".join(f"`{n}`" for n, _ in m.callers[:12] if n)
                lines.append(f"- **Called by:** {names}" + (" …" if len(m.callers) > 12 else ""))
            if m.link:
                lines.append(f"- **Doxygen:** [{m.name}]({m.link})")
            lines.append("")
    (out / "API_INDEX.md").write_text("\n".join(lines))


def emit_callgraph_index(members, out: Path):
    lines = [_hdr("Call Graph Index",
                  "Per function: outgoing calls (callees) and incoming callers, from Doxygen's "
                  "`references`/`referencedby`. Syntactic — virtual dispatch, templates and "
                  "function pointers are under-represented (see ARCHITECTURE.md).")]
    have_edges = [m for m in members.values() if m.callees or m.callers]
    lines.append(f"**{len(have_edges)} functions with call/caller edges.**\n")
    for m in sorted(have_edges, key=lambda x: (x.owner, x.name)):
        anchor = f"{m.owner}::{m.name}" if m.owner else m.name
        lines.append(f"\n### `{anchor}`  \n`{m.file}:{m.line}`" if m.file else f"\n### `{anchor}`")
        if m.callees:
            lines.append("- **calls →** " + ", ".join(f"`{n}`" for n, _ in m.callees if n))
        if m.callers:
            lines.append("- **← called by** " + ", ".join(f"`{n}`" for n, _ in m.callers if n))
    (out / "CALLGRAPH_INDEX.md").write_text("\n".join(lines))


def emit_module_index(members, compounds, out: Path):
    files = [(r, n) for r, k, n in compounds if k == "file" and (n.endswith(".cpp") or n.endswith(".hpp") or n.endswith(".h"))]
    by_file: dict[str, list[Member]] = {}
    for m in members.values():
        if m.file:
            by_file.setdefault(m.file, []).append(m)
    lines = [_hdr("Module Index",
                  "Source files as modules, their functions, and links to the Doxygen file page "
                  "(which carries the include graph).")]
    for refid, name in sorted(files, key=lambda x: x[1]):
        base = Path(name).name
        fns = sorted({m.name for m in by_file.get(next((p for p in by_file if p.endswith(base)), ""), [])})
        lines.append(f"\n## `{base}`")
        lines.append(f"- **Doxygen page (include graph, source):** [{base}]({html_link(refid, is_member=False)})")
        if fns:
            lines.append("- **Functions:** " + ", ".join(f"`{n}`" for n in fns))
    (out / "MODULE_INDEX.md").write_text("\n".join(lines))


def emit_class_index(members, compounds, out: Path):
    classes = [(r, k, n) for r, k, n in compounds if k in ("class", "struct")]
    by_owner: dict[str, list[Member]] = {}
    for m in members.values():
        by_owner.setdefault(m.owner, []).append(m)
    lines = [_hdr("Class Index", "Classes/structs and their documented member functions.")]
    lines.append(f"**{len(classes)} classes/structs.**\n")
    for refid, kind, name in sorted(classes, key=lambda x: x[2]):
        lines.append(f"\n## `{name}` ({kind})")
        lines.append(f"- **Doxygen page (collaboration + inheritance graph):** [{name}]({html_link(refid, is_member=False)})")
        fns = sorted(by_owner.get(name, []), key=lambda x: x.name)
        if fns:
            lines.append(f"- **Members ({len(fns)}):** " + ", ".join(f"`{m.name}`" for m in fns))
    (out / "CLASS_INDEX.md").write_text("\n".join(lines))


def emit_python_api(members, repo: Path, out: Path):
    classes, methods = parse_bindings(repo)
    # resolve a C++ target (Class::method or World::foo) to a member link
    by_fqn = {}
    for m in members.values():
        by_fqn[f"{m.owner}::{m.name}"] = m
        by_fqn.setdefault(m.name, m)
    lines = [_hdr("Python API Index",
                  "The Python-callable surface (what `import robosample` exposes), parsed "
                  "deterministically from `src/PyBind11.cpp`, mapped to the underlying C++ "
                  "target and its documentation. This is the primary usage-oriented entry point.")]
    if classes:
        lines.append("## Exposed classes\n")
        lines.append("| Python | C++ |")
        lines.append("|---|---|")
        for py, cpp in classes:
            lines.append(f"| `robosample.{py}` | `{cpp}` |")
        lines.append("")
    lines.append(f"## Bound methods / properties ({len(methods)})\n")
    lines.append("| Python name | C++ target | Kind | Doc |")
    lines.append("|---|---|---|---|")
    for py, cpp, variant in sorted(methods):
        target = cpp.split("::")[-1] if "::" in cpp else cpp
        m = by_fqn.get(cpp) or by_fqn.get(target)
        doc = f"[C++]({m.link})" if m and m.link else "—"
        lines.append(f"| `{py}` | `{cpp}` | {variant} | {doc} |")
    (out / "PYTHON_API_INDEX.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--xml", default=None, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    a = ap.parse_args()
    repo = a.repo.resolve()
    xml_dir = (a.xml or repo / "docs/generated/doxygen/xml").resolve()
    out = (a.out or repo / "docs/generated").resolve()
    if not (xml_dir / "index.xml").exists():
        sys.exit(f"error: {xml_dir}/index.xml not found — run Doxygen first (doxygen docs/Doxyfile)")
    out.mkdir(parents=True, exist_ok=True)
    members, compounds = parse(xml_dir, repo)
    emit_api_index(members, out)
    emit_callgraph_index(members, out)
    emit_module_index(members, compounds, out)
    emit_class_index(members, compounds, out)
    emit_python_api(members, repo, out)
    print(f"Generated 5 Markdown indexes in {out} from {len(members)} functions "
          f"across {len(compounds)} compounds.")


if __name__ == "__main__":
    main()
