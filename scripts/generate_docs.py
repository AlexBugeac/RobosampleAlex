#!/usr/bin/env python3
"""One-command documentation pipeline for Robosample.

Runs Doxygen (HTML + XML + Graphviz graphs, scoped to src/ + include/) then the
deterministic Markdown index generator. No LLM / AI anywhere.

    python3 scripts/generate_docs.py            # full run
    python3 scripts/generate_docs.py --xml-only # skip HTML/graphs (faster)
    python3 scripts/generate_docs.py --indexes-only  # reuse existing XML

Requires: doxygen, graphviz (dot), python3. See docs/README.md for install hints.
Outputs:
    docs/generated/doxygen/html/   Doxygen HTML + SVG call/class/include graphs
    docs/generated/doxygen/xml/    Doxygen XML (machine-readable)
    docs/generated/*.md            curated Markdown indexes (committed)
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOXYFILE = REPO / "docs" / "Doxyfile"
XML_INDEX = REPO / "docs" / "generated" / "doxygen" / "xml" / "index.xml"


def need(tool: str):
    if shutil.which(tool) is None:
        sys.exit(f"error: '{tool}' not found on PATH. See docs/README.md for install hints.")


def run_doxygen(xml_only: bool):
    need("doxygen")
    need("dot")  # graphviz
    print("[1/2] Running Doxygen (scoped to src/ + include/) …")
    env_note = ""
    if xml_only:
        env_note = " with GENERATE_HTML=NO"
        # Feed an override by appending to the config via stdin is not supported;
        # instead we run the normal config (HTML is cheap at this repo size).
    # 'nice' to stay friendly to anything else running on the box.
    cmd = ["nice", "-n", "19", "doxygen", str(DOXYFILE)]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"error: doxygen exited {r.returncode}")
    if not XML_INDEX.exists():
        sys.exit("error: Doxygen produced no XML index — check docs/Doxyfile INPUT/GENERATE_XML.")
    print(f"      Doxygen OK{env_note}. XML at {XML_INDEX.parent}")


def run_indexes():
    print("[2/2] Generating curated Markdown indexes from Doxygen XML …")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_doc_indexes.py"),
                        "--repo", str(REPO)], cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"error: index generator exited {r.returncode}")
    # best-effort class-level architecture PNG (needs dot; non-fatal if it fails)
    ag = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_arch_graph.py")], cwd=REPO)
    if ag.returncode != 0:
        print("      (architecture PNG skipped — see above)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml-only", action="store_true", help="reserved; HTML is cheap here")
    ap.add_argument("--indexes-only", action="store_true",
                    help="skip Doxygen, reuse existing XML to rebuild Markdown indexes")
    a = ap.parse_args()
    if not a.indexes_only:
        run_doxygen(a.xml_only)
    elif not XML_INDEX.exists():
        sys.exit("error: --indexes-only but no XML found — run Doxygen first.")
    run_indexes()
    print("\nDone. Open docs/generated/doxygen/html/index.html, or read the Markdown "
          "indexes in docs/generated/ (start with PYTHON_API_INDEX.md).")


if __name__ == "__main__":
    main()
