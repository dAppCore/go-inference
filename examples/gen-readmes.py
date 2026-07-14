# SPDX-Licence-Identifier: EUPL-1.2
"""Generate a README.md beside every examples/pkg/**/main.go from its package
doc comment — GitHub renders the README in the directory view, so each example
folder documents itself without opening the file. The main.go doc comment stays
the single source of truth; rerun this after adding examples:

    python3 gen-readmes.py
"""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent

for main in sorted((root / "pkg").rglob("main.go")):
    lines = main.read_text().splitlines()
    # The package doc comment: contiguous // block immediately before
    # `package main`, minus the SPDX header and build-constraint lines.
    doc = []
    for line in lines:
        if line.startswith("package "):
            break
        if line.startswith("//go:build") or "SPDX-Licence-Identifier" in line:
            continue
        if line.startswith("//"):
            text = line[2:].lstrip() if line != "//" else ""
            if text == "" and doc and doc[-1] == "":
                continue
            doc.append(text)
    # Trim leading/trailing blanks.
    while doc and doc[0] == "":
        doc.pop(0)
    while doc and doc[-1] == "":
        doc.pop()

    rel = main.parent.relative_to(root / "pkg")
    # Split prose vs the indented `go run` example line(s).
    prose, run = [], []
    for d in doc:
        if re.match(r"^\s*go run ", d):
            run.append(d.strip())
        elif not (d == "" and prose and prose[-1] == ""):
            prose.append(d)
    while prose and prose[-1] == "":
        prose.pop()

    out = ["<!-- SPDX-Licence-Identifier: EUPL-1.2 -->", "", f"# {rel}", ""]
    out += prose
    out += ["", "## Run", "", "```sh"]
    out += run if run else [f"go run . -model <model snapshot dir>"]
    out += ["```", ""]
    out.append("Flags and behaviour are documented in [main.go](main.go) — the code is the example.")
    out.append("")
    (main.parent / "README.md").write_text("\n".join(out))
    print(f"wrote {main.parent.relative_to(root)}/README.md")
