#!/usr/bin/env python
"""Fail if the post contains a result that no file in results/ supports.

Two checks:
1. Every `{{< var ... >}}`-free numeric claim in the post must come from
   `posts/channel-adapters/numbers.json`, which make_figures.py writes from
   results/. The post refers to numbers through that file's keys using Quarto
   inline code `{python} n["..."]`-style is not available in a plain .qmd
   without an engine, so we keep it simple: make_figures.py writes
   `numbers.json`, the post quotes those numbers literally, and this script
   verifies that every number appearing in a results sentence is present in
   numbers.json (within rounding).
2. The post may contain TODO(results: X) placeholders only while results/X
   has no files. Once an experiment has results, its placeholders must be
   resolved.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "posts" / "channel-adapters" / "index.qmd"
NUMBERS = ROOT / "posts" / "channel-adapters" / "numbers.json"
RESULTS = ROOT / "results"

text = POST.read_text()
errors = []

# 2. placeholders versus available results
for exp in sorted(set(re.findall(r"TODO\(results: (E\d\w*)", text))):
    if any((RESULTS / exp).glob("*.json")):
        errors.append(f"post still has TODO(results: {exp}) but results/{exp}/ has files")

# 1. literal numbers inside sentences tagged as results: `<!-- result -->` marks a line
if NUMBERS.exists():
    allowed = {round(float(v), 1) for v in json.loads(NUMBERS.read_text()).values() if isinstance(v, (int, float))}
    for line in text.splitlines():
        if "<!-- result -->" not in line:
            continue
        for num in re.findall(r"(?<![\w.])(\d+\.\d+|\d+)(?=%|\s|,|\.|\))", line):
            if round(float(num), 1) not in allowed:
                errors.append(f"number {num} on a result line is not in numbers.json: {line.strip()[:80]}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print("check_numbers: ok")
