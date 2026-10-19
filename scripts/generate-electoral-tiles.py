#!/usr/bin/env python3
"""Generate the square-cell Electoral College cartogram and adjusted geometry.

Run from the repository root with the dependencies in requirements-cartogram.txt.
The hand-adjusted 538-cell U.S. layout lives in data/shapley/electoral-tile-layout.json.
One unit square always means one electoral vote. Adjusted votes use fractional
boundary squares; this keeps one area scale and makes all 538 original cells
directly comparable. Nine reserved coastal cells expand from zero area.
"""
import json
import math
import subprocess
from pathlib import Path

from shapely import union_all
from shapely.geometry import box
from shapely.ops import polylabel, unary_union

ROOT = Path(__file__).resolve().parents[1]
layout = json.loads((ROOT / "data/shapley/electoral-tile-layout.json").read_text())
results = json.loads(
    subprocess.check_output(
        [
            "node",
            "--input-type=module",
            "-e",
            "import {electoralResults} from './docs/shapley/electoral-college.mjs';"
            "console.log(JSON.stringify(electoralResults));",
        ],
        cwd=ROOT,
        text=True,
    )
)

# Two cells of padding accommodate coastal additions without changing the frame.
layout = {a: [(x + 2, y + 2) for x, y in cs] for a, cs in layout.items()}
occupied = {cell for cs in layout.values() for cell in cs}
assert len(occupied) == 538
assert sum(len(cs) for cs in layout.values()) == 538


def shape(rectangles):
    return union_all(
        [
            box(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
            for r in rectangles
            if r["width"] > 0 and r["height"] > 0
        ],
        grid_size=1e-12,
    )


def cell_rect(cell):
    return {"x": cell[0], "y": cell[1], "width": 1, "height": 1}


def rings(polygon):
    assert polygon.geom_type == "Polygon" and polygon.is_valid
    assert len(polygon.interiors) == 0
    return [[list(p) for p in polygon.exterior.coords]]


baseline_shapes = {a: shape([cell_rect(c) for c in cs]) for a, cs in layout.items()}
for a, polygon in baseline_shapes.items():
    assert polygon.geom_type == "Polygon" and polygon.is_valid, a
    assert len(polygon.interiors) == 0, a

states = []
reserved = set()
for state in results:
    a = state["abbreviation"]
    cells = layout[a]
    assert len(cells) == state["votes"], a
    baseline = [cell_rect(c) for c in cells]
    adjusted = [r.copy() for r in baseline]
    target = state["share"] * 538
    change = target - state["votes"]
    polygon = baseline_shapes[a]
    center = [polygon.centroid.x, polygon.centroid.y]
    point = polylabel(polygon, tolerance=0.01)

    if change > 0:
        # Every addition touches an existing state square along a vertical edge.
        # Growing only its width preserves total area throughout interpolation.
        candidates = []
        own = set(cells)
        for x, y in cells:
            for dx in [-1, 1]:
                neighbor = (x + dx, y)
                if neighbor not in occupied and neighbor not in reserved:
                    # Favor California's Pacific coast and each other state's
                    # nearest exposed edge, keeping additions compact and local.
                    preference = (
                        neighbor[0]
                        if a == "CA"
                        else (neighbor[0] + 0.5 - center[0]) ** 2
                        + (neighbor[1] + 0.5 - center[1]) ** 2
                    )
                    candidates.append((preference, neighbor, dx))
        candidates.sort()
        used = set()
        for slot in range(math.ceil(change)):
            _, cell, direction = next(
                c for c in candidates if c[1] not in used and c[1] not in reserved
            )
            used.add(cell)
            reserved.add(cell)
            amount = min(1, change - slot)
            x, y = cell
            # direction -1 grows left from the state's right-hand shared edge.
            anchor = x + 1 if direction == -1 else x
            baseline.append({"x": anchor, "y": y, "width": 0, "height": 1})
            adjusted.append(
                {
                    "x": anchor - amount if direction == -1 else anchor,
                    "y": y,
                    "width": amount,
                    "height": 1,
                }
            )
    else:
        # Clip a strip from one state-boundary cell, keeping its opposite edge.
        # Only a fraction of one square is removed; all 47 reductions are <.2.
        own = set(cells)
        choices = []
        for index, (x, y) in enumerate(cells):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in own:
                    continue
                r = cell_rect((x, y))
                if dx:
                    r["width"] += change
                    if dx == -1:
                        r["x"] -= change
                else:
                    r["height"] += change
                    if dy == -1:
                        r["y"] -= change
                trial = adjusted.copy()
                trial[index] = r
                candidate = shape(trial)
                if (
                    candidate.geom_type == "Polygon"
                    and candidate.is_valid
                    and not candidate.interiors
                    and candidate.contains(point)
                ):
                    coastal = 0 if neighbor not in occupied else 1
                    distance = (x + 0.5 - point.x) ** 2 + (y + 0.5 - point.y) ** 2
                    choices.append((coastal, -distance, index, dx, dy, r))
        assert choices, a
        _, _, index, _, _, r = min(choices, key=lambda c: c[:5])
        adjusted[index] = r

    adjusted_shape = shape(adjusted)
    assert abs(adjusted_shape.area - target) < 1e-10, (a, adjusted_shape.area, target)
    states.append(
        {
            "abbreviation": a,
            "name": state["name"],
            "votes": state["votes"],
            "adjustedVotes": target,
            "cells": cells,
            "rings": rings(polygon),
            "center": center,
            "label": [point.x, point.y],
            "rectangles": baseline,
            "adjustedRectangles": adjusted,
            "adjustedRings": rings(adjusted_shape),
        }
    )

assert len(reserved) == 9
assert abs(sum(s["adjustedVotes"] for s in states) - 538) < 1e-10
for fraction in [i / 20 for i in range(21)]:
    polygons = []
    for state in states:
        rectangles = [
            {
                key: a[key] * (1 - fraction) + b[key] * fraction
                for key in ["x", "y", "width", "height"]
            }
            for a, b in zip(state["rectangles"], state["adjustedRectangles"])
        ]
        polygon = shape(rectangles)
        assert (
            polygon.is_valid
            and polygon.geom_type == "Polygon"
            and not polygon.interiors
        ), (fraction, state["abbreviation"], polygon.geom_type)
        polygons.append(polygon)
    assert abs(sum(p.area for p in polygons) - 538) < 1e-10
    for i, polygon in enumerate(polygons):
        for j, other in enumerate(polygons[:i]):
            assert polygon.intersection(other).area < 1e-10, (
                fraction,
                states[i]["abbreviation"],
                states[j]["abbreviation"],
            )

out = {
    "width": 43,
    "height": 32,
    "cellArea": 1,
    "totalVotes": 538,
    "description": "Each full square is one electoral vote. Fractional squares encode adjusted electoral votes, 538 times each state's exact Shapley share.",
    "diagnostics": {
        "baselineCellCount": 538,
        "reservedCoastalCells": 9,
        "adjustedRectangleCount": 547,
        "sampledMorphFrames": 21,
        "invalidPolygonCount": 0,
        "overlapPairCount": 0,
    },
    "states": states,
}
target = ROOT / "docs/shapley/electoral-tile-data.mjs"
target.write_text(
    "// Generated by scripts/generate-electoral-tiles.py from the editable square-cell layout.\nexport const ELECTORAL_TILES = "
    + json.dumps(out, separators=(",", ":"))
    + ";\n"
)
print(
    f"Wrote {target}: 538 cells, 51 connected states, 547 adjusted slots, zero overlap in 21 frames."
)
