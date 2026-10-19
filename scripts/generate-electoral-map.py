#!/usr/bin/env python3
"""Generate the three compatible Electoral College map geometries.

Run from the repository root: python scripts/generate-electoral-map.py
Requires Python packages numpy and shapely, plus node to read the existing exact
Shapley results. Generation is offline: state topology is vendored under
data/shapley/, with provenance and its redistribution notice in that directory.

The mainland uses the radial force method of Dougenik, Chrisman & Niemeyer
(1985), https://doi.org/10.1111/j.0033-0124.1985.00075.x, with a fixed target total
area and damped steps. Shared vertices preserve shared state boundaries. Alaska
and Hawaii are shape-preserving insets on the same target-area scale. The solver
then jointly simplifies all three phases, retaining identical vertex ordering.
Quantization and simplification are accepted only after checking all 153 polygon
geometries for validity, all state pairs for overlaps, and each state's area
share against its target to within 0.1% relative error. This is numerical geometry
for exactly computed Shapley shares, not an approximation to the voting game.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polylabel
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/shapley/us-states-albers-10m.json"
t = json.loads(SOURCE.read_text())
weights = json.loads(
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
lookup = {s["name"]: s for s in weights}
arcs = []
for a in t["arcs"]:
    if "transform" in t:
        p = np.cumsum(a, axis=0) * t["transform"]["scale"] + t["transform"]["translate"]
    else:
        p = np.array(a, dtype=float)
    dense = []
    for a, b in zip(p[:-1], p[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / 2)))
        dense.extend(a + (b - a) * j / count for j in range(count))
    dense.append(p[-1])
    arcs.append(np.array(dense))

points = []
pi = {}


def point_id(p):
    key = tuple(p)
    if key not in pi:
        pi[key] = len(points)
        points.append(p)
    return pi[key]


def ring(arcrefs):
    out = []
    for a in arcrefs:
        arc = arcs[a if a >= 0 else ~a]
        if a < 0:
            arc = arc[::-1]
        ids = [point_id(p) for p in arc]
        out += ids if not out else ids[1:]
    if out[-1] != out[0]:
        out.append(out[0])
    return out


states = []
for g in t["objects"]["states"]["geometries"]:
    name = g["properties"]["name"]
    if name not in lookup:
        continue
    polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
    rings = [ring(r) for p in polys for r in p]

    def ring_area(ids):
        pp = np.array([points[i] for i in ids])
        a = pp[:-1]
        b = pp[1:]
        return abs(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]) / 2)

    ras = [ring_area(ids) for ids in rings]
    rings = [ids for ids, ra in zip(rings, ras) if ra > max(ras) * 0.01 and ra > 0.01]
    # Keep Rhode Island's mainland. Small offshore islands collapse when the
    # Ocean State expands beside Connecticut and Massachusetts.
    if name == "Rhode Island":
        rings = [max(rings, key=ring_area)]
    states.append({**lookup[name], "rings": rings})
xy = np.array(points, dtype=float)
start = xy.copy()


# The source rings use a uniform exterior winding. There are no holes in this
# simplified state topology; signed areas keep the map's shared boundaries.
def metrics(xy):
    areas = []
    centroids = []
    for s in states:
        area = 0.0
        c = np.zeros(2)
        for ids in s["rings"]:
            p = xy[ids]
            a = p[:-1]
            b = p[1:]
            cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
            aa = cross.sum() / 2
            area += aa
            c += ((a + b) * cross[:, None]).sum(axis=0) / 6
        areas.append(abs(area))
        centroids.append(c / area)
    return np.array(areas), np.array(centroids)


def solve(xy, values, steps=1000):
    begin = time.time()
    inset = np.array([s["abbreviation"] in ["AK", "HI"] for s in states])
    initial_areas, initial_centers = metrics(xy)
    fixed_area = initial_areas.sum()
    desired = fixed_area * values / values.sum()
    mainland_ids = np.array(
        [
            i
            for s in states
            if s["abbreviation"] not in ["AK", "HI"]
            for ids in s["rings"]
            for i in ids
        ]
    )
    mainland_ids = np.unique(mainland_ids)
    for i in np.flatnonzero(inset):
        ids = np.unique(np.concatenate(states[i]["rings"]))
        xy[ids] = (xy[ids] - initial_centers[i]) * np.sqrt(
            desired[i] / initial_areas[i]
        ) + initial_centers[i]
    for step in range(steps):
        areas, c = metrics(xy)
        error = np.maximum(areas / desired, desired / areas)
        if step % 50 == 0:
            print(
                step,
                "max",
                error.max() - 1,
                "mean",
                error.mean() - 1,
                "sec",
                time.time() - begin,
                flush=True,
            )
        if error.max() < 1.00005:
            break
        r = np.sqrt(areas[~inset] / np.pi)
        mass = np.sqrt(desired[~inset] / np.pi) - r
        delta = xy[mainland_ids, None, :] - c[None, ~inset, :]
        d = np.maximum(np.linalg.norm(delta, axis=2), 1e-9)
        force = np.where(
            d > r, mass * r / d, mass * (d * d / (r * r)) * (4 - 3 * d / r)
        )
        xy[mainland_ids] += (
            0.25 * np.sum(delta * (force / d)[:, :, None], axis=1) / (1 + error.mean())
        )
    return xy


electoral = solve(xy.copy(), np.array([s["votes"] for s in states]), 2000)
adjusted = solve(electoral.copy(), np.array([s["share"] * 538 for s in states]), 1000)
modes = ["geographic", "electoral", "adjusted"]
coords = np.array([start, electoral, adjusted])


def select(ids, tolerance):
    if len(ids) <= 2:
        return ids
    pp = coords[:, ids, :]
    a = pp[:, :1, :]
    b = pp[:, -1:, :]
    d = b - a
    v = pp - a
    ll = np.sum(d * d, axis=2, keepdims=True)
    t = np.clip(np.sum(v * d, axis=2, keepdims=True) / np.maximum(ll, 1e-15), 0, 1)
    deviation = np.linalg.norm(v - t * d, axis=2).max(axis=0)
    i = int(deviation.argmax())
    if deviation[i] <= tolerance:
        return [ids[0], ids[-1]]
    return select(ids[: i + 1], tolerance)[:-1] + select(ids[i:], tolerance)


for tolerance in [0.15, 0.08, 0.03, 0.01, 0.003, 0]:
    keep = set()
    for arc in arcs:
        ids = [pi[tuple(p)] for p in arc if tuple(p) in pi]
        keep.update(select(ids, tolerance))
    rings = [[list(i for i in r if i in keep) for r in s["rings"]] for s in states]
    use = np.array(sorted(set(i for sr in rings for r in sr for i in r)))
    minimum = coords[:, use, :].min(axis=(0, 1))
    maximum = coords[:, use, :].max(axis=(0, 1))
    offset = np.array([30, 30]) - minimum
    cc = np.round(coords + offset, 2)
    diagnostics = {}
    valid = True
    for k, mode in enumerate(modes):
        ps = []
        for s, srings in zip(states, rings):
            polygons = [Polygon(cc[k, ids]) for ids in srings]
            p = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
            ps.append(p)
            if not p.is_valid:
                print(mode, s["abbreviation"], explain_validity(p))
                valid = False
        if not valid:
            break
        overlaps = []
        for i, p in enumerate(ps):
            for j, q in enumerate(ps[:i]):
                area = p.intersection(q).area
                if area > 1e-6:
                    overlaps.append(
                        (states[i]["abbreviation"], states[j]["abbreviation"], area)
                    )
        total = sum(p.area for p in ps)
        errors = [
            abs(
                p.area / total
                - (s["votes"] if mode == "electoral" else s["share"] * 538) / 538
            )
            / ((s["votes"] if mode == "electoral" else s["share"] * 538) / 538)
            for p, s in zip(ps, states)
        ]
        diagnostics[mode] = {
            "totalArea": round(total, 4),
            "maxRelativeAreaError": max(errors) if mode != "geographic" else None,
            "invalidPolygonCount": 0,
            "overlapPairCount": len(overlaps),
            "overlapArea": sum(o[2] for o in overlaps),
        }
        print(tolerance, mode, diagnostics[mode], overlaps)
        if overlaps or (mode != "geographic" and max(errors) > 0.001):
            valid = False
    if valid:
        break
assert valid, "Unable to produce valid geometry within area tolerance"

# Linear interpolation is also checked between the quantized endpoints. Area
# proportions are intentionally meaningful only at the two finished cartograms.
morph_frames_checked = 3
for k in range(2):
    for fraction in np.linspace(0.05, 0.95, 19):
        frame = cc[k] * (1 - fraction) + cc[k + 1] * fraction
        polygons = []
        for state, state_rings in zip(states, rings):
            parts = [Polygon(frame[ids]) for ids in state_rings]
            polygon = parts[0] if len(parts) == 1 else MultiPolygon(parts)
            assert polygon.is_valid, (
                f"Invalid intermediate polygon: {state['abbreviation']}, "
                f"transition {k}, fraction {fraction}"
            )
            polygons.append(polygon)
        for i, polygon in enumerate(polygons):
            for j, other in enumerate(polygons[:i]):
                assert polygon.intersection(other).area < 1e-6, (
                    f"Intermediate overlap: {states[i]['abbreviation']} and "
                    f"{states[j]['abbreviation']}, transition {k}, fraction {fraction}"
                )
        morph_frames_checked += 1
print(f"Validated {morph_frames_checked} endpoint and intermediate frames.")

out = {
    "width": int(np.ceil(maximum[0] - minimum[0] + 60 + 115)),
    "height": int(np.ceil(maximum[1] - minimum[1] + 60)),
    "projection": "Albers USA; Alaska and Hawaii are insets",
    "totalVotes": 538,
    "diagnostics": diagnostics,
    "morphChecks": {
        "sampledFrames": morph_frames_checked,
        "invalidPolygonCount": 0,
        "overlapPairCount": 0,
    },
    "attribution": {
        "name": "U.S. Census Bureau, 2017 cartographic boundaries via U.S. Atlas 3.0.1",
        "url": "https://github.com/topojson/us-atlas",
        "source": "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-albers-10m.json",
        "method": "Continuous area cartograms: Dougenik, Chrisman and Niemeyer (1985). Alaska and Hawaii retain their shapes and share the same area scale. Tiny offshore islands are omitted; Rhode Island uses its mainland outline.",
        "paper": "https://doi.org/10.1111/j.0033-0124.1985.00075.x",
    },
    "states": [],
}
for s, srings in zip(states, rings):
    item = {
        "abbreviation": s["abbreviation"],
        "name": s["name"],
        "votes": s["votes"],
        "adjustedVotes": s["share"] * 538,
        "labels": {},
    }
    for k, mode in enumerate(modes):
        item[mode] = [cc[k, ids, :].tolist() for ids in srings]
        polygons = [Polygon(cc[k, ids]) for ids in srings]
        point = polylabel(max(polygons, key=lambda p: p.area), tolerance=0.1)
        item["labels"][mode] = [round(point.x, 4), round(point.y, 4)]
    out["states"].append(item)


def base36(value):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if not value:
        return "0"
    result = ""
    while value:
        result = digits[value % 36] + result
        value //= 36
    return result


def encode_ring(ring):
    previous = [0, 0]
    encoded = []
    for point in ring:
        for k in range(2):
            value = round(point[k] * 100)
            delta = value - previous[k]
            previous[k] = value
            encoded.append(base36(delta * 2 if delta >= 0 else -delta * 2 - 1))
    return ".".join(encoded)


for s in out["states"]:
    for mode in modes:
        s[mode] = ";".join(encode_ring(ring) for ring in s[mode])
target = ROOT / "docs/shapley/electoral-map-data.mjs"
decode = """
// Base-36 zigzag deltas store hundredths of a pixel without runtime libraries.
function decodeRings(encoded) {
  return encoded.split(';').map(ring => {
    const values = ring.split('.').map(value => parseInt(value, 36));
    let x = 0, y = 0;
    const points = [];
    for (let i = 0; i < values.length; i += 2) {
      x += values[i] % 2 ? -(values[i] + 1) / 2 : values[i] / 2;
      y += values[i + 1] % 2 ? -(values[i + 1] + 1) / 2 : values[i + 1] / 2;
      points.push([x / 100, y / 100]);
    }
    return points;
  });
}
for (const state of data.states) {
  for (const phase of ['geographic', 'electoral', 'adjusted']) state[phase] = decodeRings(state[phase]);
}
export const ELECTORAL_MAP = data;
"""
target.write_text(
    "// Generated by scripts/generate-electoral-map.py. See that file for sources and validation.\nconst data = "
    + json.dumps(out, separators=(",", ":"))
    + ";\n"
    + decode
)
print(target, target.stat().st_size, "points", sum(len(r) for sr in rings for r in sr))
