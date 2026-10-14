# Electoral College map geometry

`us-states-albers-10m.json` is U.S. Atlas 3.0.1's projected state TopoJSON,
downloaded from
<https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-albers-10m.json>.
Its SHA-256 is
`6e7bb086a3c791490361968a3094f377f7726c5d0c4900fec03cc42db2305a3d`.

U.S. Atlas redistributes the U.S. Census Bureau's 2017 cartographic state
boundaries. The projection is Albers USA, with Alaska and Hawaii insets:
`geoAlbersUsa().scale(1300).translate([487.5, 305])`.
The source data and projection are documented by the
[U.S. Atlas maintainers](https://github.com/topojson/us-atlas).
The redistribution notice is preserved in `LICENSE-us-atlas.txt`.

Run `python scripts/generate-electoral-map.py` from the repository root to
regenerate `docs/shapley/electoral-map-data.mjs`. This authoring tool requires
Node.js and the Python dependencies in `scripts/requirements-cartogram.txt`;
the published page has no mapping-library or network dependency.

The generator reads electoral votes and the already exact Shapley–Shubik
results from `docs/shapley/electoral-college.mjs`. A state's adjusted electoral
votes are `538 × its Shapley share`; these adjusted votes sum to 538. The
winner-take-all modeling assumptions, including Maine and Nebraska, remain
those of that module.

The mainland is a continuous cartogram, using the radial force equations of
[Dougenik, Chrisman, and Niemeyer (1985)](https://doi.org/10.1111/j.0033-0124.1985.00075.x).
Shared vertices preserve state boundaries. Alaska and Hawaii retain their
shapes as separate insets, using the same area scale as the mainland in each
cartogram. Very small offshore islands are omitted; Rhode Island uses its
mainland outline. The geographic starting map uses the same simplified
outlines, so every ring and vertex matches throughout the morph.

After jointly simplifying and quantizing all three maps, the generator checks
polygon validity, pairwise overlap, and each state's area share. The accepted
maximum relative area error is 0.1%, a numerical tolerance on the geometry;
the Shapley shares themselves come from exact integer counting. Diagnostic
values are exported with the generated module. Both cartograms use a common
coordinate frame and effectively equal total filled area, rather than fitting
each independently to its bounding box.

Coordinates are stored as base-36 zigzag deltas in hundredths of a pixel. The
module decodes them to ordinary arrays of `[x, y]` points when imported. This
reduces the download size without changing the data exposed to the view.
