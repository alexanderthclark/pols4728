import { ELECTORAL_MAP } from './electoral-map-data.mjs';
import { ELECTORAL_TILES } from './electoral-tile-data.mjs';

const signedArea = ring => ring.reduce((sum, a, i) => {
  const b = ring[(i + 1) % ring.length];
  return sum + a[0] * b[1] - b[0] * a[1];
}, 0) / 2;

function perimeter(ring, winding) {
  let points = ring.map(point => [...point]);
  if (points.length > 1 && points[0].every((v, i) => v === points.at(-1)[i])) points.pop();
  if (Math.sign(signedArea(points)) !== winding) points.reverse();
  const start = points.reduce((best, p, i) => p[1] < points[best][1] || (p[1] === points[best][1] && p[0] < points[best][0]) ? i : best, 0);
  points = [...points.slice(start), ...points.slice(0, start)];
  const distances = [0];
  points.forEach((a, i) => {
    const b = points[(i + 1) % points.length];
    distances.push(distances.at(-1) + Math.hypot(b[0] - a[0], b[1] - a[1]));
  });
  const total = distances.at(-1);
  const fractions = distances.map(distance => distance / total);
  return { fractions, at(t) {
    let i = 0;
    while (i < points.length - 1 && fractions[i + 1] < t) i++;
    const amount = (t - fractions[i]) / (fractions[i + 1] - fractions[i] || 1);
    return points[i].map((value, d) => value + (points[(i + 1) % points.length][d] - value) * amount);
  } };
}

// Include every original corner in every endpoint. Equal-distance sampling alone
// would cut the tile corners and subtly change the areas being taught.
function correspond(rings) {
  const winding = Math.sign(signedArea(rings[0])) || 1;
  const outlines = rings.map(ring => perimeter(ring, winding));
  const fractions = [...new Set(outlines.flatMap(outline => outline.fractions.slice(0, -1)).map(t => +t.toFixed(12)))].sort((a, b) => a - b);
  return outlines.map(outline => fractions.map(t => outline.at(t)));
}

/** Geographic and square-cell endpoints in one coordinate frame and area scale. */
export function prepareElectoralMap(results) {
  // A taller common frame lets the square map use a phone's available width.
  // The geographic projection is only scaled uniformly and centered within it.
  const width = 950, height = ELECTORAL_MAP.height;
  const geographicScale = Math.min((width - 24) / ELECTORAL_MAP.width, (height - 24) / ELECTORAL_MAP.height);
  const geographicOffset = [(width - geographicScale * ELECTORAL_MAP.width) / 2, (height - geographicScale * ELECTORAL_MAP.height) / 2];
  const geographicPoint = p => p.map((value, d) => geographicOffset[d] + geographicScale * value);
  const unit = Math.min((width - 48) / ELECTORAL_TILES.width, (height - 48) / ELECTORAL_TILES.height);
  const offset = [(width - unit * ELECTORAL_TILES.width) / 2, (height - unit * ELECTORAL_TILES.height) / 2];
  const point = p => p.map((value, d) => offset[d] + unit * value);
  const rectangle = r => [offset[0] + unit * r.x, offset[1] + unit * r.y, unit * r.width, unit * r.height];
  const byCode = new Map(results.map(state => [state.abbreviation, state]));
  const tilesByCode = new Map(ELECTORAL_TILES.states.map(state => [state.abbreviation, state]));
  const states = ELECTORAL_MAP.states.map(source => {
    const tile = tilesByCode.get(source.abbreviation);
    const result = byCode.get(source.abbreviation);
    const geographic = source.geographic.map(ring => ring.map(geographicPoint)).sort((a, b) => Math.abs(signedArea(b)) - Math.abs(signedArea(a)));
    const label = point(tile.label);
    const main = correspond([geographic[0], tile.rings[0].map(point), tile.adjustedRings[0].map(point)]);
    const extra = geographic.slice(1).map(ring => ring.map(() => [...label]));
    return {
      abbreviation: source.abbreviation, name: source.name, votes: result.votes,
      adjustedVotes: 538 * result.share,
      geographic: [main[0], ...geographic.slice(1)],
      electoral: [main[1], ...extra], adjusted: [main[2], ...extra],
      labels: { geographic: geographicPoint(source.labels.geographic), electoral: label, adjusted: label },
      cells: tile.rectangles.map((r, i) => ({ electoral: rectangle(r), adjusted: rectangle(tile.adjustedRectangles[i]) }))
    };
  });
  return { width, height, unit, states };
}

/** Exposed edges of non-overlapping, possibly fractional, axis-aligned cells. */
export function rectangleBoundary(rectangles) {
  const groups = new Map();
  function add(axis, fixed, start, end, direction) {
    const key = `${axis}:${fixed.toFixed(7)}`;
    if (!groups.has(key)) groups.set(key, { axis, fixed, intervals: [] });
    groups.get(key).intervals.push([start, end, direction]);
  }
  for (const [x, y, w, h] of rectangles) {
    if (w < 1e-7 || h < 1e-7) continue;
    add(0, y, x, x + w, 1); add(0, y + h, x, x + w, -1);
    add(1, x, y, y + h, -1); add(1, x + w, y, y + h, 1);
  }
  const edges = [];
  for (const { axis, fixed, intervals } of groups.values()) {
    const ends = [...new Set(intervals.flatMap(interval => interval.slice(0, 2)).map(value => +value.toFixed(7)))].sort((a, b) => a - b);
    for (let i = 0; i < ends.length - 1; i++) {
      const start = ends[i], end = ends[i + 1], middle = (start + end) / 2;
      if (end - start < 1e-7) continue;
      const direction = intervals.reduce((sum, interval) => sum + (middle > interval[0] - 1e-7 && middle < interval[1] + 1e-7 ? interval[2] : 0), 0);
      if (!direction) continue;
      const a = axis === 0 ? [start, fixed] : [fixed, start];
      const b = axis === 0 ? [end, fixed] : [fixed, end];
      edges.push([a, b]);
    }
  }
  return edges;
}
