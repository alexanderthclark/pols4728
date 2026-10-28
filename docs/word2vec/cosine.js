// The same illustrative vectors are used in the analogy and its cosine pause.
export const analogyVectors = {
  man: [.45, .3], woman: [1.35, .5], king: [.75, 1.65], queen: [1.92, 1.75],
};
const {man, woman, king, queen} = analogyVectors;
const query = king.map((value, j) => value - man[j] + woman[j]);
const queryNorm = Math.hypot(...query), queenNorm = Math.hypot(...queen);
const similarity = query.reduce((sum, value, j) => sum + value * queen[j], 0)
  / (queryNorm * queenNorm);

export const cosineScene = {
  chapter: '05 · Reading the representation',
  title: 'Here, “similar” means a small angle.',
  eq: String.raw`\cos(q,h_{\text{queen}})=\frac{q^{\mathsf T}h_{\text{queen}}}{\lVert q\rVert\,\lVert h_{\text{queen}}\rVert}`,
  caption: 'Compare the query with candidate word vectors by cosine similarity. A smaller angle gives a higher score.',
  detail: 'Cosine distance = 1 − cosine similarity: smaller is closer. Use the full embeddings, not a 2D projection.',
  alt: `The query king minus man plus woman and the queen vector start at the same origin. Scrolling rescales both to unit length without changing their small angle. These illustrative vectors have cosine similarity about ${similarity.toFixed(3)} and cosine distance about ${(1 - similarity).toFixed(3)}.`,
};

export function renderCosine(t, n, helpers) {
  const {m, text, line, ease, mix, fmt, BLUE, RUST, MUTED} = helpers;
  const p = ease(t), ox = n ? 35 : 180, oy = n ? 365 : 350;
  const initialScale = n ? 140 : 155, unitScale = n ? 220 : 260;
  const endpoint = (v, length) => {
    const scale = mix(initialScale, unitScale / length, p);
    return [ox + scale * v[0], oy - scale * v[1]];
  };
  const a = endpoint(query, queryNorm), b = endpoint(queen, queenNorm);
  let s = m(String.raw`q=h_{\text{king}}-h_{\text{man}}+h_{\text{woman}}`,
    n ? 200 : 500, 35, n ? 390 : 800, 44, n ? 20 : 27);
  s += line(ox, oy, a[0], a[1], RUST, 3, true, '6 4');
  s += line(ox, oy, b[0], b[1], BLUE, 2.7, true);
  s += m('q', a[0] - (n ? 19 : 44), a[1] + (n ? -19 : 18), 50, 34, n ? 23 : 28, RUST);
  s += m(String.raw`h_{\text{queen}}`, b[0] + (n ? 32 : 52), b[1] + 20,
    n ? 100 : 160, 42, n ? 20 : 26, BLUE);
  s += text('0', ox - 12, oy + 17, 13, MUTED);
  const thetaQ = Math.atan2(query[1], query[0]);
  const thetaQueen = Math.atan2(queen[1], queen[0]);
  const radius = n ? 84 : 120;
  s += `<path d="M${ox + radius * Math.cos(thetaQueen)} ${oy - radius * Math.sin(thetaQueen)} A${radius} ${radius} 0 0 0 ${ox + radius * Math.cos(thetaQ)} ${oy - radius * Math.sin(thetaQ)}" stroke="${MUTED}" stroke-width="1.8" fill="none"/>`;
  s += m(String.raw`\theta`, ox + radius + 20, oy - radius * .58, 50, 40, 23, MUTED);
  s += text(p === 1 ? 'Both lengths are now 1; the angle is unchanged.' : 'Rescale the arrows without turning them.',
    n ? 200 : 760, n ? 77 : 140, n ? 12 : 15, MUTED);
  s += m(String.raw`\cos(q,h_{\text{queen}})=${fmt(similarity, 3)}`,
    n ? 200 : 760, n ? 395 : 215, n ? 390 : 460, 38, n ? 20 : 27, BLUE);
  s += m(String.raw`d_{\cos}=1-\cos(q,h_{\text{queen}})=${fmt(1 - similarity, 3)}`,
    n ? 200 : 760, n ? 432 : 280, n ? 390 : 470, 34, n ? 17 : 24, MUTED);
  return s;
}
