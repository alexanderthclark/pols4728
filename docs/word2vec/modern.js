// A schematic of a common text-embedding pipeline, not a vendor architecture.
// Positions are explanatory layout coordinates; no embedding values are invented.
export const modernScene = {
  chapter: '06 · Modern embeddings',
  title: 'Modern models embed whole texts.',
  eq: String.raw`\text{text}\xrightarrow{\text{trained model}}e_{\text{text}}`,
  caption: 'A transformer reads tokens in context. Many text encoders adapt a pretrained model using related text pairs.',
  detail: 'An API such as OpenAI’s runs an already trained model and returns a vector for your text. The diagram shows a common approach.',
  alt: 'The text Voters support reform is split into schematic tokens. Scrolling moves them through a trained transformer, which combines information across tokens. The resulting contextual vectors then combine into one vector for the entire text. A common training goal makes related texts have similar vectors; an embeddings API runs the trained model on new text.',
};

export function renderModern(t, n, helpers) {
  const {m, text, line, rect, group, ease, mix, BLUE, RUST, INK, MUTED, RULE} = helpers;
  const enter = ease(t / .3), context = ease((t - .3) / .3);
  const combine = ease((t - .6) / .35);
  const words = ['Voters', 'support', 'reform'];
  const input = n ? [[72, 66], [200, 66], [328, 66]] : [[135, 150], [135, 220], [135, 290]];
  const first = n ? [[90, 173], [200, 173], [310, 173]] : [[435, 150], [435, 220], [435, 290]];
  const last = n ? [[90, 240], [200, 240], [310, 240]] : [[645, 150], [645, 220], [645, 290]];
  const output = n ? [200, 335] : [870, 220];
  let s = `<g data-modern="pipeline" data-enter="${enter.toFixed(4)}" data-context="${context.toFixed(4)}" data-combine="${combine.toFixed(4)}">`;
  s += text('Voters support reform.', n ? 200 : 135, n ? 25 : 52, n ? 21 : 22, INK);
  if (!n) s += text('text', 135, 82, 15, MUTED);
  s += rect(n ? 20 : 325, n ? 111 : 85, n ? 360 : 420, n ? 168 : 250, RULE, '#fafaf8', 4, 1.2);
  s += text('Trained transformer', n ? 200 : 535, n ? 137 : 112, n ? 20 : 22, BLUE);
  if (!n) s += text('reads tokens in context', 535, 325, 17, MUTED);

  // The crossed connections show that each output can use information from
  // the whole sequence, rather than taking a separate vocabulary lookup.
  for (const a of first) for (const b of last) {
    const start = n ? [a[0], a[1] + 13] : [a[0] + 26, a[1]];
    const end = n ? [b[0], b[1] - 18] : [b[0] - 28, b[1]];
    s += group(line(...start, ...end, RULE, 1.3), .3 + .7 * context);
  }
  for (let j = 0; j < words.length; j++) {
    const a = input[j], b = first[j], c = last[j];
    const tokenWidth = n ? 108 : 150;
    s += rect(a[0] - tokenWidth / 2, a[1] - 17, tokenWidth, 34, RULE, 'white', 3);
    s += text(words[j], a[0], a[1] + 6, n ? 18 : 20, MUTED);
    const arrowStart = n ? [a[0], a[1] + 20] : [a[0] + tokenWidth / 2 + 8, a[1]];
    const arrowEnd = n ? [b[0], b[1] - 22] : [b[0] - 37, b[1]];
    if (!n) s += line(...arrowStart, ...arrowEnd, RULE, 1.3, true);

    const x = mix(mix(a[0], b[0], enter), c[0], context);
    const y = mix(mix(a[1], b[1], enter), c[1], context);
    const w = mix(tokenWidth, n ? 62 : 74, context);
    const h = mix(34, 42, context);
    s += `<g data-modern-token="${j}" data-x="${x.toFixed(4)}" data-y="${y.toFixed(4)}">`;
    s += rect(x - w / 2, y - h / 2, w, h, BLUE, 'white', 3, 1.5);
    s += group(text(words[j], x, y + 6, n ? 18 : 20, BLUE), Math.max(0, 1 - 2 * context));
    s += group(m(`h_{${j + 1}}`, x, y, 70, 40, n ? 25 : 28, BLUE), Math.max(0, 2 * context - 1));
    s += '</g>';

    const start = n ? [c[0], c[1] + 23] : [c[0] + 40, c[1]];
    const end = n ? [output[0], output[1] - 29] : [output[0] - 48, output[1]];
    s += group(line(...start, ...end, RUST, 1.4, true), .3 + .7 * context);
    // Copies of the contextual outputs converge on the whole-text vector.
    // The original contextual vectors stay in the encoder for comparison.
    const mergeX = mix(c[0], output[0], combine);
    const mergeY = mix(c[1], output[1], combine);
    s += `<g data-modern-aggregate="${j}" data-x="${mergeX.toFixed(4)}" data-y="${mergeY.toFixed(4)}">`;
    s += group(rect(mergeX - 12, mergeY - 8, 24, 16, RUST, 'white', 2, 1.5), context * (1 - combine));
    s += '</g>';
  }
  if (!n) s += text('schematic tokens', 135, 340, 15, MUTED);
  else s += text('schematic tokens', 200, 96, 13, MUTED);
  const vector = n
    ? String.raw`e_{\text{text}}=\begin{bmatrix}e_1&e_2&\cdots&e_d\end{bmatrix}^{\mathsf T}`
    : String.raw`\begin{bmatrix}e_1\\e_2\\\vdots\\e_d\end{bmatrix}`;
  // Keep KaTeX's own delimiter dimensions inside the full-size stage SVG.
  const vectorMath = m(vector, output[0], output[1], n ? 390 : 130, n ? 62 : 170, n ? 24 : 27, RUST)
    .replace(/<svg([^>]*?)width="([^"]+)" height="([^"]+)"/g,
      '<svg$1style="width:$2;height:$3" width="$2" height="$3"');
  if (!n) s += m(String.raw`e_{\text{text}}`, output[0], 109, 130, 50, 28, RUST);
  s += group(vectorMath, .15 + .85 * combine);
  s += text('one vector for the whole text', n ? 200 : 870, n ? 376 : 340, n ? 17 : 16, RUST);
  if (n) {
    s += text('Common training goal:', 200, 414, 15, MUTED);
    s += text('related texts have similar vectors.', 200, 438, 15, MUTED);
  } else {
    s += text('Common training goal: related texts have similar vectors.', 500, 397, 18, MUTED);
  }
  return s + '</g>';
}
