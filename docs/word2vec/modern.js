// A practical roadmap, not a pipeline from Word2vec to another model.
// The two branches show possible representation units for different tasks.
export const modernScene = {
  chapter: '06 · Embeddings in practice',
  title: 'Word2vec gives the intuition.',
  eq: String.raw`\text{prediction}\longrightarrow\text{learned representations}`,
  caption: 'In practice, you will often use a more advanced pretrained model.',
  detail: 'Your task may need richer representations of individual words—or one vector for a passage or document. OpenAI’s embeddings API is one example for text.',
  alt: 'Word2vec is a classical example that builds intuition. A separate application diagram branches from choosing a pretrained model to two possible representation units: words in context, illustrated by BERT-style models, and passages or documents, illustrated by text-embedding models such as those available through OpenAI’s API. Scrolling first extends the word branch, then the text branch; both remain visible at the end.',
};

export function renderModern(t, n, helpers) {
  const {m, text, line, rect, group, ease, mix, BLUE, RUST, INK, MUTED, RULE} = helpers;
  const word = ease((t - .08) / .4);
  const document = ease((t - .5) / .4);
  let s = '<g data-modern="practice">';

  // This teaching example is deliberately unconnected to the application
  // diagram: its vectors are not an input to the pretrained models.
  if (n) {
    s += text('Word2vec', 200, 34, 27, INK);
    s += text('A classical example', 200, 60, 17, MUTED);
    s += text('Learn the intuition', 200, 88, 19, BLUE);
    s += line(20, 112, 380, 112, RULE, 1);
    s += text('In an application', 200, 143, 17, MUTED);
    s += rect(105, 163, 190, 55, RULE, 'white', 3, 1.3);
    s += text('Pretrained models', 200, 197, 20, INK);
  } else {
    s += text('Word2vec', 145, 157, 30, INK);
    s += text('A classical example', 145, 191, 18, MUTED);
    s += text('Learn the intuition', 145, 235, 21, BLUE);
    s += line(290, 55, 290, 365, RULE, 1);
    s += text('In an application', 650, 40, 18, MUTED);
    s += rect(340, 170, 215, 80, RULE, 'white', 3, 1.3);
    s += text('Pretrained models', 447.5, 203, 21, INK);
    s += text('chosen for your task', 447.5, 229, 16, MUTED);
  }

  const branches = [
    {id: 'word', progress: word, color: BLUE,
      start: n ? [175, 218] : [555, 194],
      end: n ? [100, 266] : [650, 116],
      symbol: String.raw`v_{\text{word}}`},
    {id: 'text', progress: document, color: RUST,
      start: n ? [225, 218] : [555, 226],
      end: n ? [300, 266] : [650, 304],
      symbol: String.raw`v_{\text{text}}`},
  ];

  for (const branch of branches) {
    const {id, progress, color, start, end, symbol} = branch;
    const x = mix(start[0], end[0], progress);
    const y = mix(start[1], end[1], progress);
    // Actual arrow endpoints change with scroll position. The vector follows
    // the extending branch, and reverse scrolling restores both geometries.
    s += `<g data-modern-branch="${id}">`;
    s += group(line(...start, x, y, color, 1.7, true), progress);
    s += '</g>';
    s += `<g data-modern-vector="${id}">`;
    s += group(m(symbol, n ? x : x + 55, n ? y + 28 : y, 100, 42, n ? 27 : 31, color), progress);
    s += '</g>';

    let labels = '';
    if (id === 'word') {
      labels += text('Words in context', n ? 100 : 780, n ? 339 : 91, n ? 18 : 22, color, n ? 'middle' : 'start');
      labels += text('BERT-style models', n ? 100 : 780, n ? 365 : 121, n ? 14 : 17, MUTED, n ? 'middle' : 'start');
      labels += text('“support” in a sentence', n ? 100 : 780, n ? 389 : 148, n ? 13 : 16, INK, n ? 'middle' : 'start');
    } else {
      labels += text('Passages / documents', n ? 300 : 780, n ? 339 : 279, n ? 17 : 21, color, n ? 'middle' : 'start');
      labels += text('Text-embedding models', n ? 300 : 780, n ? 365 : 309, n ? 14 : 17, MUTED, n ? 'middle' : 'start');
      labels += text('e.g. OpenAI’s API', n ? 300 : 780, n ? 389 : 336, n ? 13 : 16, INK, n ? 'middle' : 'start');
    }
    s += group(labels, progress);
  }

  s += text('The task determines what you embed.', n ? 200 : 650, n ? 434 : 397, n ? 16 : 18, MUTED);
  return s + '</g>';
}
