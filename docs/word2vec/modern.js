// The requested closing slide is a simple, readable list.
export const modernScene = {
  chapter: '06 · Embeddings in practice',
  title: 'In practice, you will probably use a more advanced pretrained model.',
  eq: '',
  caption: '',
  detail: '',
  alt: '',
};

export function renderModern(headingLevel = 3) {
  return `<h${headingLevel}>Modern alternatives</h${headingLevel}>
    <ul>
      <li>BERT / BERT-style models</li>
      <li>Text embedding models, such as OpenAI’s</li>
    </ul>`;
}
