// The closing slide introduces two useful advances beyond Word2vec.
export const modernScene = {
  chapter: '06 · Embeddings in practice',
  title: 'Improvements on Word2Vec',
  eq: '',
  caption: '',
  detail: '',
  alt: '',
};

export function renderModern(headingLevel = 3) {
  return `<ul>
      <li>
        <h${headingLevel}>BERT / BERT-style models</h${headingLevel}>
        <p>More flexible architectures use word order and surrounding context, so the same word can have different representations in different sentences.</p>
      </li>
      <li>
        <h${headingLevel}>Text embedding models, such as OpenAI’s</h${headingLevel}>
        <p>Represent a whole passage or document with one vector, allowing us to compare texts rather than individual words.</p>
      </li>
    </ul>`;
}
