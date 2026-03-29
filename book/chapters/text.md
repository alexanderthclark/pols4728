(text)=
# Text

Text analysis maps unstructured language into representations that can be used for prediction, description, and inference.

## Topic objectives

- distinguish count-based and embedding-based text representations,
- understand baseline methods (bag-of-words, TF-IDF),
- connect classical NLP pipelines to transformer-based workflows.

## Data quality first

Text pipelines are sensitive to sampling and measurement bias.
Common risks include:

- resource bias (who is represented),
- incentive bias (performative communication),
- medium bias (platform constraints),
- retrieval bias (how texts were collected).

Method quality cannot rescue poor corpus design.

## Bag-of-words representation

Bag-of-words ignores token order and represents documents by term counts.
A corpus becomes a sparse document-term matrix `X \in \mathbb{R}^{n\times p}`.

Typical preprocessing:

- tokenization,
- case normalization,
- stop-word handling,
- optional lemmatization or n-grams.

## Word2Vec intuition

Skip-gram Word2Vec learns embeddings by predicting context words from a center word.
Words used in similar contexts get nearby vectors.

This moves from sparse count vectors to dense, learned representations.

## TF-IDF weighting

TF-IDF downweights ubiquitous words and upweights terms that are distinctive for a document.
For term `t` in document `d`:

$$
\text{tfidf}(t,d)=\text{tf}(t,d)\cdot\log\frac{N}{\text{df}(t)}.
$$

TF-IDF is often a strong linear baseline for classification tasks.

## Similarity and retrieval

Common similarity choices:

- cosine similarity (directional similarity of vectors),
- Euclidean distance (magnitude-sensitive),
- task-specific kernels.

For sparse text vectors, cosine similarity is usually the default.

## Topic modeling and BERTopic

Classical topic models use count-based generative assumptions.
BERTopic combines embeddings, clustering, and class-based TF-IDF summaries for interpretable topic labels.

## Transition to transformers

Classical representations are useful baselines, but contextual encoders now dominate many tasks.
See {ref}`transformers` for attention-based models and fine-tuning workflows.
