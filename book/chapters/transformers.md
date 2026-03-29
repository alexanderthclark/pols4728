(transformers)=
# Transformers

Transformers are sequence models built around attention, enabling state-of-the-art performance in language and many other domains.

## Topic objectives

- explain encoder, decoder, and encoder-decoder variants,
- understand self-attention as token-to-token information mixing,
- choose between fine-tuning, retrieval augmentation, and prompt-only usage.

## Architecture families

- Encoders (for representation/classification tasks; e.g., BERT-style)
- Decoders (autoregressive generation; GPT-style)
- Encoder-decoders (sequence-to-sequence mapping)

## Self-attention core

Given token representations `X`, one attention head computes

$$
\text{Attn}(X)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
$$

with learned projections `Q=XW_Q`, `K=XW_K`, `V=XW_V`.

Interpretation:

- right-multiplication by projection matrices transforms feature dimensions,
- left-multiplication by the attention matrix mixes information across tokens.

## Transformer block (high level)

A standard block combines:

1. multi-head attention,
2. residual connections,
3. layer normalization,
4. position-wise MLP.

Stacking blocks yields contextual token representations.

## From scratch vs fine-tuning

Practical default for social-science workflows:

- start from a pretrained model,
- add a task head (classification/regression/generation),
- fine-tune with validation-based stopping.

Train from scratch only when vocabulary/domain constraints justify it.

## Retrieval-augmented generation (RAG)

RAG combines retrieval with generation:

1. embed query,
2. retrieve relevant passages,
3. pass context + query to generator.

Benefits:

- better factual grounding,
- lower hallucination risk,
- easier source attribution.

## Operational cautions

- evaluate with task-specific metrics and qualitative error audits,
- monitor prompt sensitivity and distribution shift,
- separate model capability from pipeline quality (chunking, retrieval, indexing).
