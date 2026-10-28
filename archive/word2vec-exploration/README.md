# Word-vector exploration, retained for future use

The instructor removed the section beginning “Now use the word vectors” from
the active Word2vec lesson on October 26, 2026. `section.html` preserves the
complete original HTML fragment, including the introduction, ten-scene stage,
qualifications, sources, and reading version. It is outside `docs/`, so GitHub
Pages does not publish it.

Its unchanged implementation remains in `docs/word2vec/`: `exploration.js`,
`exploration.css`, `math.js`, `projection.js`, and `generate_projection.py`.
The numerical and rendering tests remain runnable there. The original combined
lesson is also available at commit `ca23823142ffe26c9076f4a6565e2053a1ff1a28`.

To restore it, insert `section.html` after the original afterword and before
`noscript` in `docs/word2vec/index.html`, then restore these head entries:

```html
<link rel="stylesheet" href="exploration.css">
<script type="module" src="exploration.js"></script>
```

For a separate lesson, reuse the fragment and modules with the common
`style.css` and bundled KaTeX assets, and adapt its internal return anchors.
The archive hash is recorded in `docs/word2vec/clarity-changes.json`.
