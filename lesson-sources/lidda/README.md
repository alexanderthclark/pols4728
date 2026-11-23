# LiDDA lesson source

This package builds the interactive lesson published from [`../../docs/lidda/`](../../docs/lidda/). It contains the React page, its CSS and equations, and the small numerical examples used in the explanation. The compiled lesson runs on a static host without a server or an account connection.

## Install and verify

Use Node.js 22.13 or newer. Dependency versions and their transitive resolutions are pinned in `package.json` and `package-lock.json`.

```bash
cd lesson-sources/lidda
npm ci
npm test
npm run typecheck
npm run build
```

The build writes `dist-static/`. It bundles React, KaTeX, the styles, and all math fonts locally. Asset URLs are relative, so the output can be hosted at `/pols4728/lidda/` or another directory. No CDN or application server is required. JavaScript is required for the interactive lesson.

## Update the generated course page

After reviewing a source change, run:

```bash
npm run publish:static
```

This command rebuilds the package and replaces the generated `../../docs/lidda/` directory with the new output. It only copies files in the local repository; it does not push, publish remotely, or change the repository's deployment workflow. Review the source and generated changes together before committing them. Make editorial and style changes in this source package rather than editing generated bundles.

## Files to maintain

- `app/page.tsx` organizes the lesson; `app/globals.css` supplies shared styles.
- `components/explainer/` contains the training examples, architecture diagram, scroll-driven readout, equations, and styles.
- `lib/attribution-model.ts`, `lib/attention-detail.ts`, and `lib/lesson-shapes.ts` define the fixed teaching example and padded tensor shapes. These parameters are illustrative; they are not fitted LiDDA estimates.
- `tests/` verifies attention arithmetic, padding, column aggregation, and normalized credit.
- `static/index.html` and `static/main.tsx` provide the standalone browser entry and initial section-link behavior.
- `public/favicon.svg` supplies the icon.
- `scripts/export-static.mjs` builds the standalone output; `scripts/publish-static.mjs` stages it under `docs/lidda/`.

The source intentionally excludes the original hosting configuration, authentication code, server scaffolding, and framework layout. Read the repository's `AGENTS.md` and design guidance before changing presentation. Browser review should cover forward and reverse scrolling, narrow screens, keyboard access, and reduced motion in addition to the numerical checks.

## Scope and source history

This lesson assumes students already know embeddings and attention. After concrete training examples, a two-branch diagram shows the sequence model and member/employer covariates. Two scroll stages show the attention matrices and their conversion into relative credit. Conversion prediction is summarized in the architecture rather than separate slides. The paper representation remains in an optional disclosure.

The paper specifies one self-attention layer but does not report the head count or residual/LayerNorm wiring. The illustration uses two heads and five positions (four real touches plus PAD). Summing incoming attention across valid queries and heads, then normalizing, is an explicit teaching convention for the paper's underspecified matrix-to-credit reduction.

The full earlier guide is preserved in the original Sites source commit `2c4e2f47cb0f4578326ad84e1affbfacc55f6509`. The course package contains the focused revision only.

The course presentation uses white backgrounds, the shared palette, direct labels, and outlines for matrices and output coordinates. The training journeys are synthetic; the architecture diagram distinguishes sequence inputs from context covariates. No character artwork is introduced.
