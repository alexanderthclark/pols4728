# Tabular foundation models

Buildless scroll-driven course lesson for POLS 4728, following Christoph Molnar's *Tabular Foundation Models*. White textbook layout, Latin Modern type, course palette, torso-less hero illustration. The 18-scene story tracks labeled context, global network weights, synthetic task pretraining, simplified row/column attention, and a computed Bayesian example.

Run `npm run build` to regenerate the no-JavaScript reading version, `npm test` for Bayesian arithmetic and reversible visual-state checks, and `npm run dev` for a local preview. Source files in this directory are served directly. All core teaching points are reached by native scrolling; chapter links offer shortcuts. `?motion=reduce` or system reduced-motion preference uses complete visual states at logical thresholds.

## Numerical example

Two possible worlds, with equal prior mass. World A assigns P(voted | high interest)=.8 and P(voted | low interest)=.2; B reverses them. Labels are independent given the shared world and fixed features. Observe 0, 1, 2, then 4 high-interest respondents who voted. The page computes posterior weights and the predictive probability for another high-interest respondent. No TabPFN model is run. Animation interpolation is visual interpolation between exact results, not fractional-data Bayesian updating. Other probability bars are explicitly schematic.

## Sources and attribution

Molnar's book (https://tabularfoundationmodels.com/) is the main reference, particularly chapters 2–5, with application and limitation context from chapters 7 and 12. The text is original exposition, and the two-world example and visual diagrams are original. The book is CC BY-NC 4.0. The PFN paper and TabPFN Nature paper are linked in the lesson. Version-specific limits and benchmark claims are deliberately omitted. Source references checked 2026-11-22.

## Design choices

Course design reference: the existing POLS 4728 Open textbook guide. Navy denotes the model and hypothesis A; rust denotes weight updates, held-out evaluation, and contrasting hypothesis B (direct labels distinguish these uses); teal denotes targets and predictive probabilities. Table grids are retained to convey rows and columns. Narrow screens use compact diagrams with larger labels rather than shrinking desktop SVGs. Locally bundled Latin Modern and KaTeX license files accompany their assets.
