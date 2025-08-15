# tufte_midterms_readme.md

This bundle gives you:
- Tufte’s eight midterm rows **exactly as printed** (for reference).
- A **single, consistent** real disposable personal income (DPI) per‑capita change measure you can use from **1938 to 2022**.
- **Presidential approval** for **every** midterm (Late‑October when available; otherwise closest October; else September), with the timing noted.

## File
- `tufte_midterms_master_full.csv` — one row per midterm (1938–2022).

## Columns
- `year` — midterm year.
- `pres_party` — President’s party (“D”/“R”).
- `in_original` — `True` for Tufte’s eight rows (1938, 1946, 1950, 1954, 1958, 1962, 1966, 1970).
- `vote_share_incumbent`, `normal_vote_prev8`, `vote_loss` — **exact** Table 1 values for Tufte’s rows; left open for others (fill later from Brookings/Clerk if you’re testing forecasts of vote shares).
- `pres_approval` — APP Gallup approval at the midterm (Late Oct preferred; otherwise closest), with the chosen window in `approval_window`. Tufte’s eight use his exact values.
- `delta_rdi` — Tufte’s original Δ real DPI pc in **1958 dollars** (reference only; filled on 8 rows).
- `dpi_pc_delta_ch2017` — **Annual change** in real DPI pc (chained‑2017$, NSA), from BEA/FRED **A229RX0A048NBEA**. Use this as the single “pocketbook” regressor uniformly across the full span.
- `dpi_pc_pct_yoy` — **Annual percent change** counterpart (same series).

## Notes
- We intentionally do not attempt to extend Tufte’s fixed‑base 1958‑$ series after 1970; instead we use the BEA chain‑dollar series so that **one definition** applies everywhere.
- If you need post‑1970 `vote_share_incumbent` and `normal_vote_prev8`, compute them from Brookings **Table 2‑2** (“% of all votes” → two‑party) and the House **Clerk** (for 2020/2022 coverage).

## Suggested usage
- Replicate Tufte by fitting on the eight originals (`vote_loss` ~ `delta_rdi` + `pres_approval`).
- Test out‑of‑sample on 1974–2022 by swapping in `dpi_pc_delta_ch2017` for the economic regressor (keep approval).

