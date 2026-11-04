# Backtest EGM review and test record

Status: completed. Initial findings were reproduced and repaired; both fresh final reviewers found no open Major or Blocker. Regression, browser/export and startup checks passed.

This is an adapted post-implementation EGM review. See [scope](REVIEW_SCOPE.md) and [initial reviewer evidence](INITIAL_REVIEW.md). It is not the installed plugin's complete pre-implementation workflow. All four initial reviewers completed in fresh conversation contexts and confirmed their prescribed reads. The author handled reproductions, changes and tests.

## Finding traceability

| Findings | Resolution | Verification |
| --- | --- | --- |
| N1, A1 | Include a two-close path as a one-return final diagnostic window. An undefined Sharpe remains unavailable. | Remainders 0, 1 and 2 across one and two full windows; counts and compounded returns recover the whole path, including a terminal 50% loss. |
| S1 | Store allocation source separately from page-owned widget state, restore it on entry, and invalidate incompatible candidates. | Optimizer source, complete manifest and weights survive navigation; shrinking to one ticker clears incompatible candidate use and clamps momentum holdings. |
| N2, N3 | Share date normalization between prices and explicit targets; reject missing timestamps, keep last duplicates and sort dates. | UTC and America/New_York dates, missing price/target dates, and duplicate/unsorted observations. |
| R1 | Record every actual nonzero holding change and fee. Keep current holdings when only floating-point roundoff separates the entire target. | Tiny-sleeve ledger reconstruction, fee reconciliation and constant-price no-op reset. |
| R2 | Normalize the benchmark series before reconstructing the reference price frame. | Timezone/intraday benchmark succeeds; unavailable, invalid and non-overlapping references leave strategy results available; sparse benchmark aligns both baselines and discloses exclusions. |
| R3, R4 | Normalize tolerated sums above one; reject mismatched/duplicate target assets. | Flat-price cash conservation at maximum capital, no over-allocation, and clear asset-vector validation. |
| A2, S2 | Preserve original input prices and equity/cost exports with each of five bounded saved snapshots; add saved-run export selection. | Provider-revision test preserves A's inputs/outcomes while B becomes active, and reconstructs A's input fingerprint. Save-limit replacement/rejection and clear behavior are covered. |
| A4, N6 | Add independent accounting and comparator oracles and adversarial boundaries. | Replay each cash balance, quantity and equity from executions across four strategies and three seeded price paths with gaps, fees and yield. Test full A-to-B rotation formula, partial investment, scale invariance, quarter/year boundaries, momentum scheduling and nonzero-cost walk-forward reference equality. |
| C1–C6, N5 | Clarify targets, momentum universe/budget, initial schedules, benchmark entry, fold account timing, fallback objective and comparison scale. | Methodology and visible strategy/benchmark explanations updated; one-return fold and benchmark-before-trend-entry cases added. |
| N4, A3 | Document alternatives, defaults, bounds and history requirements. | Dedicated methodology tables reflect the current engine and UI choices. |

## Test design

Financial assertions use hand-derived equations, conserved return intervals, or independent replay of recorded trades. The seeded multi-path cases check invariants under different price histories; they do not use the simulator to calculate the expected account state. Comparator equality is appropriate when both sides explicitly request equal weights and the same schedule. No dependency or market-data credential was added.

The initial targeted tests failed before fixes. Tests for malformed API inputs are distinguished from ordinary UI workflows. Provider integration and browser checks use deterministic fixtures to exercise behavior independently of live service availability; they do not establish live Yahoo, NSE or SEC uptime.

## Remaining limits

The simulator remains daily, long-only and based on synthetic fractional adjusted-price units. Linear slippage is a user assumption, not measured execution quality. Current universes can have survivorship bias. Saved runs remain session-local unless exported. Signal outcomes can overlap. Reviewing or tuning on a later window introduces selection bias. No historical options quotes, margin, trained predictive model or live trading was added.

## Final evidence

- Four initial reviewers and two new final reviewers completed, with explicit read confirmations. See [final review](FINAL_REVIEW.md). No reviewer was counted as passing without a completed report.
- Full suite: **292 passed in 24.24 seconds**, including **51 additional cases** added during this review. Command: `.venv/bin/pytest -q`.
- Compilation: `.venv/bin/python -m py_compile streamlit_app.py common/*.py views/*.py` passed.
- Server: `.venv/bin/python scripts/check_streamlit_startup.py` passed health, HTML, JavaScript and favicon responses with gzip enabled.
- Chromium at 1440×1100 and 390×844: optimizer handoff -> Performance -> Backtest retained the exact run; submitted a trend strategy; changing to two tickers invalidated the old candidate; saved a second run; explicitly selected the first saved run; downloaded original price/equity/execution CSVs; verified downloaded original-price SHA-256 against its recipe after the active universe changed; opened signal charts; no browser exceptions or mobile horizontal overflow.
- Browser data was deterministic. These checks exercise real UI controls and download bytes, without claiming provider uptime or live execution accuracy.
- `git diff --check` passed. Engine version advanced to 1.1 because accounting/normalization semantics changed.


## Deployment follow-up

The reported Streamlit Cloud import failure exposed a cached-module update path outside the initial fresh-process checks. See [the reproduction, fix and Python 3.13 verification](CLOUD_IMPORT_FIX.md). Five new regressions now cover that path through both app entrypoints.
