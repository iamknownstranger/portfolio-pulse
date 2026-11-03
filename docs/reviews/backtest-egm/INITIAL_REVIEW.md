# Initial fresh-context reviews

Baseline: `b516a98`. Four independent reviewers completed. Each used `fork_turns: "none"`, confirmed required reads and stayed read-only. Public EGM personas/standards were retrieved from google/elephant-goldfish-model; the local source checkout records commit `464db03`. The plugin was not installed or represented as an executed plugin command.

## Numeric/design critic

Confirmed complete reads of the design-critic persona, feature-spec standard, target methodology, README, backtesting, portfolio, optimization, backtesting tests and optimizer tests.

- **N1 / Major:** `rolling_windows` drops a final one-return path because it requires three equity points. A terminal 50% loss can disappear from the chart/export. Test remainder counts and compounded-return conservation.
- **N2 / Nit:** Direct `simulate_targets` normalizes prices but not matching timezone-aware target dates, silently removing decisions. Normalize both inputs consistently.
- **N3 / Nit:** A missing timestamp survives price preparation and can corrupt calendar-time cash accrual. Reject missing dates.
- **N4 / Major under supplied feature-spec rubric:** The methodology lacks alternatives with concrete trade-offs. Add the simulator-versus-fixed-weight and session-versus-durable-storage decisions.
- **N5 / clarification:** Later fold account returns include the preceding holdings until a new target executes. Explain the boundary and verify a one-return fold.
- **N6 / coverage:** Add quarter/year transitions, momentum warm-up between scheduled decisions, partially invested cash and mixed buy/sell cost cases.

## State/implementation-readiness reviewer

Confirmed complete reads of the readiness persona/standard, methodology and all supplied state, UI, data and test files; inspected installed Streamlit state cleanup behavior.

- **S1 / Major:** Optimizer handoff -> another portfolio page -> Backtest loses `_bt_source` because it is page-owned. The default source then replaces the active optimizer run with current-model weights. Persist source selection independently of widget state and test the full navigation sequence.
- **S2 / optional enhancement:** Saved runs retain recipes and outcomes but cannot export their older original input prices after another experiment replaces the active inputs. Preserve bounded export bundles or state the limitation explicitly.

## Architecture/codebase-drift reviewer

Confirmed complete reads of zoom-out/codebase-details personas, zoom-out/re-implementation standards, target methodology, README, product review and relevant production/test/CI source. No substantiated architectural-boundary, duplicate-calculation or unnecessary-abstraction defect.

- **A1 / Major:** Same terminal-return omission as N1.
- **A2 / Major:** The saved-input export limitation undermines the documented reproducibility workflow. Test save A, provider revision or universe change, run B, then export A and verify its fingerprint.
- **A3 / Major:** Observable defaults, parameter bounds and minimum-history empty states are missing from the methodology.
- **A4 / Major coverage gap:** Independently verify simultaneous sales/purchases with funded costs and the walk-forward equal-weight reference under nonzero costs. Existing tests did not exercise these central accounting/comparator contracts.

## Document-only comprehension reviewer

Read only the comprehension persona and target methodology, without source or external pages.

**Paraphrase of the proposal:** A daily equity simulator carries shared portfolio inputs through delayed execution, drifting holdings, costs, cash, benchmark comparisons, diagnostics and saved research. Fixed-rule chronological checks and rolling allocation estimation answer different questions. Options execution, brokerage accounts and stronger statistical claims remain outside scope.

**Paraphrase of the existing system:** Streamlit portfolio analytics reset to fixed model weights at each common close without costs. Shared components supply prices, allocations, metrics, filters and navigation. Current options chains and current radar scores are insufficient historical trading evidence.

- **C1 / Major:** Define active target validity, partial-cash treatment, missing components and over-allocation behavior.
- **C2 / Major:** State momentum's investment budget and whether zero-source-weight tickers remain eligible.
- **C3 / Major:** Define whether the first selected close is an initial periodic decision when a window starts mid-period.
- **C4 / Major:** Distinguish the benchmark's one-close execution lag from entry on the strategy's actual first invested date.
- **C5 / Nit:** Explain which objective replaces a failed maximum-Sharpe solve.
- **C6 / Nit:** Specify the normalized saved-comparison starting value.

## Author reproductions and additional findings

The author wrote regression cases before repairs. The first focused run produced **9 failures and 4 passes**: terminal-loss omission for two different remainders, two timezone target cases, missing price date, missing target date, omitted tiny holding in the execution ledger, source-navigation reset, and normalized-benchmark simulation failure.

- **R1:** The ledger threshold omitted small actual holding changes and their fees while still applying them to the account. A $1 trillion model with a 1e-13 sleeve acquired that sleeve without an execution row. Every applied nonzero change must remain reconstructible.
- **R2:** Backtest aligned benchmark dates using cleaned prices, then rebuilt reference inputs from the uncleaned series. A timezone/intraday-indexed benchmark made the reference empty and blocked otherwise available strategy results.
- **R3:** A tolerated target sum of `1 + 5e-11` created $50 of unfunded equity at $1 trillion capital with zero fees. Normalize accepted floating-point overshoots rather than relying on a non-negative cash clamp.
- **R4:** Extra target columns were silently dropped. Require the exact price-asset set and report missing/duplicate/unknown assets clearly.

A separate pre-fix check produced **four failures** for the tolerance/target-column cases. Two of those were existing rejections with poor error classification rather than silently accepted targets. These are recorded as validation-message improvements, not independent financial defects.
