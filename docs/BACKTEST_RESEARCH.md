# Backtest research and implementation

Reviewed 3 November 2026 against Portfolio Pulse `c4fd1dd` and Backtest.ai's public product and methodology pages.

## Product findings

[Backtest.ai's product overview](https://backtest.ai/product) describes an options research pipeline: construct a signal, inspect its evidence, optionally fit a model, simulate a strategy, combine allocations, and monitor an account. Its strongest transferable idea is the continuity of that workflow: a result should retain its inputs, assumptions, benchmark, and diagnostic evidence as the researcher moves between views.

The accessible site describes a closed beta. This review inspects its public documentation, not its application internals or a working broker integration. Marketing descriptions are not independent validation of execution quality.

Its [backtest results reference](https://backtest.ai/docs/backtests/reading-backtest-results) places the equity curve, execution/position records, risk statistics, and benchmark together. That is more useful than a single attractive return number. Its [execution assumptions](https://backtest.ai/docs/backtests/backtest-assumptions-and-realism) also make fill conventions and exclusions inspectable. These suggest explicit costs, an auditable ledger, and a visible methodology contract in Portfolio Pulse; they do not justify claiming that close-only data reproduce an options market.

The [EDA documentation](https://backtest.ai/docs/signal-analysis-models-and-validation/reading-the-eda-charts) uses signal history, distribution, forward-outcome scatter, and deciles as complementary views. Portfolio Pulse can compute those from its existing equity price histories. Its [validation documentation](https://backtest.ai/docs/signal-analysis-models-and-validation/model-outputs-and-validation) distinguishes evaluation on later observations from a model's fit to its training history. Portfolio Pulse's allocation optimizer benefits directly from that distinction.

Important limits are documented in Backtest.ai's own assumptions: historical simulations do not account for every market mechanism. Portfolio Pulse should likewise disclose its actual capabilities instead of borrowing the stronger broker, options, margin, or deployment claims of another product.

## Portfolio Pulse findings

The repository is a Streamlit research application with 11 existing dashboards, a shared model allocation, Yahoo adjusted-price adapters, Plotly presentation, and pure calculation modules. Portfolio analytics currently rebalance to fixed model weights at every common close with zero costs. That is a useful descriptive allocation model but does not simulate the execution of a decision made from a closing price.

`common/portfolio.py` already provides consistent return statistics, initial-wealth drawdowns, benchmark alignment, weights, and attribution. `common/optimization.py` computes long-only constrained historical estimates. `common/sidebar.py` owns persistent filters, and `common/navigation.py` owns explicit routes. These are strong foundations: the backtester can reuse the data, weights, metrics, navigation, and theme while giving execution its own testable contract.

The main gaps were an execution/cash ledger, holding drift, periodic strategy schedules, cost funding, explicit signal timing, saved research snapshots, and chronological allocation validation. The current options adapters retrieve available chains; they do not provide historical contract quotes. Smart Money Radar's accumulation and SEC layers have distinct data timing and cannot be treated as a historical trading signal merely by reading their latest scores.

## Feature placement

| Research idea | Implemented location | Behavior |
| --- | --- | --- |
| Hypothesis to evidence | **Backtest**, `/backtest`, within Equities — Linear | Shared portfolio filters and weights feed an independent daily stock/ETF simulation. |
| Strategy templates | Backtest settings | Buy & hold, periodic rebalance, moving-average trend filter, positive momentum rotation. |
| Execution realism appropriate to available data | Backtest engine and settings | Next-common-close execution, drifted holdings, fractional adjusted-price units, user-entered commissions/slippage, self-funded costs, optional cash yield. |
| Benchmark context | Backtest Results | Same-calendar, same-capital buy & hold reference with the same entry delay and per-side cost inputs. The existing benchmark choice is preserved. |
| Inspectable outputs | Results and Trade ledger | Net and zero-cost growth, drawdowns, invested/cash allocation history, dated decisions and executions, notional and separate costs. |
| Exploratory signal evidence | Backtest and **Performance Analytics → Signal research** | Trailing momentum or moving-average distance, time series, histogram, scatter and decile outcomes, sample counts, descriptive correlation and CSV. |
| Chronological diagnostics | Backtest → Validation | Fixed rules retain holdings/history across an initial/later split; non-overlapping 63-observation windows expose performance instability. This does not train a model. |
| Genuine rolling estimation | **Rebalancing & Optimization → Forward validation** | Each fold estimates weights on trailing prices only, executes next close, and evaluates later returns. Includes equal weight on the same schedule, caps, costs, fold dates, fallback notices and allocation exports. |
| Allocation continuity | Rebalancing → **Backtest this target** | Copies the proposed weights into Backtest without changing the portfolio model. Clearly identifies the copied target as fitted on its source history. |
| Cost awareness before changing weights | Rebalancing trade plan | Displays an estimated cost on both buy and sell notional. |
| Loss duration and recovery | **Risk Wall** and Backtest | Peak/trough/recovery episodes, deepest losses, calendar days underwater, unrecovered flags and CSV. |
| Reproducible research | Backtest → Saved runs & exports | Up to five session snapshots, named normalized comparisons, JSON recipes, price fingerprints, input prices, net/zero-cost equity, and executions. |
| Discovery from the overview | Portfolio Pulse → Your next move | Direct route into backtesting the current allocation. |
| Options research boundary | Option Chain methodology expander | Explains why a current chain cannot establish historical fills; links to the available daily equity research. |

## Execution contract

1. Start at the first common adjusted close with the selected capital entirely in cash.
2. At close **t**, compute a strategy target using prices through **t** only. An all-missing target means keep the current holdings. Zero target weights mean cash.
3. Before trading at the next common close **t+1**, mark the existing holdings at that close and accrue cash yield over elapsed calendar days. The price movement from t to t+1 belongs to the previous holdings.
4. Rebalance to the pending target. Solve `post_cost_equity + cost_rate × absolute_traded_notional = pre_trade_equity` before sizing positions. This funds both sides' costs without borrowing cash.
5. Record decision date, execution date, asset, signed synthetic units, adjusted reference price, absolute notional, commissions, slippage and reason. Record end-of-close equity, weights and costs.
6. The final decision has no subsequent execution close. Leave existing positions marked at the final close; do not invent a closing transaction or liquidation fee.

**Schedules.** Buy & hold decides once at the first observed close. Periodic strategies decide daily or on the first observed close of a new week, month or quarter, then execute one common observation later. Trend checks every close: each source-weight sleeve invests only above its trailing moving average, otherwise stays cash. Momentum ranks positive trailing returns, selects up to N assets, and equally weights the selected names. Ties follow the stable input-column order. If fewer than N qualify, they share the invested allocation equally. Trend needs `lookback` prices; momentum needs `lookback + 1`. Warm-up occurs inside the selected window and stays in cash; scheduled momentum decisions may occur later still.

**Costs and data.** Cost inputs are basis points per side on absolute traded notional. Slippage is a separate cash debit against an adjusted-close reference price, not an observed spread or a calibrated fill price. Adjusted-price units are synthetic fractional model quantities, not broker share quantities. Common complete dates are selected before signals or returns; missing values are never forward-filled. Benchmark alignment can remove additional dates, which is disclosed. Cash yield uses effective annual compounding over calendar days. Taxes, liquidity limits, nonlinear market impact, cash flows and forced terminal liquidation are absent.

**Metrics.** Reuse the existing arithmetic-return metrics: sample volatility and ratios scale by 252 observations, the selected risk-free rate is converted to a daily effective rate, and CAGR uses elapsed calendar time. Drawdowns include starting capital. An execution count is a buy/sell asset event, not a round-trip win rate. Net-versus-zero-cost terminal wealth difference includes foregone compounding and is therefore different from the sum of cash costs.

## Validation contract

A fixed-strategy split partitions return observations chronologically. The split close is the later window's baseline; no return belongs to both partitions. Holdings and trailing signal state continue through the boundary. The later window stops being an untouched holdout once it influences design choices. A target optimized on the entire window remains in-sample even if its return path is subsequently split.

Walk-forward allocation validation is separate. For each decision close, the optimizer sees exactly the trailing training observations and the price baseline needed to form their returns. It chooses new weights under the selected objective and cap. They execute at the next common close. Subsequent return windows are disjoint; the final one may be shorter. The initial training period is excluded from the performance curve. The equal-weight comparator follows the identical decision/execution schedule with identical capital and costs. A maximum-Sharpe fallback retains its actual objective and notice per fold.

Signal EDA pairs the value at decision close t with the asset return from execution close t+1 through t+1+h. Unobservable tails and insufficient warm-up are dropped. Adjacent forward horizons overlap: displayed counts are not independent trials. Deciles use value boundaries, retain ties, and can yield fewer than ten buckets; a constant signal has no decile chart. These charts do not establish significance or remove selection/survivorship bias.

## Research persistence

Saved runs live in Streamlit session state, including their equity, statistics, recipe and execution ledger. Five runs bound memory and keep comparisons readable. Comparisons preserve original dates, normalize starting capital and expose date differences. They are not ranked as equivalent experiments.

A recipe contains an engine version, configuration, model weights, benchmark, risk-free rate, requested and observed dates, source label and assumptions. `price_sha256` hashes the exact exported price CSV bytes; `run_id` hashes the complete recipe. The combined input-price export includes `__benchmark__` when a reference is available. A recipe alone cannot reproduce revised provider histories: keep the accompanying prices. Settings do not create broker transactions or persist beyond the session without export.

## Deliberately deferred

- Historical options trading needs contract identifiers, dated quote history, point-in-time selection, spread and settlement conventions, assignment treatment and validated margin accounting. Current chains are insufficient.
- A custom signal language or AI strategy generator needs a typed, restricted expression interface and computation provenance. There is no arbitrary Python execution or ungrounded AI claim in this change.
- Model training, coefficient diagnostics, statistical significance and multiple-testing corrections need a larger statistical design than charting a holdout.
- Paper/live promotion needs broker integration, durable positions and an account ledger, reconciliation, controls and monitoring. Session-saved research is not a paper-trading account.
- Full-universe point-in-time constituent and delisting history is required to address survivorship bias. Selecting today's tickers and using adjusted price history does not solve it.
- Automatically backtesting Smart Money Radar's latest scores would leak current cross-sectional information into the past. Its price/volume and SEC evidence need historical snapshots, publication lags and rolling recomputation first.

## Verification

Tests cover independent wealth arithmetic, delayed entry, holding drift, periodic schedules, self-funded entry/exit fees, cash interest, invalid targets, warm-up, future-data perturbations, missing common dates, signal/outcome alignment, tied deciles, drawdown recovery, disjoint validation returns, rolling training boundaries and stable run manifests. Streamlit integration tests cover the Backtest route, strategy submission, save/navigation persistence, copied optimizer weights, applied-filter changes, walk-forward output, unavailable benchmarks and short histories. The full existing suite, source compilation and real HTTP/static-asset startup check remain part of validation. Browser review uses deterministic prices so UI and routing checks do not depend on provider availability.
