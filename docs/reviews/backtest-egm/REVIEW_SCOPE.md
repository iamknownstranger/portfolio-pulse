# Backtest review scope

This is a post-implementation application of the Elephant–Goldfish approach described at https://egm.dev/ and https://github.com/google/elephant-goldfish-model. It is not a claim to have run the installed EGM plugin's complete pre-implementation gate sequence.

The context-aware author resolves findings. Independent reviewers are spawned with `fork_turns: "none"`; they receive file paths, review standards and a focused task, without the author conversation. Parent-turn isolation does not establish exclusion of host instruction files. Reviewer reports must confirm the files read. Missing reports do not count as passing reviews.

The baseline is commit `b516a98`. The target contract is `docs/BACKTEST_RESEARCH.md`, with README and current source as explicit context where allowed. Comprehension review reads only the target document. The other reviewers examine numeric accounting and execution timing, state/persistence and provenance, and architecture/document-to-code drift. Reviews are read-only; the author reproduces findings, writes regression tests, fixes code and records resolutions. A fresh follow-up checks substantive fixes.

Release criteria:

- No unresolved finding that changes results, leaks future data, misstates the run configuration, loses saved evidence or crashes an ordinary supported workflow.
- Meaningful deterministic tests for accounting and execution boundaries, missing/irregular prices, benchmark alignment, chronological folds, state transitions and export provenance.
- Financial assertions checked against independent arithmetic or replayed executions, rather than duplicating engine code.
- Full regression suite, source compilation and real HTTP/static-asset startup checks pass.
- Browser checks cover strategy submission, navigation, saved-run/export behavior and desktop/mobile presentation using deterministic prices.
- Record remaining data/methodology limits and exact validation evidence before publishing the authorized branch and draft PR.
