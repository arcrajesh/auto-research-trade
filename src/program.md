# autoresearch-trade

Autonomous research for trading strategies.  The agent iterates on
`src/strategy.py`, backtests against a fixed harness, and keeps only
improvements.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun30`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small.  Read these files for full context:
   - `src/README.md` — layout overview.
   - `src/data_pipeline.py` — fixed constants, data download, feature helpers, evaluation harness. **Do not modify.**
   - `src/backtest.py` — fixed backtesting engine and metric computation. **Do not modify.**
   - `src/strategy.py` — the file you modify. Strategy logic, indicators, position sizing.
4. **Verify data exists**: Run `uv run src/data_pipeline.py` (or `python -m src.data_pipeline`) to confirm data downloads correctly and splits are populated.  Data is cached in `~/.cache/autoresearch_trade/`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a backtest on the **validation split** (2020-01-01 → 2022-12-31).  You launch it as:

```bash
uv run src/backtest.py
```

The backtest uses realistic execution costs (0.05 % commission, 5 bps slippage) and a $100 000 starting capital.

**What you CAN do:**
- Modify `src/strategy.py` — this is the only file you edit.  Everything is fair game: indicators, entry/exit signals, position sizing, risk management, parameters.

**What you CANNOT do:**
- Modify `src/data_pipeline.py`.  It is read-only.  It contains the fixed data loading, feature computation, and evaluation harness.
- Modify `src/backtest.py`.  It is read-only.  It contains the fixed backtesting engine, metric computation, and composite score.
- Install new packages or add dependencies beyond what is in `pyproject.toml`.
- Introduce lookahead bias or data leakage (use only past/present bars; shift signals where necessary).
- Run exhaustive parameter grid searches.  You may do targeted experiments but not brute-force sweeps.

**The goal is simple: maximise the composite_score.**

The composite score is:
- `sharpe` when `max_drawdown ≤ 0.25`
- `sharpe - 10 * (max_drawdown - 0.25)` when `max_drawdown > 0.25`

Higher composite_score = better.

**Secondary constraints** (not hard-gated, but respected):
- Minimum trade count: aim for ≥ 10 trades on the validation split.
- Maximum annualised turnover: keep turnover reasonable (< 50×).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the backtest as-is.

## Output format

Once the backtest finishes it prints a summary like this:

```
---
composite_score:   0.432100
sharpe:            0.432100
annualized_return: 0.085200
max_drawdown:      0.152300
mar:               0.559700
win_rate:          0.4500
avg_rr:            1.8200
avg_holding_days:  12.3
trade_count:       28
turnover:          3.2100
```

You can extract the key metric from the log file:

```bash
grep "^composite_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	composite_score	sharpe	max_drawdown	trade_count	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score achieved — use 0.000000 for crashes
3. sharpe ratio — use 0.000000 for crashes
4. max_drawdown — use 0.000000 for crashes
5. trade_count — use 0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	composite_score	sharpe	max_drawdown	trade_count	status	description
a1b2c3d	0.432100	0.432100	0.152300	28	keep	baseline
b2c3d4e	0.523400	0.523400	0.143200	32	keep	add RSI filter
c3d4e5f	0.312000	0.312000	0.189000	24	discard	switch to EMA crossover
d4e5f6g	0.000000	0.000000	0.000000	0	crash	broke imports
```

NOTE: do not commit the `results.tsv` file — leave it untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `src/strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run src/backtest.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^composite_score:\|^sharpe:\|^max_drawdown:\|^trade_count:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If composite_score improved (HIGHER is better), you "advance" the branch, keeping the git commit
9. If composite_score is equal or worse, you `git reset` back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should complete quickly (seconds to a minute for a backtest). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous.
