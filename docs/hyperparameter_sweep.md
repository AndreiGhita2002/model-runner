# Hyperparameter Sweep — Agent Instructions

Run this as a Claude Code session on the server (fisherman). Start it, detach tmux, come back later.

```
claude -p "$(cat docs/hyperparameter_sweep.md)"
```

---

## Goal

Find the best values for the Shisha optimizer's three hyperparameters:
- **deep_alpha** — patience before moving to the next stage (default: 10)
- **sibling_alpha** — how many stages to try before declaring optimum (default: 2)
- **tolerance** — fraction of the best throughput that counts as "within noise" (default: 0.01)

"Best" means highest requests_per_second averaged across all 4 models.

## Context

- The zero-rebalance bug (runs 16-19) has been fixed. The optimiser now rebalances correctly.
- The baseline to beat is the seed config (no rebalancing). If rebalancing makes things worse, the parameters are bad.
- Rebalance interval is fixed at 10 (don't change this).
- The server is CPU-only (32 cores, gloo backend, 4 ranks, 8 OMP threads per rank).
- With rebalance_interval=10 and n_microbatches=32, you need at least ~3200 requests for 10 optimiser calls. Use REQUEST_NUM=1000+ for meaningful results.

## How to run an evaluation

```bash
make eval ALPHA=<deep_alpha> TOLERANCE=<tolerance> REQUEST_NUM=<number of requests>
```

Notes:
- `ALPHA` sets deep_alpha. sibling_alpha is currently hardcoded (default 2) — to change it you need to edit `model_runner/pipeline_optimizer.py` line ~372.
- Use `REQUEST_NUM=1000` for sweep runs. Use 3000+ for final validation.
- Output goes to `data/runs/` as a timestamped JSON file.
- Each run takes a few minutes with 1000 requests. If values look good, increase to 3000 for validation.

## How to analyse a run

After each run, check the latest JSON file in `data/runs/`:

```python
import json, glob
latest = sorted(glob.glob("data/runs/*.json"))[-1]
with open(latest) as f:
    data = json.load(f)

for model, result in data["results"].items():
    batches = result.get("batches", [])
    rebalances = sum(1 for b in batches if b.get("rebalance", {}).get("did_rebalance", False))
    at_optimum = sum(1 for b in batches if b.get("rebalance", {}).get("at_optimum", False))
    rps = result.get("requests_per_second", 0)
    print(f"{model}: rps={rps:.2f}, rebalances={rebalances}, at_optimum={at_optimum}")
```

## Search strategy

1. **Verify rebalancing works.** Run with default parameters (tolerance=0.05, alpha=5) and REQUEST_NUM=1000 to confirm rebalancing happens. If 0 rebalances, stop and report — there may be a regression.

2. **Explore tolerance first** (most impactful). Try: 0.01, 0.02, 0.05, 0.1, 0.15. Keep alpha=5 fixed. Look for the sweet spot where rebalancing happens but isn't too aggressive.

3. **Then explore deep_alpha.** Try: 3, 5, 8, 10, 15. Use the best tolerance from step 2. Lower alpha = faster exploration but more rebuilds. Higher alpha = fewer rebuilds but slower to adapt.

4. **Finally try sibling_alpha.** Try: 1, 2, 3. This requires editing the default in `pipeline_optimizer.py` (line ~372). This controls how many stages to explore before stopping.

5. **Skip combinations that are clearly bad.** If a run has 0 rebalances, don't try higher tolerance or higher alpha — go lower. If a run is significantly worse than seed, don't explore nearby values.

6. **Record everything.** After each run, append a summary line to `data/sweep_results.txt`:
   ```
   alpha=5 tolerance=0.01 sibling=2 | avg_rps=X.XX | rebalances=N | at_optimum=N | file=<filename>
   ```

7. **Stop when** you've found parameters where:
   - Rebalancing actually happens (>0 rebalances)
   - RPS is equal to or better than the no-rebalance baseline
   - The optimiser reaches optimum within the run

## What a good result looks like

- The optimiser rebalances a few times early on, finds a good config, reaches optimum, and stops
- Throughput (RPS) is equal to or better than the seed config
- Batch times stabilise after the optimiser settles

## What a bad result looks like

- 0 rebalances (tolerance too high, alpha too high, or not enough requests)
- Hundreds of rebalances that never stop (tolerance too low, never reaches optimum)
- RPS significantly worse than baseline (rebalancing overhead outweighs gains)

## Important

- Do NOT modify any code except sibling_alpha default value when testing that parameter
- Do NOT change REBALANCE_INTERVAL, NPROC, OMP_THREADS, or N_MICROBATCHES
- Do NOT run with REQUEST_NUM > 2000 during the sweep — save long runs for final validation
- Always use `uv run` for Python commands
- Write results to `data/sweep_results.txt` so they persist
