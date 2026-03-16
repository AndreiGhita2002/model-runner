# Evaluation + Baseline Data

### Scripts:
- `graphs.py` — generates graphs from baseline and evaluation data.
  - `-i`, `--input-dir` — directory containing JSON data files (default: `data/sample/`)
  - `-o`, `--output` — output image path (e.g. `graphs.png`). Displays interactively if not set.
  - Expected files in the input directory: `sequential_baseline.json`, `tensor_parallel_baseline.json`, `gpipe_baseline.json`, `evaluation_output.json`. Missing files are skipped with a warning.
  - Produces two graphs: a **throughput bar chart** (requests/sec per model) and a **batch times line plot** (per-batch elapsed time, with rebalance markers for the adaptive dataset). With `-o`, the batch times plot is saved as `<stem>_batch_times.png`.

### Data Directories:

- **sample/** — Small amount of data for testing graph generation. Run on an unusually small batch size and small number of cores; should not be taken seriously.

- **baselines/** — Baseline measurements on server `fisherman` (32-core CPU, no GPU). 10 requests per model, batch_size=1, n_microbatches=32. All use 4 models: ConvNeXt Small, ConvNeXt Base, EfficientNet B6, RegNet X 16GF.
  - `sequential.json` — Single-threaded sequential inference (1 OMP thread).
  - `tensor_parallel.json` — Single-process inference with 32 OMP threads (intra-op parallelism).
  - `gpipe_NxM.json` — Static GPipe pipeline with N ranks and M OMP threads per rank (N*M=32). Explores the tradeoff between pipeline parallelism and intra-op parallelism. `gpipe_32x1` is missing RegNet due to FX tracing failures at high rank counts.

- **runs/** – Adaptive pipeline evaluation runs on server `fisherman`. All use 4 ranks, 8 OMP threads, batch_size=1, n_microbatches=32, 5000 requests per model, seed=37.
  - `run1.json` – Shisha with rebalance interval=10
  - `run2.json` – Shisha same as run1
  - `run3.json` – Shisha without the rebalance interval
  - `run4.json` – Shisha same as run 3
  - `run5.json` – Greedy, no rebalance interval
  - `run6.json` – Greedy with rebalance interval=5
  - `run7.json` – Shisha with rank_w assignment, nearest_lightest_fep balance strategy and return to optimum
  - `run8.json` – run7 with multiple optimum attempts
  - `run9.json` – added rebalance interval back with multiple optimum attempts
  - `run10.json` – caching module graph for rebalance
  - `run11.json` – same as run 10
  - `run12.json` – multiple online tuning steps per rebalance 
  - `run13.json` – same as run 12 but with `_slowest_stage_offset >= num_stages / 2`
  - `run14.json` – one tuning step per rebalance 
  - `run15.json` – num_stages - 1 (only look at second-slowest stage then stop exploration)


- What have we learnt from this?
  - multiple optimum attempts are GOOD, but there should be a limit -- we dont need a sort for the slowest
  - rebalance interval is GOOD
  - more online tuning steps per rebalance is BAD, despite what my intuition says <- make sure you move the second slowest stage
  - run14 is the best for 3/4, and run13 is best for 1/4 (regnet)
   - so online step is usually BAD, but GOOD for regnet  

- What is left to explore?
  - other shisha strategies from the original paper

