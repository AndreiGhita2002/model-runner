# Evaluation + Baseline Data

### Scripts:
- `graphs.py` — generates graphs from baseline and evaluation data.
  - `-i`, `--input-dir` — directory containing JSON data files (default: `data/sample/`)
  - `-o`, `--output` — output image path (e.g. `graphs.png`). Displays interactively if not set.
  - Expected files in the input directory: `sequential_baseline.json`, `tensor_parallel_baseline.json`, `gpipe_baseline.json`, `evaluation_output.json`. Missing files are skipped with a warning.
  - Produces two graphs: a **throughput bar chart** (requests/sec per model) and a **batch times line plot** (per-batch elapsed time, with rebalance markers for the adaptive dataset). With `-o`, the batch times plot is saved as `<stem>_batch_times.png`.

### Data Directories:
- /sample/ -- includes a small amount of data for testing purposes. 
Note: this data was run on an unusually small batch size and small number of cores,
so it should not be taken seriously.

