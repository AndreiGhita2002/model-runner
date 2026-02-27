.PHONY: install test-flask test-pipeline eval quick-eval simple-baseline gpipe-baseline benchmark clean \
       eval-sequential eval-tensor-parallel eval-gpipe eval-all-baselines

# Number of distributed ranks (processes). Override with: make eval NPROC=5
NPROC ?= 8
# CPU threads per rank — divides total cores evenly across ranks to avoid oversubscription
#OMP_THREADS = $(shell echo $$(( $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) / $(NPROC) )))
OMP_THREADS = 4
# Common torchrun invocation with OMP_NUM_THREADS set to suppress the default warning
TORCHRUN = OMP_NUM_THREADS=$(OMP_THREADS) uv run --no-sync torchrun --nproc_per_node=$(NPROC) -m

install:
	uv pip install -e .

eval: install
	$(TORCHRUN) tests.evaluation

quick-eval: install
	$(TORCHRUN) tests.quick_evaluation

simple-baseline: install
	OMP_NUM_THREADS=$(OMP_THREADS) uv run --no-sync python -m tests.baseline simple

gpipe-baseline: install
	$(TORCHRUN) tests.baseline gpipe

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

# --- Evaluation server baselines (32-core machine) ---
EVAL_CORES ?= 32
EVAL_NPROC ?= $(EVAL_CORES)

eval-sequential: install
	OMP_NUM_THREADS=1 uv run --no-sync python -m tests.baseline simple -o sequential_baseline.json

eval-tensor-parallel: install
	OMP_NUM_THREADS=$(EVAL_CORES) uv run --no-sync python -m tests.baseline simple -o tensor_parallel_baseline.json

eval-gpipe: install
	OMP_NUM_THREADS=1 uv run --no-sync torchrun --nproc_per_node=$(EVAL_NPROC) -m tests.baseline gpipe -o gpipe_baseline.json

eval-all-baselines: eval-sequential eval-tensor-parallel eval-gpipe

graph-baselines: install
	uv run python data/baseline_graphs.py

clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
