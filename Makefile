.PHONY: install test-flask test-pipeline eval quick-eval simple-baseline gpipe-baseline benchmark clean \
       eval-sequential eval-tensor-parallel eval-gpipe eval-all-baselines

# Number of distributed ranks (processes). Override with: make eval NPROC=5
NPROC ?= 4
# CPU threads per rank — divides total cores evenly across ranks to avoid oversubscription
#OMP_THREADS = $(shell echo $$(( $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) / $(NPROC) )))
OMP_THREADS = 2
# Common torchrun invocation with OMP_NUM_THREADS set to suppress the default warning
TORCHRUN = OMP_NUM_THREADS=$(OMP_THREADS) uv run --no-sync torchrun --nproc_per_node=$(NPROC) -m

#TODO: update this to have a all the cores

# -------- Evaluation Commands ---------
EVAL_CORES ?= 8  # should be 32 on fisherman
EVAL_NPROC ?= $(EVAL_CORES)
REQUEST_NUM ?= 5  # should be 100+
BATCH_COUNT ?= 8  # should be 32
COMMON_ARGS ?= -n $(REQUEST_NUM) -b $(BATCH_COUNT)

install:
	uv pip install -e .

eval: install
	$(TORCHRUN) tests.evaluation $(COMMON_ARGS) -o ./data/sample/evaluation_output.json

quick-eval: install
	$(TORCHRUN) tests.quick_evaluation

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

eval-sequential: install
	OMP_NUM_THREADS=1 uv run --no-sync python -m tests.baseline simple $(COMMON_ARGS) -o ./data/sample/sequential_baseline.json

eval-tensor-parallel: install
	OMP_NUM_THREADS=$(EVAL_CORES) uv run --no-sync python -m tests.baseline simple $(COMMON_ARGS) -o ./data/sample/tensor_parallel_baseline.json

eval-gpipe: install
	OMP_NUM_THREADS=1 uv run --no-sync torchrun --nproc_per_node=$(EVAL_NPROC) -m tests.baseline gpipe $(COMMON_ARGS) -o ./data/sample/gpipe_baseline.json

eval-all-baselines: eval-sequential eval-tensor-parallel eval-gpipe

graphs: install
	uv run python data/graphs.py


clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
