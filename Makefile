.PHONY: install test-flask test-pipeline eval quick-eval simple-baseline gpipe-baseline benchmark clean \
       eval-sequential eval-tensor-parallel eval-gpipe eval-all-baselines

MAX_CORES ?= 32  # should be 32 on fisherman
MAX_NPROC ?= $(MAX_CORES)
REQUEST_NUM ?= 5000  # should be 100+
BASELINE_REQUEST_NUM ?= 10  # baselines don't change over time, fewer runs suffice
BATCH_COUNT ?= 1  # samples per request (1 image each)
N_MICROBATCHES ?= 32  # requests grouped per forward pass
COMMON_ARGS ?= -n $(REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES)
BASELINE_ARGS ?= -n $(BASELINE_REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES)
DATA_OUTPUT_DIR ?= ./data/run1

# Number of distributed ranks (processes). Override with: make eval NPROC=5
NPROC ?= 4
# CPU threads per rank — divides total cores evenly across ranks to avoid oversubscription
#OMP_THREADS = $(shell echo $$(( $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) / $(NPROC) )))
OMP_THREADS = 8
# Common torchrun invocation with OMP_NUM_THREADS set to suppress the default warning
TORCHRUN = OMP_NUM_THREADS=$(OMP_THREADS) uv run --no-sync torchrun --nproc_per_node=$(NPROC) -m

install:
	uv pip install -e .

#todo: run this five more times

eval: install
	$(TORCHRUN) tests.evaluation $(COMMON_ARGS) -o $(DATA_OUTPUT_DIR)/evaluation_output.json

quick-eval: install
	$(TORCHRUN) tests.quick_evaluation

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

eval-sequential: install
	OMP_NUM_THREADS=1 uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(DATA_OUTPUT_DIR)/sequential_baseline.json

#TODO: get in between baselines too

eval-tensor-parallel: install
	OMP_NUM_THREADS=$(MAX_CORES) uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(DATA_OUTPUT_DIR)/tensor_parallel_baseline.json

eval-gpipe: install
	$(TORCHRUN) tests.baseline gpipe $(BASELINE_ARGS) -o $(DATA_OUTPUT_DIR)/gpipe_baseline.json

eval-all-baselines: eval-sequential eval-tensor-parallel eval-gpipe

eval-all: install eval-all-baselines eval

graphs: install
	uv run python data/graphs.py -i $(DATA_OUTPUT_DIR)


clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
