.PHONY: install test-flask test-pipeline eval quick-eval simple-baseline gpipe-baseline benchmark clean \
       eval-sequential eval-tensor-parallel eval-gpipe eval-gpipe-sweep eval-all-baselines

MAX_CORES ?= 32  # should be 32 on fisherman
MAX_NPROC ?= $(MAX_CORES)
REQUEST_NUM ?= 5000  # should be 100+
BASELINE_REQUEST_NUM ?= 10  # baselines don't change over time, fewer runs suffice
BATCH_COUNT ?= 1  # samples per request (1 image each)
N_MICROBATCHES ?= 32  # requests grouped per forward pass
OPTIMIZER ?= shisha
REBALANCE_INTERVAL ?=
ASSIGNMENT_CHOICE ?=
BALANCE_STRATEGY ?=
ALPHA ?=
COMMON_ARGS ?= -n $(REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES) --optimizer $(OPTIMIZER) \
	$(if $(REBALANCE_INTERVAL),--rebalance-interval $(REBALANCE_INTERVAL)) \
	$(if $(ASSIGNMENT_CHOICE),--assignment-choice $(ASSIGNMENT_CHOICE)) \
	$(if $(BALANCE_STRATEGY),--balance-strategy $(BALANCE_STRATEGY)) \
	$(if $(ALPHA),--alpha $(ALPHA))
BASELINE_ARGS ?= -n $(BASELINE_REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES)
BASELINES_DIR ?= ./data/baselines
RUNS_DIR ?= ./data/runs
GPIPE_OUTPUT ?= gpipe.json

# Number of distributed ranks (processes). Override with: make eval NPROC=5
NPROC ?= 4
# CPU threads per rank — divides total cores evenly across ranks to avoid oversubscription
#OMP_THREADS = $(shell echo $$(( $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) / $(NPROC) )))
OMP_THREADS = 8
# Common torchrun invocation with OMP_NUM_THREADS set to suppress the default warning
TORCHRUN = OMP_NUM_THREADS=$(OMP_THREADS) uv run --no-sync torchrun --nproc_per_node=$(NPROC) -m

install:
	uv pip install -e .

eval: install
	$(TORCHRUN) tests.evaluation $(COMMON_ARGS) -o $(RUNS_DIR)

quick-eval: install
	$(TORCHRUN) tests.quick_evaluation

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

eval-sequential: install
	OMP_NUM_THREADS=1 uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(BASELINES_DIR)/sequential.json

eval-tensor-parallel: install
	OMP_NUM_THREADS=$(MAX_CORES) uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(BASELINES_DIR)/tensor_parallel.json

eval-gpipe: install
	$(TORCHRUN) tests.baseline gpipe $(BASELINE_ARGS) -o $(BASELINES_DIR)/$(GPIPE_OUTPUT)

eval-gpipe-sweep: install
	bash tests/gpipe_sweep.sh

eval-all-baselines: eval-sequential eval-tensor-parallel eval-gpipe

eval-all: install eval-all-baselines eval

graphs: install
	uv run python data/graphs.py -b $(BASELINES_DIR) -r $(RUNS_DIR)


clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
