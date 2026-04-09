.PHONY: install test-flask test-pipeline eval eval-extended-set quick-eval simple-baseline gpipe-baseline benchmark clean \
       eval-sequential eval-tensor-parallel eval-gpipe eval-gpipe-sweep eval-all-baselines sweep csv \
       interf-eval interf-eval-clean experiment

MAX_CORES ?= 32  # should be 32 on fisherman
MAX_NPROC ?= $(MAX_CORES)
REQUEST_NUM ?= 5000  # should be 100+
BASELINE_REQUEST_NUM ?= 10  # baselines don't change over time, fewer runs suffice
BATCH_COUNT ?= 1  # samples per request (1 image each)
N_MICROBATCHES ?= 32  # requests grouped per forward pass
OPTIMIZER ?= reactive
REBALANCE_INTERVAL ?=
ASSIGNMENT_CHOICE ?=
DEEP_ALPHA ?=
SIBLING_ALPHA ?=
TOLERANCE ?=
OPTIMUM_TOLERANCE ?=
OPTIMUM_ESCAPE ?=
MODEL_SET ?=
COMMON_ARGS ?= -n $(REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES) --optimizer $(OPTIMIZER) \
	$(if $(MODEL_SET),--model-set $(MODEL_SET)) \
	$(if $(REBALANCE_INTERVAL),--rebalance-interval $(REBALANCE_INTERVAL)) \
	$(if $(ASSIGNMENT_CHOICE),--assignment-choice $(ASSIGNMENT_CHOICE)) \
	$(if $(DEEP_ALPHA),--alpha $(DEEP_ALPHA)) \
	$(if $(SIBLING_ALPHA),--sibling-alpha $(SIBLING_ALPHA)) \
	$(if $(TOLERANCE),--tolerance $(TOLERANCE)) \
	$(if $(OPTIMUM_TOLERANCE),--optimum-tolerance $(OPTIMUM_TOLERANCE)) \
	$(if $(OPTIMUM_ESCAPE),--optimum-escape $(OPTIMUM_ESCAPE))
BASELINE_ARGS ?= -n $(BASELINE_REQUEST_NUM) -b $(BATCH_COUNT) -m $(N_MICROBATCHES)
BASELINES_DIR ?= ./data/baselines
RUNS_DIR ?= ./data/runs
GPIPE_OUTPUT ?= gpipe.json
REBALANCE_TIME ?=
OPTIMUM ?= 1
SHOW_REBALANCE ?= 1
GRAPH_ARGS ?= $(if $(REBALANCE_TIME),--rebalance-time) $(if $(OPTIMUM),--optimum) $(if $(SHOW_REBALANCE), --rebalance)

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

# Full experiment suite (runs A-E)
REPETITIONS ?= 1
SCHEDULE ?= experiment
experiment: install
	uv run python -m tests.experiment --nproc $(NPROC) --repetitions $(REPETITIONS) \
		$(if $(MODEL_SET),--model-set $(MODEL_SET)) \
		$(if $(STEP_DURATION),--duration $(STEP_DURATION)) \
		$(if $(SKIP),--skip $(SKIP)) \
		$(if $(KEEP_LOGS),--keep-logs) \
		--schedule $(SCHEDULE)

eval-extended: install
	$(MAKE) eval MODEL_SET=extended

quick-eval: install
	$(TORCHRUN) tests.quick_evaluation

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

baseline-sequential: install
	OMP_NUM_THREADS=1 uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(BASELINES_DIR)/sequential.json

baseline-tensor-parallel: install
	OMP_NUM_THREADS=$(MAX_CORES) uv run --no-sync python -m tests.baseline simple $(BASELINE_ARGS) -o $(BASELINES_DIR)/tensor_parallel.json

baseline-gpipe: install
	$(TORCHRUN) tests.baseline gpipe $(BASELINE_ARGS) -o $(BASELINES_DIR)/$(GPIPE_OUTPUT)

baseline-gpipe-sweep: install
	bash tests/gpipe_sweep.sh

sweep: install
	bash tests/sweep.sh $(REQUEST_NUM)

baselines: baseline-sequential baseline-tensor-parallel baseline-gpipe

graphs: install
	uv run python data/graphs.py $(GRAPH_ARGS) -b $(BASELINES_DIR) -r $(RUNS_DIR)

csv:
	uv run python data/runs_to_csv.py -d $(RUNS_DIR) -o data/runs_summary.csv

interf-csv:
	uv run python data/interference_to_csv.py -o data/interference_summary.csv

# Interference experiments
interf-eval: install
	uv run python -m tests.interference.interfere_eval --nproc $(NPROC)

interf-test: install
	uv run python -m tests.interference.interfere_eval --no-interference --duration 10 --nproc $(NPROC)

bench-test: install
	uv run python -m tests.interference.test_benchmarks

clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
