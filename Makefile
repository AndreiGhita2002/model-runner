.PHONY: install test-flask test-pipeline eval quick-eval simple-baseline pipeline-baseline benchmark clean

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
	$(TORCHRUN) tests.baseline pipeline

test-flask: install
	uv run --no-sync python -m tests.flask_test

# Run full hardware benchmark (finds optimal OMP_NUM_THREADS and NPROC)
benchmark: install
	uv run --no-sync python -m tests.benchmark

clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
