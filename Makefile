.PHONY: install test test-all test-pipeline clean

TORCHRUN = uv run --no-sync torchrun --nproc_per_node=2 -m

install:
	uv pip install -e .

eval:
	$(TORCHRUN) tests.evaluation

baseline:
	uv run --no-sync torchrun --nproc_per_node=1 -m tests.baseline

test-pipeline:
	$(TORCHRUN) tests.pipeline_test

clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
