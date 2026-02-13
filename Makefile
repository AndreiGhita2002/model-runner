.PHONY: install test-flask test-pipeline eval baseline clean

TORCHRUN = uv run --no-sync torchrun --nproc_per_node=2 -m

install:
	uv pip install -e .

eval: install
	$(TORCHRUN) tests.evaluation

baseline: install
	uv run --no-sync torchrun --nproc_per_node=1 -m tests.baseline

test-flask: install
	uv run --no-sync python -m tests.flask_test

test-pipeline: install
	$(TORCHRUN) tests.pipeline_test

clean:
	rm -rf build/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
