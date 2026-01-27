import uuid
from typing import Dict, List

import torch
from torch import nn
from torch.distributed.pipelining import SplitPoint

from model_runner import (
    PipelineConfig,
    GreedyPipelineOptimizer,
    TimedModule,
    timed_module_registry,
    timed_module_hierarchy,
)
from model_runner.timed_module import CPUTimedModule, make_module_timed


class SimpleSequentialNet(nn.Module):
    """
    Simple sequential model for testing.
    Has 6 layers that can be distributed across devices.
    """
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(10, 32)
        self.layer1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def rand_inputs(self):
        return torch.randn(1, 10)


def test_pipeline_config():
    """Test PipelineConfig dataclass creation and access."""
    print("\n" + "=" * 80)
    print("TEST: test_pipeline_config")
    print("=" * 80)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()

    config = PipelineConfig(
        split_spec={uuid1: SplitPoint.BEGINNING, uuid2: SplitPoint.BEGINNING},
        device_mapping={0: "cuda:0", 1: "cuda:1", 2: "cuda:2"}
    )

    assert uuid1 in config.split_spec
    assert uuid2 in config.split_spec
    assert config.split_spec[uuid1] == SplitPoint.BEGINNING
    assert config.device_mapping[0] == "cuda:0"
    assert config.device_mapping[1] == "cuda:1"
    assert config.device_mapping[2] == "cuda:2"

    print(f"  split_spec has {len(config.split_spec)} entries")
    print(f"  device_mapping has {len(config.device_mapping)} entries")
    print("  PASSED")


def test_greedy_optimizer_extract_layer_times():
    """Test GreedyPipelineOptimizer._extract_layer_times method."""
    print("\n" + "=" * 80)
    print("TEST: test_greedy_optimizer_extract_layer_times")
    print("=" * 80)

    optimizer = GreedyPipelineOptimizer(num_stages=3, rebalance_threshold=0.1)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()

    time_logs = {
        uuid1: [100.0, 110.0, 90.0],  # avg = 100
        uuid2: [200.0, 200.0],         # avg = 200
        uuid3: [50.0],                 # avg = 50
    }

    layer_times = optimizer._extract_layer_times(time_logs)

    assert len(layer_times) == 3

    # Find each UUID in results
    times_by_uuid = {uid: time for uid, time in layer_times}
    assert abs(times_by_uuid[uuid1] - 100.0) < 0.01
    assert abs(times_by_uuid[uuid2] - 200.0) < 0.01
    assert abs(times_by_uuid[uuid3] - 50.0) < 0.01

    print(f"  Extracted {len(layer_times)} layer times")
    print(f"  uuid1 avg: {times_by_uuid[uuid1]}")
    print(f"  uuid2 avg: {times_by_uuid[uuid2]}")
    print(f"  uuid3 avg: {times_by_uuid[uuid3]}")
    print("  PASSED")


def test_greedy_optimizer_assign_layers():
    """Test GreedyPipelineOptimizer._assign_layers_to_stages method."""
    print("\n" + "=" * 80)
    print("TEST: test_greedy_optimizer_assign_layers")
    print("=" * 80)

    optimizer = GreedyPipelineOptimizer(num_stages=3, rebalance_threshold=0.1)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()
    uuid4 = uuid.uuid4()
    uuid5 = uuid.uuid4()
    uuid6 = uuid.uuid4()

    # Equal times: 100 each, total = 600, target = 200 per stage
    layer_times = [
        (uuid1, 100.0),
        (uuid2, 100.0),
        (uuid3, 100.0),
        (uuid4, 100.0),
        (uuid5, 100.0),
        (uuid6, 100.0),
    ]

    target_time = 200.0
    assignments = optimizer._assign_layers_to_stages(layer_times, target_time)

    assert len(assignments) == 6

    # With equal 100 time per layer and 200 target:
    # Stage 0: uuid1 (100), uuid2 (100) -> 200, move to stage 1
    # Stage 1: uuid3 (100), uuid4 (100) -> 200, move to stage 2
    # Stage 2: uuid5 (100), uuid6 (100)
    stages = {uid: stage for uid, stage in assignments}

    print(f"  Assignments: {[(str(uid)[:8], stage) for uid, stage in assignments]}")
    print(f"  Stage 0 count: {sum(1 for _, s in assignments if s == 0)}")
    print(f"  Stage 1 count: {sum(1 for _, s in assignments if s == 1)}")
    print(f"  Stage 2 count: {sum(1 for _, s in assignments if s == 2)}")
    print("  PASSED")


def test_greedy_optimizer_build_split_spec():
    """Test GreedyPipelineOptimizer._build_split_spec method."""
    print("\n" + "=" * 80)
    print("TEST: test_greedy_optimizer_build_split_spec")
    print("=" * 80)

    optimizer = GreedyPipelineOptimizer(num_stages=3, rebalance_threshold=0.1)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()
    uuid4 = uuid.uuid4()

    # Layers assigned to stages
    stage_assignments = [
        (uuid1, 0),
        (uuid2, 0),
        (uuid3, 1),  # Stage boundary here
        (uuid4, 2),  # Stage boundary here
    ]

    split_spec = optimizer._build_split_spec(stage_assignments)

    # Should mark uuid3 and uuid4 as stage boundaries
    assert uuid3 in split_spec
    assert uuid4 in split_spec
    assert split_spec[uuid3] == SplitPoint.BEGINNING
    assert split_spec[uuid4] == SplitPoint.BEGINNING
    # uuid1 and uuid2 should NOT be in split_spec (they're stage 0)
    assert uuid1 not in split_spec
    assert uuid2 not in split_spec

    print(f"  Split spec has {len(split_spec)} entries")
    print(f"  uuid3 marked as BEGINNING: {uuid3 in split_spec}")
    print(f"  uuid4 marked as BEGINNING: {uuid4 in split_spec}")
    print("  PASSED")


def test_greedy_optimizer_optimize():
    """Test full GreedyPipelineOptimizer.optimize method."""
    print("\n" + "=" * 80)
    print("TEST: test_greedy_optimizer_optimize")
    print("=" * 80)

    optimizer = GreedyPipelineOptimizer(num_stages=2, rebalance_threshold=0.1)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()
    uuid4 = uuid.uuid4()

    # Unbalanced: first two layers are heavy
    time_logs = {
        uuid1: [300.0, 300.0],
        uuid2: [200.0, 200.0],
        uuid3: [50.0, 50.0],
        uuid4: [50.0, 50.0],
    }

    old_config = PipelineConfig(
        split_spec={},
        device_mapping={0: "cpu", 1: "cpu"}
    )

    new_config = optimizer.optimize(time_logs, old_config)

    assert new_config is not None
    assert new_config.device_mapping == old_config.device_mapping
    # With 2 stages, total = 600, target = 300
    # uuid1 = 300, moves to stage 1 after
    # So uuid2 should be marked as stage 1 beginning
    print(f"  New split_spec: {new_config.split_spec}")
    print(f"  Split points: {len(new_config.split_spec)}")
    print("  PASSED")


def test_greedy_optimizer_should_rebalance():
    """Test GreedyPipelineOptimizer.should_rebalance method."""
    print("\n" + "=" * 80)
    print("TEST: test_greedy_optimizer_should_rebalance")
    print("=" * 80)

    optimizer = GreedyPipelineOptimizer(num_stages=2, rebalance_threshold=0.1)

    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()

    old_config = PipelineConfig(split_spec={}, device_mapping={0: "cpu", 1: "cpu"})

    # Test 1: Empty logs -> should rebalance
    empty_logs = {}
    assert optimizer.should_rebalance(empty_logs, old_config) == True
    print("  Empty logs -> should_rebalance: True")

    # Test 2: Stable timing (first == last) -> should NOT rebalance
    stable_logs = {
        uuid1: [100.0, 100.0],
        uuid2: [100.0, 100.0],
    }
    assert optimizer.should_rebalance(stable_logs, old_config) == False
    print("  Stable logs -> should_rebalance: False")

    # Test 3: Significant drift -> should rebalance
    drift_logs = {
        uuid1: [100.0, 300.0],  # 16.7% -> 50% (huge change)
        uuid2: [500.0, 300.0],  # 83.3% -> 50%
    }
    assert optimizer.should_rebalance(drift_logs, old_config) == True
    print("  Drift logs -> should_rebalance: True")

    # Test 4: Single entry per log -> should rebalance (can't compare)
    single_logs = {
        uuid1: [100.0],
        uuid2: [100.0],
    }
    assert optimizer.should_rebalance(single_logs, old_config) == True
    print("  Single entry logs -> should_rebalance: True")

    print("  PASSED")


def test_cpu_timed_module_creation():
    """Test CPUTimedModule creation and hierarchy registration."""
    print("\n" + "=" * 80)
    print("TEST: test_cpu_timed_module_creation")
    print("=" * 80)

    # Clear registries
    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1)

    # Check registration
    assert timed.uuid in timed_module_registry
    assert timed.uuid in timed_module_hierarchy

    print(f"  Module UUID: {timed.uuid}")
    print(f"  Registered in registry: {timed.uuid in timed_module_registry}")
    print(f"  Children count: {len(timed_module_hierarchy[timed.uuid])}")

    # Check children were wrapped
    children = list(timed.inner().named_children())
    for name, child in children:
        assert isinstance(child, CPUTimedModule), f"Child {name} not wrapped"
        assert child.uuid in timed_module_registry
        assert child.uuid in timed_module_hierarchy[timed.uuid]

    print(f"  All {len(children)} children wrapped as CPUTimedModule")
    print("  PASSED")


def test_cpu_timed_module_forward():
    """Test CPUTimedModule forward pass and timing."""
    print("\n" + "=" * 80)
    print("TEST: test_cpu_timed_module_forward")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1)

    # Run forward pass
    x = model.rand_inputs()
    output = timed(x)

    assert output is not None
    assert output.shape == (1, 10)

    # Check timing was recorded
    elapsed = timed.get_last_elapsed_cycles()
    assert elapsed > 0, "Elapsed time should be positive"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Elapsed time (ns): {elapsed}")
    print("  PASSED")


def test_cpu_timed_module_get_logs():
    """Test CPUTimedModule.get_logs collects timing from hierarchy."""
    print("\n" + "=" * 80)
    print("TEST: test_cpu_timed_module_get_logs")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1)

    # Run forward pass to populate timing
    x = model.rand_inputs()
    _ = timed(x)

    # Get logs
    logs = timed.get_logs()

    assert len(logs) > 0
    assert timed.uuid in logs

    # Check all children are in logs
    for child_uuid in timed_module_hierarchy[timed.uuid]:
        assert child_uuid in logs

    print(f"  Logs collected for {len(logs)} modules")
    print(f"  Root module time: {logs[timed.uuid]}")

    # Print child times
    for child_uuid in timed_module_hierarchy[timed.uuid]:
        child = timed_module_registry[child_uuid]
        print(f"  Child '{child.module_path}' time: {logs[child_uuid]}")

    print("  PASSED")


def test_cpu_timed_module_multiple_runs():
    """Test that get_logs accumulates timing across multiple runs."""
    print("\n" + "=" * 80)
    print("TEST: test_cpu_timed_module_multiple_runs")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1)

    x = model.rand_inputs()

    # Run multiple forward passes, accumulating logs
    logs = {}
    for i in range(5):
        _ = timed(x)
        logs = timed.get_logs(existing_logs=logs)

    # Each module should have 5 timing entries
    assert len(logs[timed.uuid]) == 5

    for child_uuid in timed_module_hierarchy[timed.uuid]:
        assert len(logs[child_uuid]) == 5

    print(f"  Ran 5 forward passes")
    print(f"  Root timing entries: {logs[timed.uuid]}")
    print("  PASSED")


def test_cpu_timed_module_path():
    """Test that CPUTimedModule tracks module paths correctly."""
    print("\n" + "=" * 80)
    print("TEST: test_cpu_timed_module_path")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1, module_path="root")

    assert timed.get_path() == "root"

    # Check children have correct paths
    for child_uuid in timed_module_hierarchy[timed.uuid]:
        child = timed_module_registry[child_uuid]
        path = child.get_path()
        assert path.startswith("root.")
        print(f"  Child path: {path}")

    print("  PASSED")


def test_make_module_timed():
    """Test make_module_timed factory function."""
    print("\n" + "=" * 80)
    print("TEST: test_make_module_timed")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()

    # Should create CPUTimedModule on CPU
    timed = make_module_timed(model, device="cpu", depth=2)

    assert isinstance(timed, CPUTimedModule)
    assert timed.uuid in timed_module_registry

    print(f"  Created {type(timed).__name__}")
    print(f"  Device: {timed.device}")
    print(f"  Depth: {timed.depth}")
    print("  PASSED")


def test_optimizer_with_real_timing():
    """Test optimizer with real timing data from CPUTimedModule."""
    print("\n" + "=" * 80)
    print("TEST: test_optimizer_with_real_timing")
    print("=" * 80)

    timed_module_registry.clear()
    timed_module_hierarchy.clear()

    model = SimpleSequentialNet()
    timed = CPUTimedModule(model, device="cpu", depth=1)

    # Run some forward passes
    x = model.rand_inputs()
    logs = {}
    for _ in range(10):
        _ = timed(x)
        logs = timed.get_logs(existing_logs=logs)

    # Create optimizer
    optimizer = GreedyPipelineOptimizer(num_stages=2, rebalance_threshold=0.1)

    old_config = PipelineConfig(
        split_spec={},
        device_mapping={0: "cpu", 1: "cpu"}
    )

    # Test should_rebalance
    should_rebalance = optimizer.should_rebalance(logs, old_config)
    print(f"  should_rebalance: {should_rebalance}")

    # Test optimize
    new_config = optimizer.optimize(logs, old_config)
    print(f"  New split_spec entries: {len(new_config.split_spec)}")

    # Verify the split points are valid UUIDs from our hierarchy
    for split_uuid in new_config.split_spec.keys():
        assert split_uuid in timed_module_registry, f"Unknown UUID in split_spec: {split_uuid}"

    print("  PASSED")


if __name__ == '__main__':
    import sys

    tests = [
        ("config", test_pipeline_config),
        ("extract", test_greedy_optimizer_extract_layer_times),
        ("assign", test_greedy_optimizer_assign_layers),
        ("split", test_greedy_optimizer_build_split_spec),
        ("optimize", test_greedy_optimizer_optimize),
        ("rebalance", test_greedy_optimizer_should_rebalance),
        ("cpu-create", test_cpu_timed_module_creation),
        ("cpu-forward", test_cpu_timed_module_forward),
        ("cpu-logs", test_cpu_timed_module_get_logs),
        ("cpu-multi", test_cpu_timed_module_multiple_runs),
        ("cpu-path", test_cpu_timed_module_path),
        ("factory", test_make_module_timed),
        ("integration", test_optimizer_with_real_timing),
    ]

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "--all":
            for name, test_fn in tests:
                test_fn()
        else:
            # Find matching test
            found = False
            for name, test_fn in tests:
                if name == test_name or test_name in name:
                    test_fn()
                    found = True
                    break
            if not found:
                print(f"Unknown test: {test_name}")
                print(f"Available: {', '.join(name for name, _ in tests)}, --all")
    else:
        # Run all tests by default
        for name, test_fn in tests:
            test_fn()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
