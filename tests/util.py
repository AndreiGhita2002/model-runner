import torch
import torch.nn as nn
from typing import Callable, Generator
from torch.distributed.pipelining import SplitPoint


def _children_at_depth(
    module: nn.Module, depth: int, prefix: str = ""
) -> Generator[tuple[str, nn.Module], None, None]:
    """Yield (dotted_path, submodule) pairs at a given depth.

    Recurses into named_children(). Modules that have no children at a
    shallower depth are yielded as leaves immediately.
    """
    if depth <= 0:
        if prefix:
            yield prefix, module
        return
    children = list(module.named_children())
    if not children:
        # Leaf reached before target depth — yield it anyway
        if prefix:
            yield prefix, module
        return
    for name, child in children:
        path = f"{prefix}.{name}" if prefix else name
        yield from _children_at_depth(child, depth - 1, path)


def _uniform_split(names: list[str], num_stages: int) -> dict[str, SplitPoint]:
    """Fallback: evenly spaced splits when cost information is unavailable."""
    n = len(names)
    splits_needed = min(num_stages - 1, n - 1)
    spec: dict[str, SplitPoint] = {}
    for k in range(1, splits_needed + 1):
        idx = (k * n) // num_stages
        if idx < n:
            spec[names[idx]] = SplitPoint.BEGINNING
    return spec


def gpipe_split_spec(
    model: nn.Module, num_stages: int, depth: int = 2
) -> dict[str, SplitPoint]:
    """Compute a cost-balanced split_spec for GPipe-style pipeline parallelism.

    Partitions consecutive submodules (at the given depth) into *num_stages*
    cells, minimising cost variance using parameter count as the cost proxy
    (GPipe paper, Section 2.2).

    Returns a dict mapping submodule names to ``SplitPoint.BEGINNING``,
    suitable for passing to ``torch.distributed.pipelining.pipeline()``.

    Differences from the GPipe paper:

    - **Cost proxy**: GPipe uses estimated FLOPs; this uses parameter count.
      These diverge for convolutions where FLOPs scale with spatial
      dimensions — early layers with large inputs are under-costed here.
    - **Granularity**: GPipe partitions individual layers; this partitions at
      depth-2 submodules (~8-11 blocks for torchvision models). This is
      coarser but necessary because ``torch.distributed.pipelining``
      requires split_spec keys that match named submodule paths.
    - **Algorithm**: This uses the optimal dynamic programming partition
      (minimise maximum stage cost) in O(N²K), matching the GPipe paper.
    """
    children = list(_children_at_depth(model, depth))
    # If there aren't enough children at the requested depth, go deeper
    while len(children) < num_stages:
        depth += 1
        deeper = list(_children_at_depth(model, depth))
        if len(deeper) == len(children):
            # Can't get more children by going deeper — model is too small
            break
        children = deeper
    if not children:
        return {}
    if len(children) < num_stages:
        raise ValueError(
            f"Model only has {len(children)} submodules but {num_stages} stages "
            f"were requested. Use fewer ranks or a larger model."
        )

    names = [name for name, _ in children]
    costs = [sum(p.numel() for p in mod.parameters()) for _, mod in children]
    total_cost = sum(costs)

    if total_cost == 0:
        return _uniform_split(names, num_stages)

    K = min(num_stages, len(children))
    split_indices = _dp_min_max_partition(costs, K)

    spec: dict[str, SplitPoint] = {}
    for idx in split_indices:
        spec[names[idx]] = SplitPoint.BEGINNING
    return spec


def _dp_min_max_partition(costs: list[int], K: int) -> list[int]:
    """Find K-way partition of consecutive items minimising the maximum stage cost.

    Returns a list of split indices (where each new stage begins), excluding 0.
    Uses O(N²K) dynamic programming with backtracking.
    """
    N = len(costs)
    if K >= N:
        return list(range(1, N))

    # prefix[i] = sum(costs[0..i-1])
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + costs[i]

    def stage_cost(i: int, j: int) -> int:
        """Cost of a stage spanning modules [i, j)."""
        return prefix[j] - prefix[i]

    # dp[k][i] = min possible max-stage-cost partitioning costs[0..i-1] into k stages
    INF = float("inf")
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    # choice[k][i] = the split point j that achieved dp[k][i]
    choice = [[0] * (N + 1) for _ in range(K + 1)]

    # Base case: 1 stage covering costs[0..i-1]
    for i in range(1, N + 1):
        dp[1][i] = prefix[i]

    # Fill DP: k stages over i modules
    for k in range(2, K + 1):
        for i in range(k, N + 1):
            for j in range(k - 1, i):
                val = max(dp[k - 1][j], stage_cost(j, i))
                if val < dp[k][i]:
                    dp[k][i] = val
                    choice[k][i] = j

    # Backtrack to recover split points
    splits = []
    i = N
    for k in range(K, 1, -1):
        j = choice[k][i]
        splits.append(j)
        i = j

    splits.sort()
    return splits


def generate_batch(rand_inputs: Callable[[], torch.Tensor], batch_size: int, seed: int) -> torch.Tensor:
    """Generate a batch of inputs from a single seed.

    Sets the seed once, then calls rand_inputs() batch_size times and
    concatenates the results. Each sample is different because the RNG
    state advances between calls, but the full batch is deterministically
    reproducible from the seed.
    """
    torch.manual_seed(seed)
    return torch.cat([rand_inputs() for _ in range(batch_size)], dim=0)
