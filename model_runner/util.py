import struct
import uuid
from typing import Generator

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint


def uuids_to_tensor(uuids: list[uuid.UUID], pad_to: int) -> torch.Tensor:
    """Encode UUIDs as a flat int32 tensor for P2P transfer via torch.distributed.

    Each UUID is serialized as 4 signed int32 values (16 bytes). Remaining slots
    up to ``pad_to`` are filled with zeros (nil UUID), which acts as the padding sentinel.

    Args:
        uuids: UUIDs to encode. Length must be <= pad_to.
        pad_to: Total number of UUID slots in the output tensor.

    Returns:
        Flat int32 tensor of shape ``(pad_to * 4,)``.
    """
    ints = []
    for u in uuids:
        ints.extend(struct.unpack('>4i', u.bytes))
    # Pad remaining slots with zeros (nil UUID = sentinel)
    ints.extend([0] * (pad_to - len(uuids)) * 4)
    return torch.tensor(ints, dtype=torch.int)


def _unwrap(module: nn.Module) -> nn.Module:
    """Unwrap TimedModule wrappers to get the raw module."""
    while hasattr(module, '_inner'):
        module = module._inner
    return module


def _children_at_depth(
    module: nn.Module, depth: int, prefix: str = ""
) -> Generator[tuple[str, nn.Module], None, None]:
    """Yield (dotted_path, submodule) pairs at a given depth.

    Transparently unwraps TimedModule wrappers when traversing.
    """
    module = _unwrap(module)
    if depth <= 0:
        if prefix:
            yield prefix, module
        return
    children = list(module.named_children())
    if not children:
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


def _dp_min_max_partition(costs: list[int], K: int) -> list[int]:
    """Find K-way partition of consecutive items minimising the maximum stage cost.

    Returns a list of split indices (where each new stage begins), excluding 0.
    Uses O(N²K) dynamic programming with backtracking.
    """
    N = len(costs)
    if K >= N:
        return list(range(1, N))

    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + costs[i]

    def stage_cost(i: int, j: int) -> int:
        return prefix[j] - prefix[i]

    INF = float("inf")
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    choice = [[0] * (N + 1) for _ in range(K + 1)]

    for i in range(1, N + 1):
        dp[1][i] = prefix[i]

    for k in range(2, K + 1):
        for i in range(k, N + 1):
            for j in range(k - 1, i):
                val = max(dp[k - 1][j], stage_cost(j, i))
                if val < dp[k][i]:
                    dp[k][i] = val
                    choice[k][i] = j

    splits = []
    i = N
    for k in range(K, 1, -1):
        j = choice[k][i]
        splits.append(j)
        i = j

    splits.sort()
    return splits


def gpipe_split_spec(
    model: nn.Module, num_stages: int, depth: int = 2
) -> dict[str, SplitPoint]:
    """Compute a cost-balanced split_spec for GPipe-style pipeline parallelism.

    Partitions consecutive submodules (at the given depth) into *num_stages*
    cells, minimising cost variance using parameter count as the cost proxy.
    """
    children = list(_children_at_depth(model, depth))
    while len(children) < num_stages:
        depth += 1
        deeper = list(_children_at_depth(model, depth))
        if len(deeper) == len(children):
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


def tensor_to_uuids(t: torch.Tensor) -> list[uuid.UUID | None]:
    """Decode a flat int32 tensor back to UUIDs. Inverse of ``uuids_to_tensor``.

    Args:
        t: Flat int32 tensor of shape ``(n * 4,)`` produced by ``uuids_to_tensor``.

    Returns:
        List of ``n`` entries. Each entry is a ``uuid.UUID``, or ``None`` for
        nil-UUID slots (the padding sentinel).
    """
    values = t.tolist()
    result = []
    for i in range(0, len(values), 4):
        chunk = values[i:i + 4]
        if all(v == 0 for v in chunk):
            result.append(None)
        else:
            raw = struct.pack('>4i', *chunk)
            result.append(uuid.UUID(bytes=raw))
    return result
