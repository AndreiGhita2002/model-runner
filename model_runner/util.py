import struct
import uuid

import torch


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
