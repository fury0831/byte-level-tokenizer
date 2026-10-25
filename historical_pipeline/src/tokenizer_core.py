"""Byte-level Trie matching and deterministic greedy/optimal segmentation.

Vocabulary learning is supplied by the attributed course baseline. Both new
encoders use exactly the same prepared Trie, token IDs, and strict decoder.
Only Python's standard library is required.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType


MODEL_FORMAT = "msdm5051-tokenizer-v1"
MAX_MODEL_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ByteRuntime:
    """Immutable byte vocabulary and Trie, detached from the saved model.

    Node 0 is the root. Each child mapping contains byte -> node index;
    a terminal entry is either a token ID (including 0) or None.
    """

    tokens: tuple[bytes, ...]
    children: tuple[Mapping[int, int], ...]
    terminals: tuple[int | None, ...]
    max_token_length: int


def build_runtime(model: dict) -> ByteRuntime:
    """Validate a saved model and build fresh state without modifying it.

    Learned metadata is retained in the saved model. Trie encoders need only
    the vocabulary and never depend on a training or query cache.
    The caller/harness checks the requested vocabulary budget and provenance.
    """
    if not isinstance(model, dict) or model.get("format") != MODEL_FORMAT:
        raise ValueError("Unsupported tokenizer model format")
    values = model.get("tokens_hex")
    if not isinstance(values, list) or len(values) < 256:
        raise ValueError("Model must contain all 256 base-byte tokens")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("Each tokens_hex entry must be a string")

    tokens = tuple(bytes.fromhex(value) for value in values)
    if tokens[:256] != tuple(bytes([i]) for i in range(256)):
        raise ValueError("IDs 0-255 must match their byte values")
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Tokens must be nonempty and distinct")

    # Use the same serialization convention as the official size check.
    payload = json.dumps(
        model, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_MODEL_BYTES:
        raise ValueError("Model exceeds the current 1 MiB contract")
    children: list[dict[int, int]] = [{}]
    terminals: list[int | None] = [None]
    for token_id, token in enumerate(tokens):
        node = 0
        for byte in token:
            child = children[node].get(byte)
            if child is None:
                child = len(children)
                children[node][byte] = child
                children.append({})
                terminals.append(None)
            node = child
        terminals[node] = token_id

    return ByteRuntime(
        tokens=tokens,
        children=tuple(MappingProxyType(branch) for branch in children),
        terminals=tuple(terminals),
        max_token_length=max(map(len, tokens)),
    )


def _utf8_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("encode expects a string")
    return text.encode("utf-8", errors="strict")


def iter_matches(
    data: bytes, start: int, runtime: ByteRuntime,
) -> Iterator[tuple[int, int]]:
    """Yield every (exclusive end, token ID) at start, in increasing end order.

    Traversal does not stop at the first terminal and does not treat a path
    without a terminal as a complete token. Matching is entirely byte-based.
    """
    if not 0 <= start <= len(data):
        raise ValueError("Match start is outside the input")
    node = 0
    stop = min(len(data), start + runtime.max_token_length)
    for position in range(start, stop):
        child = runtime.children[node].get(data[position])
        if child is None:
            break
        node = child
        token_id = runtime.terminals[node]
        if token_id is not None:
            yield position + 1, token_id


def encode_bytes(text: str, runtime: ByteRuntime) -> list[int]:
    """Keep the stage-2 byte fallback available for correctness diagnostics."""
    return list(_utf8_bytes(text))


def encode_greedy(text: str, runtime: ByteRuntime) -> list[int]:
    """Choose the longest complete token at each current byte position.

    This is a local choice, not a minimum-token guarantee. Every model has
    all byte tokens, so the initial single-byte choice always makes progress.
    """
    data = _utf8_bytes(text)
    result: list[int] = []
    position = 0
    while position < len(data):
        best_end, best_id = position + 1, data[position]
        for end, token_id in iter_matches(data, position, runtime):
            if end > best_end:
                best_end, best_id = end, token_id
        result.append(best_id)
        position = best_end
    return result


def encode_dp(text: str, runtime: ByteRuntime) -> list[int]:
    """Minimize token count for this record under the fixed vocabulary.

    cost[n] = 0; cost[i] = min(1 + cost[end]) over every token at i.
    On equal costs prefer a longer current token, then a smaller token ID.
    Store next positions and IDs to reconstruct iteratively in O(n) space.
    The Trie search takes O(n * L) time, where L is maximum token byte length.
    This proves optimal segmentation, not optimal vocabulary or LM quality.
    """
    data = _utf8_bytes(text)
    n = len(data)
    costs = [0] * (n + 1)
    next_positions = [0] * n
    chosen_ids = [0] * n

    for position in range(n - 1, -1, -1):
        best_end, best_id = position + 1, data[position]
        best_cost = 1 + costs[best_end]
        for end, token_id in iter_matches(data, position, runtime):
            candidate_cost = 1 + costs[end]
            if candidate_cost < best_cost or (
                candidate_cost == best_cost
                and (end > best_end or (end == best_end and token_id < best_id))
            ):
                best_cost, best_end, best_id = candidate_cost, end, token_id
        costs[position] = best_cost
        next_positions[position] = best_end
        chosen_ids[position] = best_id

    result: list[int] = []
    position = 0
    while position < n:
        result.append(chosen_ids[position])
        position = next_positions[position]
    return result


def decode_strict(ids: list[int], runtime: ByteRuntime) -> str:
    """Join token bytes before strict UTF-8 decoding, including partial tokens."""
    if not isinstance(ids, list):
        raise TypeError("decode expects a list of integer token IDs")
    if any(type(i) is not int or not 0 <= i < len(runtime.tokens) for i in ids):
        raise ValueError("Invalid token ID")
    data = b"".join(runtime.tokens[i] for i in ids)
    return data.decode("utf-8", errors="strict")
