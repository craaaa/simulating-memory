from __future__ import annotations
from typing import Dict

MAX_KEYS = 4
NUM_SLOTS = MAX_KEYS  # backward-compat alias
EMPTY_SLOT_KEYS = [f"empty_{i}" for i in range(1, MAX_KEYS + 1)]


class WorkingMemory:
    """Key-value working memory with exactly MAX_KEYS fixed slots, mirroring
    the ~4-chunk limit of human short-term memory (Cowan, 2001).

    The store always has exactly MAX_KEYS entries. It starts with MAX_KEYS
    empty placeholder slots (``empty_1`` .. ``empty_N``, value ``""``). The
    only mutation is ``replace_key``: swap an existing key (empty or filled)
    for a new key/value. Store size never changes, so "memory is full" is
    structurally impossible — every write is a replace of something already
    occupying one of the fixed slots.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {k: "" for k in EMPTY_SLOT_KEYS}

    def replace_key(self, old_key: str, new_key: str, value: str) -> str:
        if old_key not in self._store:
            return f"Error: key '{old_key}' not found — nothing to replace."
        if new_key != old_key and new_key in self._store:
            return (
                f"Error: key '{new_key}' already in use by another slot. "
                "Use that key's own name as old_key to replace it, or pick a different new_key."
            )
        value = str(value)
        if old_key == new_key:
            self._store[old_key] = value
            return f"Key '{old_key}' updated."
        # Preserve insertion order except for the slot being replaced.
        new_store: Dict[str, str] = {}
        for k, v in self._store.items():
            if k == old_key:
                new_store[new_key] = value
            else:
                new_store[k] = v
        self._store = new_store
        return f"Key '{old_key}' replaced with '{new_key}'."

    @property
    def store(self) -> Dict[str, str]:
        return dict(self._store)

    @property
    def filled_count(self) -> int:
        return sum(1 for v in self._store.values() if v != "")

    @property
    def slot_utilization(self) -> float:
        return self.filled_count / MAX_KEYS

    def snapshot(self) -> str:
        lines = []
        for k, v in self._store.items():
            lines.append(f'  "{k}": (empty slot)' if v == "" else f'  "{k}": {repr(v)}')
        return "\n".join(lines)

    def to_recall_text(self) -> str:
        filled = {k: v for k, v in self._store.items() if v != ""}
        if not filled:
            return "(memory is empty)"
        return "\n".join(f"{k}: {v}" for k, v in filled.items())

    def to_turn_text(self) -> str:
        """Full current state (including empty placeholder slots), for showing
        the model what's live *right now* mid-encoding — unlike
        ``to_recall_text()``, which drops empty slots for the final answer
        prompt where they're irrelevant, this must show every slot's exact
        current key so the model can target a valid ``old_key``.

        Explicitly labels and quotes the key (``old_key="..."``) rather than
        a bare ``key: value`` line — the plain colon format was repeatedly
        mistaken for a single string (the model would pass the whole
        "key: value" line as old_key), since nothing marked where the key
        ends and the value begins."""
        lines = []
        for k, v in self._store.items():
            if v == "":
                lines.append(f'- old_key="{k}"  (empty slot)')
            else:
                lines.append(f'- old_key="{k}"  value="{v}"')
        return "\n".join(lines)
