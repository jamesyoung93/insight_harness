"""Idempotent JSONL persistence for deterministic digest snapshots.

History is deliberately a tiny adapter around an append-only file.  Selection
logic lives in :mod:`harness.digest`; this module only provides corruption-
tolerant reads, a stable history fingerprint, and record-once semantics.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Iterable


DIGEST_HISTORY_PATH = Path(__file__).parent.parent / "data" / "digest_history.jsonl"

_WRITE_LOCK = threading.Lock()


def _stable_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def history_fingerprint(records: Iterable[dict]) -> str:
    """Hash only selection-relevant fields, never append timestamps."""

    stable = [
        {
            "digest_key": record.get("digest_key"),
            "persona": record.get("persona"),
            "scope": record.get("scope", {}),
            "data_version": record.get("data_version"),
            "governance_fingerprint": record.get("governance_fingerprint"),
            "item_keys": record.get("item_keys", []),
        }
        for record in records
    ]
    return _stable_hash(stable)


class DigestHistoryStore:
    """Small JSONL store with sequential and in-process concurrent idempotence."""

    def __init__(self, path: str | Path | None = None):
        # Resolve the module constant at call time so test/deployment adapters
        # can redirect history without relying on a definition-time default.
        self.path = Path(path if path is not None else DIGEST_HISTORY_PATH)

    def load(self) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            return []
        records: list[dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("digest_key"), str):
                records.append(record)
        return records

    def get(self, digest_key: str) -> dict | None:
        return next((record for record in self.load()
                     if record.get("digest_key") == digest_key), None)

    def recent(self, *, persona: str, exclude_key: str | None = None,
               limit: int = 5) -> list[dict]:
        if limit <= 0:
            return []
        records = [
            record for record in self.load()
            if record.get("persona") == persona
            and (exclude_key is None or record.get("digest_key") != exclude_key)
        ]
        return records[-limit:]

    def record_once(self, record: dict) -> dict:
        """Append ``record`` once per digest key and return the stored record.

        A module lock makes Streamlit threads safe.  The second read inside the
        lock is essential: two reruns may both observe a missing key before one
        acquires the writer.
        """

        digest_key = record.get("digest_key")
        if not isinstance(digest_key, str) or not digest_key:
            raise ValueError("digest history records require a non-empty digest_key")
        with _WRITE_LOCK:
            existing = self.get(digest_key)
            if existing is not None:
                return existing
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            return dict(record)
