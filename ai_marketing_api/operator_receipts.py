"""Private durable RPC receipts, never a knowledge store or model retry queue.

A reservation identity is claimed exactly once across processes. A surviving
``running`` row after a hard crash remains pending; another worker cannot prove
the provider stopped, and must never buy a replacement invocation implicitly.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Any


SCHEMA = "operator.editorial_receipt.v1"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_RESERVATION = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_STEPS = frozenset({"draft_master", "adapt_platform", "verify_claims", "review_editorial"})
_ROLES = frozenset({"commercial", "industry", "personal_ip"})


class EditorialReceiptError(Exception):
    def __init__(self, code: str, status_code: int):
        super().__init__(code)
        self.code, self.status_code = code, status_code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def request_sha256(wire_payload: dict) -> str:
    return hashlib.sha256(canonical_json(wire_payload)).hexdigest()


def _state_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except ImportError:
        # The standalone API Docker image intentionally ships no Hermes core.
        # Match its existing profile-scoped config directory in that image.
        home = Path(os.getenv("HERMES_HOME", "/data/hermes"))
    return home / "marketing_api" / "editorial_receipts.sqlite3"


class EditorialReceiptStore:
    def __init__(self, path: Path | None = None):
        self.path = path or _state_path()

    @contextmanager
    def _connection(self):
        connection = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name == "posix":
                # A pre-created container volume may start as 0755. Tighten
                # only this API's dedicated state directory, never HERMES_HOME.
                self.path.parent.chmod(0o700)
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if os.name == "posix":
                    os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            # journal_mode does not consistently honor SQLite's busy handler
            # during concurrent first-open initialization. Only this local
            # metadata operation is retried; a model invocation never is.
            deadline = time.monotonic() + 10
            while True:
                try:
                    connection.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("""CREATE TABLE IF NOT EXISTS editorial_receipts (
                role_id TEXT NOT NULL, step TEXT NOT NULL, reservation_id TEXT NOT NULL,
                request_sha256 TEXT NOT NULL, status TEXT NOT NULL
                    CHECK(status IN ('running','completed','unknown')),
                claim_token TEXT NOT NULL, response_json TEXT, response_sha256 TEXT,
                failure_code TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(role_id,step,reservation_id))""")
            yield connection
        except (OSError, sqlite3.Error):
            raise EditorialReceiptError("editorial_receipt_store_unavailable", 503) from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _scope(role_id, step, reservation_id, sha):
        if (role_id not in _ROLES or step not in _STEPS
                or not isinstance(reservation_id, str) or not _RESERVATION.fullmatch(reservation_id)
                or not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha)):
            raise EditorialReceiptError("editorial_receipt_binding_invalid", 422)

    @staticmethod
    def _check_hash(row, sha):
        if not secrets.compare_digest(row["request_sha256"], sha):
            raise EditorialReceiptError("editorial_receipt_request_mismatch", 409)

    @staticmethod
    def _envelope(row):
        value = {"schema_version": SCHEMA, "status": row["status"],
                 "role_id": row["role_id"], "step": row["step"],
                 "budget_reservation_id": row["reservation_id"],
                 "request_sha256": row["request_sha256"]}
        if row["status"] == "completed":
            raw = row["response_json"]
            try:
                if (not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES
                        or hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["response_sha256"]):
                    raise ValueError()
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise ValueError()
            except (ValueError, TypeError):
                raise EditorialReceiptError("editorial_receipt_response_invalid", 503) from None
            value["response"] = response
        return value

    def get(self, role_id, step, reservation_id, sha):
        self._scope(role_id, step, reservation_id, sha)
        # Reading a missing receipt must not initialize a state database.
        if not self.path.exists():
            raise EditorialReceiptError("editorial_receipt_not_found", 404)
        try:
            with sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=10) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute("SELECT * FROM editorial_receipts WHERE role_id=? AND step=? AND reservation_id=?",
                                         (role_id, step, reservation_id)).fetchone()
        except sqlite3.Error:
            raise EditorialReceiptError("editorial_receipt_store_unavailable", 503) from None
        if row is None:
            raise EditorialReceiptError("editorial_receipt_not_found", 404)
        self._check_hash(row, sha)
        return self._envelope(row)

    def claim(self, role_id, step, reservation_id, sha):
        self._scope(role_id, step, reservation_id, sha)
        token = secrets.token_hex(32)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT OR IGNORE INTO editorial_receipts(role_id,step,reservation_id,request_sha256,status,claim_token) VALUES(?,?,?,?,'running',?)",
                               (role_id, step, reservation_id, sha, token))
            row = connection.execute("SELECT * FROM editorial_receipts WHERE role_id=? AND step=? AND reservation_id=?",
                                     (role_id, step, reservation_id)).fetchone()
            self._check_hash(row, sha)
            connection.commit()
        return token if secrets.compare_digest(row["claim_token"], token) else None, self._envelope(row)

    def finish(self, role_id, step, reservation_id, token, response):
        raw = canonical_json(response)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise EditorialReceiptError("editorial_receipt_response_too_large", 503)
        with self._connection() as connection:
            changed = connection.execute("""UPDATE editorial_receipts
                SET status='completed',response_json=?,response_sha256=?,updated_at=CURRENT_TIMESTAMP
                WHERE role_id=? AND step=? AND reservation_id=? AND claim_token=? AND status='running'""",
                (raw.decode("utf-8"), hashlib.sha256(raw).hexdigest(), role_id, step, reservation_id, token)).rowcount
            if changed != 1:
                raise EditorialReceiptError("editorial_receipt_claim_invalid", 409)

    def unknown(self, role_id, step, reservation_id, token, *, cancelled=False):
        with self._connection() as connection:
            connection.execute("""UPDATE editorial_receipts
                SET status='unknown',failure_code=?,updated_at=CURRENT_TIMESTAMP
                WHERE role_id=? AND step=? AND reservation_id=? AND claim_token=? AND status='running'""",
                ("editorial_step_cancelled" if cancelled else "editorial_step_failed", role_id, step, reservation_id, token))


async def execute_with_receipt(*, role_id, step, reservation_id, wire_payload, execute):
    store = EditorialReceiptStore()
    sha = request_sha256(wire_payload)
    token, existing = store.claim(role_id, step, reservation_id, sha)
    if token is None:
        if existing["status"] == "completed":
            return existing["response"]
        raise EditorialReceiptError("editorial_receipt_" + existing["status"], 409)
    try:
        response = await execute()
        # Persist the complete private RPC reply before exposing it to a caller.
        # A known completed RPC can still contain an unknown provider outcome;
        # its original diagnostic reply is cached too, never bought again.
        store.finish(role_id, step, reservation_id, token, response)
        return response
    except BaseException as exc:
        try:
            store.unknown(role_id, step, reservation_id, token, cancelled=isinstance(exc, asyncio.CancelledError))
        except EditorialReceiptError:
            pass  # A durable running marker still forbids another invocation.
        raise
