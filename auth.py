"""auth.py -- local users, roles and password verification.

What this is: a single-machine access gate for the Streamlit UI, with three
roles that decide which panels a person can use. Passwords are PBKDF2-HMAC-
SHA256 with a per-user random salt, compared in constant time, stored in a local
SQLite file. Repeated failures lock an account out for a while. Nothing is sent
anywhere.

What this is NOT, stated plainly so nobody over-trusts it:
  * Not multi-tenant security. Anyone who can read `auth/users.db` and run this
    code can attempt passwords; anyone with shell access to the machine already
    owns everything.
  * Not encryption at rest. The index, chat history and uploaded files are all
    unencrypted; use OS/disk encryption if that matters.
  * Not protection for `api.py`, which has **no** authentication. Bind it to
    127.0.0.1 (the default) or put an authenticating reverse proxy in front.

Roles are deliberately coarse, matching how the tool is actually used: a
reviewer asks questions, an analyst also feeds the corpus, an administrator also
removes sources and manages accounts.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from config import settings

logger = logging.getLogger("evidence_ai.auth")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    role            TEXT NOT NULL,
    salt            TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);

-- Bearer tokens for api.py. Only the SHA-256 digest is stored, plus a short
-- prefix so a person can tell two tokens apart in a list. Tokens are
-- high-entropy random strings, so a fast hash is correct here: PBKDF2 exists to
-- slow down guessing of human-chosen passwords, and there is nothing human
-- about these.
CREATE TABLE IF NOT EXISTS api_tokens (
    token_hash   TEXT PRIMARY KEY,
    prefix       TEXT NOT NULL,
    username     TEXT NOT NULL,
    label        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);
"""

TOKEN_PREFIX = "eai_"
# Shown in listings so tokens can be told apart; far too short to guess from.
TOKEN_VISIBLE_CHARS = 12

_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


# Coarse on purpose. "ask" is the read path; everything else mutates state or
# exports it.
PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"ask"}),
    Role.ANALYST: frozenset({"ask", "ingest"}),
    Role.ADMIN: frozenset({
        "ask", "ingest", "delete", "manage_collections",
        "export_history", "manage_users",
    }),
}


def can(role: "Role | str | None", permission: str) -> bool:
    """Permission check. Unknown or missing roles get nothing."""
    try:
        resolved = role if isinstance(role, Role) else Role(role)
    except ValueError:
        return False
    return permission in PERMISSIONS[resolved]


class AuthError(ValueError):
    """A user-facing refusal: weak password, duplicate account, bad role."""


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex). A fresh salt per user, never per login."""
    salt = salt or secrets.token_bytes(32)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, settings.auth_pbkdf2_iterations
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Constant-time comparison; a mismatch must not leak via timing."""
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, settings.auth_pbkdf2_iterations
    )
    return hmac.compare_digest(digest.hex(), hash_hex or "")


def validate_password(password: str) -> None:
    if len(password) < settings.auth_min_password_length:
        raise AuthError(
            f"Password must be at least {settings.auth_min_password_length} characters."
        )


def validate_username(username: str) -> str:
    name = (username or "").strip()
    if not _USERNAME_RE.match(name):
        raise AuthError(
            "Username must be 3-32 characters using letters, digits, dot, "
            "underscore or hyphen."
        )
    return name


class UserStore:
    """SQLite-backed accounts. One connection per process behind a lock, like
    the rest of this project's stores."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.auth_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- reads ----------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def needs_bootstrap(self) -> bool:
        """True when no account exists yet, so the UI can offer to create the
        first administrator instead of shipping a default password."""
        return self.count() == 0

    def get(self, username: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, role, created_at, last_login_at, failed_attempts, "
                "locked_until FROM users ORDER BY username"
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- writes ---------------------------------------------------------

    def create_user(self, username: str, password: str, role: str = Role.VIEWER.value) -> dict:
        name = validate_username(username)
        validate_password(password)
        resolved = self._resolve_role(role)
        if self.get(name):
            raise AuthError(f"Account '{name}' already exists.")
        salt, digest = hash_password(password)
        with self._lock:
            self._conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?,?)",
                (name, resolved.value, salt, digest, _iso(_now()), None, 0, None),
            )
            self._conn.commit()
        logger.info("Created %s account '%s'", resolved.value, name)
        return {"username": name, "role": resolved.value}

    def set_password(self, username: str, password: str) -> None:
        validate_password(password)
        salt, digest = hash_password(password)
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE users SET salt = ?, password_hash = ?, failed_attempts = 0, "
                "locked_until = NULL WHERE username = ?",
                (salt, digest, username),
            )
            self._conn.commit()
        if not cursor.rowcount:
            raise AuthError(f"No account named '{username}'.")

    def set_role(self, username: str, role: str) -> None:
        resolved = self._resolve_role(role)
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                (resolved.value, username),
            )
            self._conn.commit()
        if not cursor.rowcount:
            raise AuthError(f"No account named '{username}'.")

    def delete_user(self, username: str) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM users WHERE username = ?", (username,)
            )
            # Tokens must not outlive the account. authenticate_token already
            # refuses a token whose user is gone, but leaving the hashes behind
            # keeps credential material in a database an operator believes is
            # clean -- found by inspection after a test account was deleted and
            # its revoked token row survived.
            self._conn.execute(
                "DELETE FROM api_tokens WHERE username = ?", (username,)
            )
            self._conn.commit()
        if not cursor.rowcount:
            raise AuthError(f"No account named '{username}'.")
        if self.count() == 0:
            logger.warning("Last account deleted; the UI will offer bootstrap again")

    def authenticate(self, username: str, password: str) -> dict | None:
        """Return the account on success, None on failure.

        A locked account is refused without checking the password, and every
        failure pushes the lockout closer. The lock is per account, which is the
        honest limit of a local gate: it slows guessing, it does not stop an
        attacker with the database file.
        """
        row = self.get((username or "").strip())
        if row is None:
            # Hash anyway so a missing account and a wrong password take
            # similar time and the username list is not enumerable by timing.
            verify_password(password, "00" * 32, "00" * 32)
            return None

        locked_until = row["locked_until"]
        if locked_until:
            try:
                if _now() < datetime.fromisoformat(locked_until):
                    logger.warning("Locked account attempt: %s", row["username"])
                    return None
            except ValueError:
                pass

        if verify_password(password, row["salt"], row["password_hash"]):
            with self._lock:
                self._conn.execute(
                    "UPDATE users SET failed_attempts = 0, locked_until = NULL, "
                    "last_login_at = ? WHERE username = ?",
                    (_iso(_now()), row["username"]),
                )
                self._conn.commit()
            return {"username": row["username"], "role": row["role"]}

        attempts = row["failed_attempts"] + 1
        lock = (
            _iso(_now() + timedelta(minutes=settings.auth_lockout_minutes))
            if attempts >= settings.auth_max_failed_attempts
            else None
        )
        with self._lock:
            self._conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                (0 if lock else attempts, lock, row["username"]),
            )
            self._conn.commit()
        logger.warning(
            "Failed login for '%s' (%s)", row["username"],
            "now locked" if lock else f"attempt {attempts}",
        )
        return None

    def locked_until(self, username: str) -> str | None:
        """ISO timestamp this account is locked until, or None if it is not.

        `authenticate` answers None for a wrong password AND for a locked account,
        which is deliberate: neither outcome should confirm the other, and a
        missing account is hashed anyway so usernames are not enumerable by
        timing. A caller that needs to tell a legitimate user "stop trying for
        five minutes" asks here instead -- otherwise that user keeps mistyping and
        keeps the lock alive.
        """
        row = self.get((username or "").strip())
        if row is None or not row["locked_until"]:
            return None
        try:
            if _now() < datetime.fromisoformat(row["locked_until"]):
                return row["locked_until"]
        except ValueError:
            return None
        return None

    # ---- API tokens -----------------------------------------------------

    def create_token(self, username: str, label: str = "") -> str:
        """Mint a bearer token for `username`.

        Returns the plaintext token exactly once -- it is never stored, only its
        digest is. Losing it means minting another and revoking this one.
        """
        if not self.get(username):
            raise AuthError(f"No account named '{username}'.")
        token = f"{TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_tokens VALUES (?,?,?,?,?,?,?)",
                (digest, token[:TOKEN_VISIBLE_CHARS], username, (label or "").strip(),
                 _iso(_now()), None, 0),
            )
            self._conn.commit()
        logger.info("Issued API token '%s...' for %s", token[:TOKEN_VISIBLE_CHARS], username)
        return token

    def list_tokens(self, username: str | None = None) -> list[dict]:
        query = ("SELECT prefix, username, label, created_at, last_used_at, revoked "
                 "FROM api_tokens")
        params: tuple = ()
        if username:
            query += " WHERE username = ?"
            params = (username,)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY created_at", params).fetchall()
        return [dict(row) for row in rows]

    def revoke_token(self, prefix: str, username: str | None = None) -> None:
        """Revoke by visible prefix, so nobody has to keep the plaintext around
        just to cancel it."""
        with self._lock:
            if username:
                cursor = self._conn.execute(
                    "UPDATE api_tokens SET revoked = 1 WHERE prefix = ? AND username = ?",
                    (prefix, username),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE api_tokens SET revoked = 1 WHERE prefix = ?", (prefix,)
                )
            self._conn.commit()
        if not cursor.rowcount:
            raise AuthError(f"No API token starting '{prefix}'.")

    def authenticate_token(self, token: str) -> dict | None:
        """Resolve a bearer token to an account, or None.

        A revoked token, an unknown token and a token whose account was deleted
        all fail the same way. Password lockout does not apply: the lockout
        exists to slow password guessing, and a 256-bit random token is not
        guessable -- but deleting the account kills its tokens immediately.
        """
        if not token:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_tokens WHERE token_hash = ?", (digest,)
            ).fetchone()
            if row is None or row["revoked"]:
                return None
            account = self.get(row["username"])
            if account is None:
                return None
            self._conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE token_hash = ?",
                (_iso(_now()), digest),
            )
            self._conn.commit()
        return {
            "username": account["username"],
            "role": account["role"],
            "token_label": row["label"],
            "via": "api_token",
        }

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    @staticmethod
    def _resolve_role(role) -> Role:
        try:
            return role if isinstance(role, Role) else Role(role)
        except ValueError:
            raise AuthError(
                f"Unknown role '{role}'. Choose one of: "
                + ", ".join(r.value for r in Role)
            ) from None


_store: UserStore | None = None
_store_lock = threading.Lock()


def get_user_store() -> UserStore:
    """Process-wide singleton, mirroring get_pipeline() and get_history()."""
    global _store
    with _store_lock:
        if _store is None:
            _store = UserStore()
        return _store
