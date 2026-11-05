"""
scripts/create_user.py -- manage local accounts from the command line.

The UI can create accounts too, but a headless machine (or a first deployment
where nobody should log in as an administrator yet) needs this. Passwords are
read with getpass so they never land in shell history or the process list.

    .venv\\Scripts\\python scripts\\create_user.py --username reviewer --role viewer
    .venv\\Scripts\\python scripts\\create_user.py --list
    .venv\\Scripts\\python scripts\\create_user.py --username reviewer --set-password
    .venv\\Scripts\\python scripts\\create_user.py --username reviewer --delete

API tokens (for api.py, which requires one on every endpoint except /health):
    .venv\\Scripts\\python scripts\\create_user.py --username svc --create-token --label ingest-bot
    .venv\\Scripts\\python scripts\\create_user.py --list-tokens
    .venv\\Scripts\\python scripts\\create_user.py --revoke-token eai_AbCdEfGh

The plaintext token is printed once and never stored -- only its SHA-256 digest
is. Lose it and you mint another.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth import AuthError, Role, get_user_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="Account name (3-32 chars)")
    parser.add_argument("--role", default=Role.VIEWER.value,
                        choices=[r.value for r in Role],
                        help="What the account may do (default: viewer)")
    parser.add_argument("--list", action="store_true", help="List accounts and exit")
    parser.add_argument("--set-password", action="store_true",
                        help="Replace the password of an existing account")
    parser.add_argument("--set-role", action="store_true",
                        help="Change the role of an existing account")
    parser.add_argument("--delete", action="store_true", help="Delete the account")
    parser.add_argument("--create-token", action="store_true",
                        help="Mint an API bearer token for --username")
    parser.add_argument("--label", default="", help="Label for the new token")
    parser.add_argument("--list-tokens", action="store_true",
                        help="List API tokens (prefixes only) and exit")
    parser.add_argument("--revoke-token", metavar="PREFIX",
                        help="Revoke the token whose visible prefix matches")
    args = parser.parse_args()

    store = get_user_store()

    if args.list:
        users = store.list_users()
        if not users:
            print("No accounts yet. Create the first one with --username/--role.")
            return 0
        print(f"{'USERNAME':22s} {'ROLE':10s} {'CREATED':26s} {'LAST LOGIN':26s}")
        for user in users:
            print(f"{user['username']:22s} {user['role']:10s} "
                  f"{user['created_at']:26s} {user['last_login_at'] or '-':26s}")
        return 0

    if args.list_tokens:
        tokens = store.list_tokens()
        if not tokens:
            print("No API tokens yet.")
            return 0
        print(f"{'TOKEN':16s} {'ACCOUNT':18s} {'LABEL':18s} {'REVOKED':8s} LAST USED")
        for token in tokens:
            print(f"{token['prefix'] + '...':16s} {token['username']:18s} "
                  f"{(token['label'] or '-'):18s} "
                  f"{'yes' if token['revoked'] else 'no':8s} "
                  f"{token['last_used_at'] or 'never'}")
        return 0

    if args.revoke_token:
        try:
            store.revoke_token(args.revoke_token)
            print(f"Revoked token '{args.revoke_token}...'.")
            return 0
        except AuthError as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 1

    if not args.username:
        parser.error("--username is required unless --list or --list-tokens is used")

    try:
        if args.create_token:
            token = store.create_token(args.username, args.label)
            print(f"Token for '{args.username}' (shown once -- store it now):")
            print(f"  {token}")
            print("Use it as:  Authorization: Bearer " + token)
            return 0

        if args.delete:
            store.delete_user(args.username)
            print(f"Deleted account '{args.username}'.")
            return 0

        if args.set_password:
            password = getpass.getpass("New password: ")
            if password != getpass.getpass("Repeat password: "):
                print("Passwords do not match.", file=sys.stderr)
                return 1
            store.set_password(args.username, password)
            print(f"Password updated for '{args.username}'.")
            return 0

        if args.set_role:
            store.set_role(args.username, args.role)
            print(f"'{args.username}' is now a {args.role}.")
            return 0

        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 1
        created = store.create_user(args.username, password, args.role)
        print(f"Created {created['role']} account '{created['username']}'.")
        print("Passwords are stored as PBKDF2-HMAC-SHA256 with a per-user salt.")
        return 0
    except AuthError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
