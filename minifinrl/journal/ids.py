"""Profile IDs: the only key to a person's journal, so they must be
impossible to guess. 16 random characters from a 30-symbol alphabet without
look-alikes (no 0/O, 1/I/L, U) give about 78 bits, written TR-XXXX-XXXX-XXXX-XXXX."""

from __future__ import annotations

import re
import secrets

ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
PATTERN = r"^TR(-[2-9A-HJKMNP-TV-Z]{4}){4}$"
_ID = re.compile(PATTERN)


def new_profile_id() -> str:
    chars = "".join(secrets.choice(ALPHABET) for _ in range(16))
    return "TR-" + "-".join(chars[i:i + 4] for i in range(0, 16, 4))


def normalise(raw: str) -> str:
    """What people type: lower case, spaces, missing dashes."""
    s = re.sub(r"[^0-9A-Za-z]", "", raw).upper()
    if s.startswith("TR"):
        s = s[2:]
    return "TR-" + "-".join(s[i:i + 4] for i in range(0, len(s), 4)) if len(s) == 16 else raw.strip().upper()


def is_profile_id(s: str) -> bool:
    return bool(_ID.match(s))


def new_trade_id() -> str:
    return secrets.token_hex(8)
