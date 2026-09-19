"""API key authentication.

One key per consumer, so a key can be revoked without affecting the others and so
the rate limiter has an identity to key on (#16). Keys come from configuration and
never from source.

Chosen over signed tokens or mTLS deliberately: this is a machine-to-machine
scoring API with a small, known set of callers. JWTs would add key rotation,
expiry and a verification path to solve problems this service does not have, and
mTLS would need a certificate authority. A shared secret compared in constant time
is the smallest thing that is actually correct here. See docs/decisions/0002.
"""

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Consumer:
    """Who is calling. Used as the rate-limit identity and in logs."""

    name: str


def parse_keys(raw: dict[str, str]) -> dict[str, Consumer]:
    """Build the lookup from a {consumer name: key} mapping."""
    return {key: Consumer(name=name) for name, key in raw.items()}


def identify(presented: str, keys: dict[str, Consumer]) -> Consumer | None:
    """Return the consumer a key belongs to, or None.

    Every configured key is compared, and all of them are compared even after a
    match, so the time taken does not reveal which key matched or how many keys
    exist. compare_digest keeps each individual comparison constant-time; a plain
    `==` leaks the length and the matching prefix.
    """
    found: Consumer | None = None
    for candidate, consumer in keys.items():
        if secrets.compare_digest(presented, candidate):
            found = consumer
    return found
