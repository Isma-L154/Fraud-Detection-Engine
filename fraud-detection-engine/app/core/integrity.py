"""Artifact integrity.

`joblib.load` is `pickle` underneath, and unpickling executes code chosen by
whoever wrote the file. The model artifact is therefore a trust boundary, not
data: loading one is equivalent to running it.

Demonstrated rather than asserted — a crafted `.pkl` loaded with `joblib.load`
runs its `__reduce__` as the container's application user, at startup, before a
single request is served.

The mitigation here is a checksum the deployer records out of band, not one stored
next to the artifact: anyone able to rewrite the artifact could rewrite a checksum
that travels with it.

A serialisation format that is not executable — ONNX, or skops for scikit-learn —
would remove the class of problem rather than guard it, and is the better long-term
answer. It is deferred deliberately: it changes the training pipeline, the artifact
format and the loading path at once, which is its own piece of work. Tracked as a
follow-up on #15.
"""

import hashlib
from pathlib import Path

# Read in blocks rather than loading a 3.5 MB artifact into memory twice.
_BLOCK_SIZE = 1024 * 1024


def sha256_of(path: Path) -> str:
    """Hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact does not match the digest it was expected to have."""


def verify(path: Path, expected_sha256: str) -> None:
    """Raise unless `path` hashes to `expected_sha256`.

    Fails closed: a mismatch means the file on disk is not the one that was
    reviewed, and loading it would execute whatever it now contains.
    """
    actual = sha256_of(path)
    if actual != expected_sha256.lower().strip():
        raise ArtifactIntegrityError(
            f"Model artifact at {path} does not match the expected digest. "
            f"expected={expected_sha256} actual={actual}. "
            "Refusing to load: unpickling this file would execute its contents."
        )
