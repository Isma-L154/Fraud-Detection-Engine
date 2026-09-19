"""The on-disk artifact format.

The artifact used to be a bare sklearn Pipeline, and the version the API reported
was a string constant in source. The two were unconnected: retraining and replacing
the file left the service still reporting 1.0.0, so a fraud decision could not be
attributed to the model that made it. For a system making financial determinations
that is the audit trail failing.

The artifact now carries its own metadata, in the same file, so it cannot drift
from the model it describes — and so the integrity digest in app/core/integrity.py
covers the metadata as well as the weights.

Bare pipelines are still loadable. An artifact trained before this change is not
wrong, it is unlabelled, and the loader says so rather than inventing a version.
"""

from dataclasses import dataclass, field
from typing import Any

# Bumped when the envelope's shape changes, not when a model is retrained.
FORMAT_VERSION = 1

UNKNOWN_VERSION = "unknown"


@dataclass(frozen=True)
class ArtifactMetadata:
    """What produced this model, and how it scored when it was produced."""

    # MLflow run id — the lookup key for the parameters, metrics and artifacts of
    # the run that produced this file.
    run_id: str = ""
    trained_at: str = ""
    sklearn_version: str = ""
    # Which data produced it. A model fitted on the synthetic sample is not a
    # meaningful scorer, and the artifact should say so rather than look identical.
    dataset: str = ""
    # Metrics measured at the operating point the service actually uses, not at
    # sklearn's default 0.5 (#17).
    metrics: dict[str, float] = field(default_factory=dict)
    decision_threshold: float | None = None

    @property
    def version(self) -> str:
        """The identifier reported in every prediction response."""
        return self.run_id or UNKNOWN_VERSION


@dataclass(frozen=True)
class Artifact:
    """A pipeline together with the record of what produced it."""

    pipeline: Any
    metadata: ArtifactMetadata


def envelope(pipeline: Any, metadata: ArtifactMetadata) -> dict[str, Any]:
    """The dict written to disk by the training script."""
    return {
        "format_version": FORMAT_VERSION,
        "pipeline": pipeline,
        "metadata": metadata,
    }


def unpack(loaded: Any) -> Artifact:
    """Interpret whatever `joblib.load` returned.

    Accepts the current envelope and a bare pipeline. A bare pipeline yields empty
    metadata, so the service reports "unknown" rather than a version it made up.
    """
    if isinstance(loaded, dict) and "pipeline" in loaded:
        metadata = loaded.get("metadata") or ArtifactMetadata()
        if isinstance(metadata, dict):
            metadata = ArtifactMetadata(**metadata)
        return Artifact(pipeline=loaded["pipeline"], metadata=metadata)

    return Artifact(pipeline=loaded, metadata=ArtifactMetadata())
