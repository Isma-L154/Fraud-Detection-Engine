"""Risk presentation rules.

These map a probability to a label a human reads. They are a **presentation**
concern and are deliberately separate from the fraud decision itself, which lives
in `Settings.decision_threshold`.

The two used to be one constant: `is_fraud` was derived from `RISK_THRESHOLDS["LOW"]`,
so every MEDIUM and HIGH transaction was fraud and LOW was not. The three-level
scale collapsed to a binary one, and retuning the operating point silently
relabelled every response.

They change for different reasons. The decision threshold moves when the cost of a
manual review changes relative to the cost of a missed fraud. The buckets move when
whoever reads the response wants a different granularity.
"""

# Upper bound of each bucket, exclusive. A probability below LOW is "LOW", below
# MEDIUM is "MEDIUM", anything else is "HIGH".
RISK_THRESHOLDS = {
    "LOW": 0.30,
    "MEDIUM": 0.70,
    "HIGH": 1.01,  # catch-all upper bound
}


def risk_level(probability: float) -> str:
    """Map a fraud probability to its presentation bucket."""
    if probability < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if probability < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"
