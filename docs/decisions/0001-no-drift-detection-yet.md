# 0001 — No drift detection, and `evidently` removed

**Status:** accepted · 2026-09-19
**Issue:** #23

## Context

`evidently` was a pinned dependency, imported nowhere, and the README described a
"Monitoring Layer — drift detection and experiment tracking" that did not exist.

Drift matters here more than in most systems. Fraud patterns change
*adversarially* — that is the nature of the problem — so a model trained on a
fixed historical dataset degrades by default rather than by accident. The service
currently has no way to know that the transactions it scores no longer resemble
the ones it was trained on.

## The decision

**Remove `evidently`. Do not implement distribution-based drift detection yet.
Use the prediction distribution already exposed in metrics as the interim signal.**

## Why

Implementing it properly needs four things this project does not have:

1. **A home for the reference distribution.** Drift is measured against training
   data — a 150 MB file the service does not have and should not carry.
2. **Somewhere to collect live feature distributions.** The service is stateless
   and stores nothing. Persisting scored transactions introduces a datastore,
   which activates the row-level-security control currently marked N/A, and raises
   a retention question about what is financial data — V1-V28 are PCA components
   of real card transactions.
3. **A batch pipeline.** Computing drift in the request path would add latency to
   every prediction to answer a question that only makes sense over a window.
4. **A decision about what happens when drift is found.** An alert, a metric, or
   an automatic retrain. The automatic one is dangerous: a model that retrains
   itself on drifted data is a model that adapts to an attacker.

The cost is a datastore, a retention policy, an RLS implementation and a scheduled
job. That is a milestone, not a dependency.

## What we have instead

`fraud_predictions_total{risk_level, is_fraud}` (#22). **A shift in the output
distribution is itself a drift signal** — if the share of HIGH-risk predictions
moves without a corresponding change in traffic, something has changed in the
inputs or in the world.

It is weaker than feature-level drift detection: it sees the model's *reaction* to
drift rather than the drift itself, so it cannot attribute a change to a feature,
and it will not notice drift the model happens to be robust to. But it costs
nothing, needs no datastore, retains no personal data, and is already deployed and
scrapeable.

## Revisit when

- a datastore is introduced for any reason — the main blocker disappears
- the prediction distribution shifts and nobody can say which feature moved
- the model is retrained on new data and the two need comparing

## Consequences

- `evidently` and its 71-package dependency closure are gone from the project (#9)
- The README no longer claims drift detection (#27)
- Control 6 (row-level security) stays N/A, because no datastore was introduced
