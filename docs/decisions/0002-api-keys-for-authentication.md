# 0002 — API keys, not JWTs or mTLS

**Status:** accepted · 2026-09-19
**Issues:** #10, #11, #16

## Context

No endpoint authenticated anything. `/predict`, `/predict/batch` and `/retrain`
were open to anyone who could reach the port. The only access restriction was the
CORS allowlist, which restrains browsers and not `curl` — the exact
misunderstanding the security baseline warns about.

Without authentication there was also no account to rate limit against, so the
limiter keyed on IP, and `/retrain` was an unprotected write endpoint waiting for
an implementation.

## The decision

**One API key per consumer, in an `X-API-Key` header, compared in constant time.**

## Why not the alternatives

**Signed tokens (JWT)** solve delegation, expiry and stateless verification across
services. This is a single service with a small, known set of machine callers.
Adopting JWTs would add key rotation, clock-skew handling and a verification path
to solve problems that do not exist here, and each of those is somewhere a
mistake can hide.

**mTLS** is stronger and is the right answer inside a service mesh. It needs a
certificate authority, issuance and renewal. There is no deployment at all — that
infrastructure would exist solely to authenticate a service nobody is calling.

**A shared secret compared in constant time** is the smallest thing that is
actually correct for this shape of API. If real consumers appear and delegation or
expiry becomes a genuine requirement, moving to signed tokens is a contained
change: the dependency boundary (`require_consumer`) already exists.

## Consequences

- Keys come from configuration, never source, and `Settings` refuses to build
  outside development without them — the service cannot accidentally start open.
- Keys under 32 characters are rejected, and two consumers cannot share one: a
  shared key cannot be revoked independently, and the limiter could not tell them
  apart.
- **Rate limiting is now per account** (#16). IPs are shared behind carrier NAT
  and rotate; a per-IP limit punishes the wrong callers and fails to stop the
  right ones.
- `X-Forwarded-For` is honoured only from a configured proxy address. It is
  client-controlled: trusting it unconditionally lets anyone send a different
  address per request and never be limited — worse than ignoring it, because the
  limit then looks like a control while being none.
- `/health` stays unauthenticated. A load balancer cannot present a credential and
  liveness is not sensitive.
- **Development with no keys configured serves anonymously.** This is the one
  place the system fails open, and it is bounded by configuration rather than by a
  runtime check that could be wrong.

## Revisit when

- a consumer needs delegated or time-limited access
- consumers stop being machines under the same operator
- the service moves inside a mesh where mTLS is ambient rather than bespoke
