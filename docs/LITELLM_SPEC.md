# Optional shared generation gateway contract

Status: implemented, 2026-11-23. The [review report](REVIEW_LITELLM.md) and
[local validation record](testing-results-review-litellm.json) document acceptance
evidence and its limits. The companion Agentic repository's `docs/LITELLM_SPEC.md`
defines the shared deployment and complete acceptance contract. RAGs exposes an additive `litellm` configuration profile
using its existing typed router and the versioned gateway alias
`gateway/inkling-2-mini-iq4nl-v1`.

The router must enforce a true wall deadline over connection, headers, body
and cleanup. Production uses verified HTTPS, a service-scoped credential and
an optional explicitly trusted CA. A changed CA is part of the approval trust
fingerprint. Tool permissions, approval, input/output bounds and private
reasoning protection remain application responsibilities.

The profile uses explicit OpenAI `max_tokens` semantics without a reasoning
effort override. Generation is opt-in; default-on router/agent/memory flags do
not authorize unconfigured outbound calls. Existing direct and local Inkling
profiles remain available for comparison. Embeddings retain their own provider
and versioned identity and are not sent through generation fallback routes.

The data and runtime review specifications require regressions for source
validity, bounded chunk allocation, complete native-agent migration, router and
federation deadlines, resolved embedding identity, responsive upload handlers,
and atomic durable conversation finalization. Core tests, actual PostgreSQL,
whole-package coverage, real HTTP smoke and the pinned proxy fixture gate must
pass. Fixture timing is not real Inkling model performance.
