# Whole-codebase review remediation and optional gateway

This change implements the RAGs findings from the 2026-11-23 review of
`9813f4cdbcd0f6de0c3dd966946162d27e15743d`. The companion execution-plane
findings and shared gateway deployment are documented in
[agentic-system](https://github.com/MasterLKH180cm/agentic-system/blob/master/docs/REVIEW_LITELLM.md).

## Corrected behavior

| Review finding | Change | Regression evidence |
| --- | --- | --- |
| CR-01: older reindexed content reappeared after source expiry | Close every superseded processing copy and choose the latest processing revision before validity filtering | SQLite and actual PostgreSQL validity, history and legacy-data cases |
| CR-02: chunk limits ran after excessive allocation | Lazy span iteration with a shared 1,024-record and 8 MiB retained UTF-8 budget before draft expansion | High-overlap, parent/child, contextual and tokenizer boundary cases |
| CR-03: migration omitted native agent data | Copy runs, messages and observations atomically, preserve identities and sequence, verify counts and reject unsupported populated tables | Actual PostgreSQL import, conflict, rollback and SQLite `ANALYZE` cases |
| CR-04: router/federation timeouts did not cover slow headers | Shared cancellable transport deadline covering connect, headers, body and client cleanup | Actual slow-drip HTTP servers for chat, embeddings and federation |
| CR-06: embedding identity ignored the gateway endpoint | Include resolved endpoint and optional immutable space ID in reuse and compatibility checks | Different same-dimension embedding spaces and normalized endpoint cases |
| CR-07: multipart ingestion blocked the API event loop | Offload extraction/provider/storage work to eight bounded slots; keep admission until cancelled background work ends | Concurrent actual HTTP upload/image and liveness probes, saturation and cancellation |
| CR-09: concurrent conversation turns interleaved | Commit completed checkpoint and validated user/assistant pair in one transaction with revision fencing | Parallel SQLite/PostgreSQL completion, rollback and duplicate finalization cases |

The detailed design contracts are [data](REVIEW_DATA_FIX_SPEC.md) and
[runtime](REVIEW_RUNTIME_FIX_SPEC.md). No coverage floor, baseline exclusion,
tenant boundary or default-on router/agent/memory policy was relaxed.

## Optional LiteLLM path

The new `litellm` CLI profile uses the existing router with verified HTTPS,
a bounded explicit CA bundle, required authentication and versioned generation
alias `gateway/inkling-2-mini-iq4nl-v1`. The CA content hash participates in the
native-agent approval fingerprint. The application has no LiteLLM SDK dependency.
Local baseline embeddings remain separate. See [LITELLM.md](LITELLM.md).

The model benchmark adds `inkling-litellm`, `--ca-file` and
`compare --require-same-deployment`. Live proxy runs require a complete deployment
manifest before credentials are read. Comparison validates all client-runtime
fields and binds adapter/harness hashes to the measurement. Two equally missing
or malformed runtime objects cannot establish parity.

## Upgrade implications

Existing router embedding profiles stored the literal endpoint `router`; they
must be reindexed before dense retrieval after this change. Offline/direct
profiles retain their previous identity when no explicit embedding space ID is
set. Give changed embedding deployments a new `RAGS_EMBEDDING_SPACE_ID` and
changed generation deployments a new immutable alias. Do not silently repoint
existing aliases used by contextual, hierarchy or graph-derived indexes.

Database import remains an explicit operator action against a quiesced source
and an empty PostgreSQL target. No user database was migrated by this development
work. The source stays read-only, and application row copies roll back together
on failure. New upload overload responses are explicit HTTP 503 responses.

## Validation and limits

[testing-results-review-litellm.json](testing-results-review-litellm.json)
records the local selected-test inventory, whole-package coverage, source hashes
and actual PostgreSQL native-agent process checks. Five external encoder/research
tests are explicitly delegated to the existing dedicated CI jobs. Published
commit validation is available in
[RAGs CI](https://github.com/MasterLKH180cm/RAGs/actions/workflows/ci.yml).

The companion repository checks real LiteLLM processes, TLS, virtual keys,
PostgreSQL and both production adapters against deterministic HTTP fixtures.
Fixtures establish contracts, not real Inkling quality, latency or throughput.
Actual model performance remains unmeasured on this host because the selected
weights exceed available RAM and no GPU is available. A complete deployment
manifest records operator claims; it does not attest to the loaded weights.
