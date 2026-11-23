# Runtime review corrections

This change closes CR-04 (router transport), CR-07 (multipart ingestion), and
CR-09 (conversation completion) from the 2026-11-23 whole-code review.

Status: implemented. See the [review report](REVIEW_LITELLM.md) and
[local validation record](testing-results-review-litellm.json) for executed
regressions and their source snapshot. Published CI remains separate evidence.

## Required behavior

1. Router generation and embedding calls share the bounded asynchronous HTTP
   transport used by direct providers. One monotonic deadline includes client
   setup, connect, response headers, raw body reads and owned resource cleanup.
   Redirects and compressed responses remain rejected, ambient proxy settings
   remain disabled, input/output byte limits remain enforced, and public errors
   do not contain credentials or provider response content. Timeout remains a
   distinct `model_timeout` failure. The synchronous public router API is kept.
   Optional `RAGS_ROUTER_CA_FILE` supplies a bounded regular nonsymlink PEM CA
   bundle for a private HTTPS gateway. Its validated bytes are snapshotted at
   configuration load and its SHA-256 enters the execution fingerprint. Neither
   ambient CA/proxy environment variables nor `verify=False` can weaken trust.
2. Multipart file intake stays bounded. Text/PDF extraction, image validation
   and captioning, embedding and storage run outside the API event loop. A
   finite per-process admission bound rejects excess ingestion work rather than
   creating an unlimited worker queue. An admitted worker retains its slot until
   the synchronous operation actually returns, including caller cancellation.
3. A completed native agent checkpoint and its optional user/assistant history
   pair commit in one database transaction. Session ordering is allocated while
   holding the SQLite write transaction or PostgreSQL session advisory lock.
   The existing run compare-and-swap revision prevents duplicate finalization;
   rollback leaves neither a completed checkpoint nor a partial history turn.
   Tenant/subject session isolation and verified-citation provenance remain.
   No new schema or migration of old conversation messages is required.

## Acceptance evidence

- Actual loopback peers that drip headers and body bytes cannot extend router
  chat/embedding execution beyond the complete deadline. Owned HTTP resources
  close, and malformed/compressed/oversized/truncated replies retain their guards.
- Concurrent HTTP liveness remains responsive during blocked multipart text and
  image processing. Admission overflow returns a fixed capacity error and does
  not call the provider or persist a document.
- Concurrent same-session completions retain adjacent user/assistant pairs;
  injected failure between the two inserts rolls back both messages and the
  terminal checkpoint. SQLite and actual PostgreSQL execute the transaction
  checks. Repeating a stale completion cannot append a second pair.

These are integration and correctness checks with synthetic provider responses.
They do not establish real-model inference speed, quality or hardware readiness.
