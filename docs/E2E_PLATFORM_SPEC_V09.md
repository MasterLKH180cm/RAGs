# Agent Platform process and browser acceptance specification

This specification precedes the new process-level acceptance harness.

The gate starts the shipped RAG API in an independent process and a bounded,
loopback-only OpenAI-compatible fixture. It uses synthetic policy documents and
disposable credentials. No production endpoint, private content, paid model,
or downloaded model weights are involved. These are integration and UI contract
results, never Inkling model quality or inference-throughput measurements.

Required HTTP checks:

1. Send the canonical Inkling model identity, `max_completion_tokens` and the
   selected reasoning effort through the production ModelRouter.
2. Complete a real agent-to-tool-to-RAG-to-model round trip, preserve the actual
   retrieved citation, and persist the authorized conversation.
3. Reject a truncated tool-call response before preparing or executing a tool.
4. Deny another tenant access to runs and conversation memory.
5. Execute the shipped evidence-review workflow, persist its approval boundary,
   restart the API process, resume from the recorded revision, and reject replay.
6. Exercise the admin cookie/CSRF interface against the live HTTP application.

An explicit browser gate additionally uses real Chromium to sign in, perform
an agent task, execute/approve the visual workflow and sign out. It checks that
credentials never enter browser storage or URLs and that logout clears results.
Missing browser packages or executables fail this selected gate; the HTTP-only
mode has an explicitly smaller declared scope and contains no skipped tests.

Every gate writes expected and completed checks plus cleanup status. Any failed
check exits nonzero. Reports must omit credentials, raw model prompts and model
reasoning. Optional PostgreSQL execution requires an explicitly supplied,
disposable `RAGS_E2E_POSTGRES_URL`; it must never silently select SQLite.
