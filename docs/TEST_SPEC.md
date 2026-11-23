# Complete test layers and bounded code-quality corrections

Status: specified before implementation of this testing iteration.
Baseline service commit: `71c484527e534e893929ada5b694440efd7bb7a9`.
Companion baseline: `7d1616e1d696d74733343e53135f5bc9e11b9e6a`.

## Goal and boundaries

Make unit, integration and end-to-end verification independently runnable and
auditable; add meaningful missing contracts and fix defects exposed by them.
Coverage measures executed code, not correctness or model quality. Do not claim
100% behavioral completeness or invent support for unexecuted upstream
research engines. The subsequent four-engine implementation follows its own
[committed specification](RESEARCH_COMPLETION_SPEC.md), with separate actual
algorithm, trained-checkpoint and HTTP gates described in
[the execution ledger](RESEARCH_COMPLETION.md). Frozen admission data, profiles, prompts and quality targets
remain unchanged. A valid-provider response must preserve existing behavior.

The baseline ordinary suite is 280 passed and 5 optional model tests skipped.
With all `rags` modules included, line coverage is 3007/3963 (75.8769%) and
branch coverage is 1018/1486 (68.5061%). MCP and worker subprocesses and optional
models have separate execution evidence and are not covered by that parent-only
measurement. Preserve the scope when comparing before and after numbers.

## T-RAG-01: explicit test layers

- Every collected pytest case has exactly one primary layer: `unit`,
  `integration` or `e2e`. Unit tests isolate pure logic or one component with
  controlled dependencies. Integration tests execute real storage, ASGI routes,
  network servers or process boundaries. E2E tests drive actual service APIs and
  independent workers, or the actual browser interface.
- Classify existing mixed test files at function granularity. Unknown/unmarked
  tests and conflicting primary markers must fail collection. New test files
  may use explicit module/function markers.
- Resource markers identify actual encoders and each research engine. Ordinary
  core selections explicitly deselect those resources; dedicated jobs provision
  and execute them. Do not count a deselected or skipped test as passed.
- Strict resource mode fails a selected test suite if any case skips, xfails or
  unexpectedly passes an expected-failure marker. Missing requested services,
  packages or checkpoints must not produce a green full gate.
- Emit JUnit results and a machine-readable collection inventory with node IDs,
  layers and resource markers. Counts from overlapping jobs are not added as
  unique tests.

## T-RAG-02: missing unit and provider contracts

- Validate OpenAI embedding/chat and Anthropic content envelopes, including
  missing/wrong nesting, empty choices/blocks, non-text values, booleans,
  nonfinite numbers and oversized numeric values. Malformed successful provider
  responses must return fixed typed 502 errors, not uncaught exceptions or
  client-validation errors. Provider body text and credentials must not leak.
- Extract small shared response-validation helpers to replace repeated unsafe
  direct indexing. Preserve prompts, model identities, valid text/vector
  outputs, token budgets and bounded structured-output behavior.
- Test budget rejection before a provider request, usage-field filtering and
  concurrent request-local usage isolation.
- Add pure contracts for native ranking/aggregation, source identity, finite
  values and exact zero; avoid tests that merely repeat implementation lines.
- Review static diagnostics. Preserve intentional multilingual punctuation and
  framework conventions; do not apply broad cosmetic rewrites or blanket
  suppressions. Make sequence alignment/adjacent-pair intent explicit where a
  concrete ambiguity exists, retaining valid-input results.

## T-RAG-03: real storage and API integration

- Exercise successful UTF-8 and real PDF upload, empty/invalid encoding,
  unsupported media, page/extracted-text limits and unauthenticated access.
  Assert authorization before extraction/provider/storage work, without claiming
  that the framework never parses incoming multipart framing.
- Inject a failure after SQL mutation begins and prove atomic rollback across
  source revisions, chunks, FTS and derived indexes. A provider failure before
  a transaction does not satisfy this rollback test.
- Run simultaneous CAS reindex writers against actual SQLite. Exactly one
  expected-revision write may win; the losing write must not corrupt source
  history or derived records. Preserve tenant isolation and delete semantics.
- Close an acquired SQLite connection even when connection initialization or a
  PRAGMA fails before normal transaction work. Protect setup with the same
  cleanup scope and exercise a failing setup on a tracked real connection.
- Retain existing ingestion, six chunking strategies, token budgets, retrieval
  variants, temporal/structured results, refusal and federation regressions.

## T-RAG-04: real end-to-end gates

- Keep actual RAPTOR and LightRAG algorithm/HTTP lifecycle gates and the trained
  BGE, ColBERT and CLIP checks. Select the intended engine explicitly instead of
  counting the other engine's skipped test as success.
- The companion real RAG/API/worker/MCP flow must remain valid, including
  refusal, exact structured zero, source update/delete/history and tenant scope.
- Add the companion's actual process-loss/restart, cancellation/late-result,
  SSE-resume and browser scenarios described in its TEST_SPEC. No direct SQL
  state rewriting, in-process tool monkeypatching or paid-model dependency may
  stand in for those E2E paths.
- Preserve the actual fixed-profile runtime contract and historical quality
  audit. The manual complete quality benchmark retains strict enforcement; the
  known failed profile must not be promoted by test infrastructure changes.

## T-RAG-05: coverage, CI and review

- Record line and branch coverage for the complete `rags` package. Do not omit
  difficult production modules or add coverage exclusions to reach a target.
- Ordinary unit+integration acceptance floors: line coverage at least 80%,
  branch coverage at least 72%. Baseline percentages above were measured before
  fixes. Do not lower these floors after inspecting new results.
- Publish per-file coverage, XML/JSON/HTML coverage artifacts, JUnit reports,
  inventory, real E2E reports and a before/after review. Label parent-only
  coverage separately from subprocess/optional-engine execution evidence.
- Run lint and relevant regression tests after corrections; retain actual
  external gates in CI and verify the final published service revision.
- Document every concrete code-smell finding, correction, verification and
  remaining limitation. Commit this specification before test/refactor work.
