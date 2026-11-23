# v0.9 validation review

## Completed upgrade

RAGs v0.9 adds explicit Inkling/Tinker request semantics, independent embedding
configuration, conservative UTF-8 byte budgeting, strict local JSON validation,
and bounded transport for both direct and routed providers. Generation,
embedding and vision replies are streamed under response-size and elapsed-time
limits. Truncated replies cannot execute tools or satisfy a generation contract.

Tinker generation requires an explicit `RAGS_GENERATION_API_KEY` or `_FILE`.
It never inherits ambient OpenAI/Tinker credentials. Its default endpoint is the
canonical Tinker sampling API; independent embedding endpoints and credentials
retain their existing behavior. The separately configured model benchmark and
Agentic preset use dedicated `TINKER_API_KEY(_FILE)` credentials.

Agent checkpoints retain bounded private reasoning only for tool continuation;
public history and exports omit it. The approval fingerprint binds execution
contracts, schemas, permissions, model mapping and limits. Existing completed,
failed and stopped v0.8 history remains readable by its owner. **Pending v0.8
agent approvals cannot resume under the new contract**; explicitly start a new
run. This is an intentional approval invalidation, not a seamless pending-run
migration. The admin bootstrap/sign-in race has a deterministic DOM regression.

## Local verification

The machine-readable record is [testing-results-v09.json](testing-results-v09.json).

| Layer | Result |
| --- | ---: |
| Unit tests | 777 passed |
| Integration tests, including actual PostgreSQL | 424 passed |
| Total executed pytest cases | 1201 passed |
| PostgreSQL cases included above | 27 passed |
| Explicitly deselected encoder/research resource cases | 5 |
| Selected skips, errors and failures | 0 |
| Complete-package line coverage | 84.63% |
| Complete-package branch coverage | 76.95% |
| Existing line/branch gates | 80% / 72%, passed |

Coverage includes all 49 production Python modules, with no new exclusions
or lowered thresholds. The full suite runs under a test-only real-HTTP transport
guard: ordinary tests may use injected transports or literal loopback servers.
Explicit resource tests retain their provisioned external dependencies.

Both Ruff gates, JavaScript syntax/DOM regressions, whitespace checks and the
locked dependency check passed. The final 0.9 wheel was installed separately and
verified with Python isolated mode so the source checkout cannot mask missing
package contents. The wheel includes the Inkling profile, bounded transport,
model profiles, benchmark module and dashboard assets.

Independent API processes passed **6/6 acceptance checks on SQLite** and **6/6
on PostgreSQL**: canonical Inkling wire fields; actual knowledge-search/citation
round trip; refusal of truncated tools; tenant isolation; restart/approval/CAS
persistence; and admin cookie/CSRF behavior. These use an explicitly labelled
local model fixture and synthetic policy data. Browser mode adds real Chromium
sign-in, an agent task, visual workflow approval and server-side logout checks.
Local Chromium is blocked by this sandbox's Unix-socket policy; actual browser
execution is an independent required CI job, never counted as a local pass.

The synthetic PostgreSQL 17.12/pgvector 0.8.7 read-path measurement used 500
documents, eight chunks per document and 30 measured samples after three warmups.
Scoped revision-join p95 was 1.977 ms and scoped exact cosine p95 was 1.696 ms.
These are single-connection query measurements, not generation latency or
production capacity. They do not constitute a before/after release comparison.

## Model evidence and quality boundaries

The [Inkling readiness report](../reports/inkling-mini-readiness-v09.json) is
**blocked / missing_model_credential**, with zero provider requests. An
authorized [remote Actions attempt](https://github.com/MasterLKH180cm/RAGs/actions/runs/45103025522)
returned the same blocked result. Its [archived report](../reports/inkling-mini-readiness-actions-v09.json)
was verified against the downloaded artifact digest and matches the local
adapter, harness, dataset and control hashes. No live
Inkling latency, throughput, quality or cost result is claimed. The production
router benchmark, its six synthetic cases, strict comparator and manually
triggered Actions workflow are documented in
[INKLING_BENCHMARK.md](INKLING_BENCHMARK.md). The default paid run is bounded to
42 requests including warmups; a missing credential produces an explicit
blocked report and exit status 2. Ordinary pull-request CI makes no paid calls.

The unchanged frozen Qwen admission report retains **21/21 correctness checks**
and **7/11 quality targets**; four quality targets still fail and profile
promotion remains failed. That report is recorded evidence, not a fresh full
quality run for this release. The profile stays unapproved/disabled by default.
The real pinned-model runtime and research/encoder jobs remain in CI. A new
admission scoring/retrieval change or proposed promotion must run the complete
frozen benchmark with quality enforcement.

Exact-revision GitHub Actions checks provide the separate browser, Docker,
encoder, research-upstream and admission-runtime results. Their counts are not
merged into the local pytest or synthetic model-contract totals above.
