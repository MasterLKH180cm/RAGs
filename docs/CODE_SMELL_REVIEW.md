# Whole-repository code-smell review and README improvement

Reviewed on **2026-11-26** against baseline
`6b1f97357ce934a75bfae3102ef8e72512d3da16`.
The [specification](MAINTAINABILITY_REVIEW_SPEC.md) was committed before implementation.
This report covers the current RAGs repository; downloaded model weights and
upstream dependency implementations are outside its scope.

## Outcome

The review found **18 confirmed code defects: 17 P2 and one P3**. All are fixed
in this change, with regressions for the affected behavior. No P0 or P1 finding
was confirmed. The absence of a confirmed finding is not a proof that a path
cannot contain other defects.

The entry README now starts with interface selection, one Python environment,
an authenticated local service, and a first collection/document answer. Detailed
model, deployment, authentication, migration and historical validation material
is preserved in [README_REFERENCE.md](README_REFERENCE.md). The first quick-start
heading moved from line 370 to line 38, and the entry page is about 300 lines
instead of 1,464. These counts describe navigation, not a code-quality score.

The review inspected every production Python module, the shipped admin assets,
executable scripts, templates, deployment files and CI/test-contract
infrastructure. Associated tests were inspected around the relevant contracts;
this does not claim a separate manual audit of every line of every test fixture.
The complete file inventory is below.

Severity describes the demonstrated consequence: **P2** is an incorrect behavior,
bounded-resource failure, concurrency fault or meaningful contract violation;
**P3** is lower-impact operational inconsistency. Architectural observations
without a demonstrated failing behavior are recorded separately.

## Confirmed findings and fixes

### CR-01 · P2 · OAuth introspection had incomplete network bounds

**Location:** [`auth._introspect`](../rags/auth.py) and
[`provider_transport`](../rags/provider_transport.py).

The synchronous response path applied a per-read timeout, rather than a complete
request deadline. A response delivering chunks within that timeout completed
after approximately 1.35 seconds with a configured one-second limit. Decoded
iteration also allowed decompression before enforcing the response-size limit.

The OAuth form request now uses the same owned asynchronous transport and full
deadline as model requests, with explicit request/response byte ceilings,
identity encoding, no redirects and `trust_env=False`. A bounded form facade
preserves Basic authentication and form encoding. Clients and response streams
are closed before returning, including calls originating inside an event loop.
Malformed provider responses map to the existing sanitized service error.

Regression evidence: [`test_auth_unit.py`](../tests/test_auth_unit.py) exercises
slow headers/chunks, compression and oversized envelopes without consuming the
body, malformed JSON and cleanup; [`test_provider_transport.py`](../tests/test_provider_transport.py)
covers form encoding, bounds and the existing JSON facade. The JSON facade still
forces its original content type, and ordinary calls preserve authentication
configured by a client factory.

### CR-02 · P2 · Non-ASCII CSRF input raised an internal error

**Location:** [`admin.install_admin`](../rags/admin.py), CSRF validation.

An authenticated request containing a non-ASCII CSRF header reached
`secrets.compare_digest` with an unsupported string and returned HTTP 500.
Validation now rejects that input before comparison with HTTP 403. The
regression verifies no document mutation and continued usability of the valid
session: [`test_admin.py`](../tests/test_admin.py).

### CR-03 · P2 · Approval and recovery accepted coerced decisions or revisions

**Location:** [`agents.api`](../rags/agents/api.py),
[`AgentEngine.resume`](../rags/agents/engine.py) and
[`Workflow.resume` and `recover`](../rags/agents/workflow.py).

The HTTP model converted `1`, `"yes"`, `"true"` and `"on"` into boolean approval,
and accepted numeric strings as revisions. Direct SDK recovery also compared a
supplied value with a stored integer before passing the stored revision onward;
Python equality therefore let `True` or `1.0` match revision `1`.

Approval/recovery envelopes now use strict, forbid-extra models. SDK entrypoints
require an exact nonnegative integer revision. Invalid requests cannot consume a
checkpoint or execute a callback. This was an input-contract defect, not a
demonstrated tenant authorization bypass. Actual HTTP and direct-SDK regressions
are in [`test_agent_api_integration.py`](../tests/test_agent_api_integration.py),
[`test_agent_engine_integration.py`](../tests/test_agent_engine_integration.py),
[`test_agent_workflow_api.py`](../tests/test_agent_workflow_api.py) and
[`test_agent_workflow_integration.py`](../tests/test_agent_workflow_integration.py).

### CR-04 · P2 · Valid model output could strand a durable agent run

**Location:** [`AgentEngine._step`](../rags/agents/engine.py),
[`StateStore`](../rags/agents/state.py) and the workflow transition boundary.

The actual router accepted a 293,112-byte reply with eight permitted calls and
12,023-byte argument strings. Each individual limit passed, but duplicating
transcript and pending-call data exceeded the 256 KiB checkpoint cap. The
exception left a durable `running`, revision-0 record without a supported resume
path. Approval feedback and a tool result could cause the same budget mismatch.

Checkpoint encoding now has a distinct `CheckpointBudgetExceeded` exception.
All agent transitions reserve space for a terminal reason and stop from the last
valid state if the next state cannot fit. Uncheckpointed calls do not execute;
pending calls do not continue after a budget stop. Workflow approval-wait writes
are inside its handled execution boundary, and a shared `StateStore.terminate`
persists terminal status without truncating saved evidence.

Denial of an exact-cap waiting workflow uses the same terminal path: replacing
`approved: null` with `false` would otherwise require one extra byte. The
original caller revision is validated before fallback, and a concurrent update
still fails compare-and-swap rather than being overwritten.

An authorized tool handler can finish before its oversized result is rejected.
Stopping subsequent pending actions does not roll back an external write.

An exact-limit legacy record may not fit even one diagnostic field. In that case,
the terminal `stopped`/`failed` status is saved with the old JSON unchanged, so
`stop_reason`, an error field, or the updated `approved` value can be absent.
Only a size exception uses this
fallback; CAS, definition and storage errors retain their existing behavior.
Regressions use the actual router parser, approval feedback, tool side-effect
counts, full legacy records and workflow failure paths in the agent/workflow
integration tests linked above. The cap was not increased.

### CR-05 · P2 · Replay-safety declarations accepted truthy non-booleans

**Location:** [`ToolSpec`](../rags/agents/engine.py),
[`Workflow.then` and `parallel`](../rags/agents/workflow.py).

For example, `idempotent="false"` was treated as a positive replay-safety
declaration. Approval/idempotency flags now require exact booleans, consistently
with the existing branch contract. Parallel and approved actions still require
explicit replay safety. Tests cover strings, integers and `None` in
[`test_agent_contracts_unit.py`](../tests/test_agent_contracts_unit.py) and the
workflow integration suite. These are trusted operator SDK declarations; HTTP
requests cannot register arbitrary Python callbacks.

### CR-06 · P2 · Workflow context aliases rewrote earlier results

**Location:** [`Workflow.start`, `_context` and parallel execution](../rags/agents/workflow.py).

The frozen context dataclass contained shallow-copied dictionaries. Mutating a
nested prior output or input changed the next checkpoint and the caller's input.
Parallel callbacks also shared a mutable context.

Inputs and per-node contexts now use independent deep snapshots. Each parallel
branch receives its own snapshot; only returned callback output is committed as
new output. Sequential and event-ordered parallel regressions prove prior-state,
caller-input and sibling isolation in the workflow integration suite. Callbacks
are still trusted code, not sandboxed execution.

### CR-07 · P2 · Finite extreme vectors produced invalid cosine scores

**Location:** [`native.cosine`](../rags/native.py).

Identical vectors `[1e200, 0]` produced `NaN`, while `[1e-200, 0]` produced zero.
Overflow and underflow could therefore misorder actual SQLite retrieval results.
Both vectors are independently scaled before computing the dot product and
norms; the result is clamped to the mathematical cosine range. Zero-vector
semantics remain unchanged. This stabilizes the existing formula rather than
changing the ranking policy.

Evidence: [`test_native_units.py`](../tests/test_native_units.py) covers large,
small, opposite and zero vectors, while
[`test_native_integration.py`](../tests/test_native_integration.py) verifies
ingestion and dense ranking through the SQLite store.

### CR-08 · P2 · Malformed Anthropic completion envelopes escaped validation

**Location:** [`providers.response_text`](../rags/providers.py).

An object/list `stop_reason` raised `TypeError`; a text-plus-tool-action response
could be mistaken for a completed answer and ingested as an image caption.
The parser now validates the completion-reason type and permitted text/thinking
block forms. Tool actions and unknown blocks fail with sanitized
`invalid_model_output` instead of being accepted as a complete answer.

Provider tests and actual `/images` route tests prove HTTP 502 and no index
mutation: [`test_providers.py`](../tests/test_providers.py) and
[`test_vision_providers.py`](../tests/test_vision_providers.py).

### CR-09 · P2 · Image validation changed global policy and leaked bomb errors

**Location:** [`vision.validate_image`](../rags/vision.py).

Every validation changed Pillow's process-global pixel limit. A tiny PNG with a
large declared dimension raised an uncaught `DecompressionBombError` and returned
HTTP 500. The global assignment is removed; the explicit local pixel ceiling is
retained. Both Pillow bomb warnings and errors map to the existing invalid-image
response. Actual upload regressions exercise local, warning and error paths
without allocating huge pixel buffers, and verify no database write.

### CR-10 · P2 · Federation lost request deadlines and provider usage

**Location:** [`RAGService._federated`](../rags/retrieval.py) and usage tracking in
[`providers`](../rags/providers.py).

Executor threads did not inherit `ContextVar` state. With 0.5 seconds remaining,
local provider work received a fresh 30-second budget, remote work received ten
seconds, and local embedding usage was omitted from the response.

Each worker now receives an independent copied context. Remote work also uses
the remaining deadline. A request-scoped, locked usage accumulator permits
concurrent workers to append safely without mixing independent requests.
[`test_federated_contexts.py`](../tests/test_federated_contexts.py) uses actual
SQLite retrieval and concurrent requests to verify timeout propagation, usage
separation and rejection before remote I/O after budget exhaustion.

### CR-11 · P2 · Concurrent cold requests loaded the same encoder twice

**Location:** [`providers._semantic_encoder`](../rags/providers.py).

`lru_cache` protects cache bookkeeping but does not serialize a cache-miss
factory. Concurrent requests constructed two complete encoders while retaining
only one. A lock now covers cache lookup and construction. Cache inspection and
clearing seams remain available. Controlled concurrency tests establish one
constructor call and one shared instance; a failed construction releases the
lock so a later request can retry. This establishes lifecycle behavior, not a
new inference-throughput benchmark.

### CR-12 · P2 · CLI file limits checked a different file state from the read

**Location:** [`cli`](../rags/cli.py), [`collection_cli`](../rags/collection_cli.py)
and new [`file_io`](../rags/file_io.py).

Environment, goal and question files used `stat` followed by an unbounded
`read_text`. A growing environment fixture passed the initial size check and
installed a 131,072-character value despite a 65,536-byte limit.

A shared helper opens once, checks that descriptor for a regular file and size,
reads at most `limit + 1`, and checks again before returning strict UTF-8 text.
Supported platforms use `O_NOFOLLOW` for the leaf and `O_NONBLOCK` to reject a
substituted FIFO without waiting. This is not a sandbox against replacement of
parent directories, and platforms without those flags cannot promise the same
leaf-symlink protection. Existing collection CLI error mapping is retained and
environment updates occur only after successful parsing.

Regression evidence: [`test_file_io.py`](../tests/test_file_io.py),
[`test_cli.py`](../tests/test_cli.py) and
[`test_collection_cli.py`](../tests/test_collection_cli.py).

### CR-13 · P2 · Self-RAG preparation could replace concurrently created files

**Location:** [`scripts/selfrag_local.py`](../scripts/selfrag_local.py).

An existence check followed by a download and `os.replace` overwrote a target
created during the download. Multiple preparers also shared fixed conversion
partial paths. A tiny fixture reproduced the overwrite without model weights.

Publication now uses no-clobber hard links. A nonblocking OS advisory lock owns
one cache's preparation, and an existing mismatched final model is rejected
before conversion. Exceptions clean owned partial files; process exit releases
lock ownership. The lock file remains intentionally, because unlinking it could
let processes lock different inodes. Its presence is not evidence of an active
or stale owner.

Tests cover concurrent target publication, preparation rejection, exception
cleanup and a real child-process crash:
[`test_selfrag_local.py`](../tests/test_selfrag_local.py) and
[`test_selfrag_setup_integration.py`](../tests/test_selfrag_setup_integration.py).
Pinned weights, converters, runtime digests and conversion parameters are
unchanged. No new actual-model quality result is inferred from these fixtures.

### CR-14 · P2 · Older admin responses replaced the latest document page

**Location:** [`admin_static/app.js`](../rags/admin_static/app.js), `loadDocuments`.

The identity epoch rejected another tenant's response, but two requests within
one session could complete out of order. A late old page or error replaced newer
results. A monotonically increasing request identity now guards both success
and error rendering, alongside the tenant epoch.
[`scripts/test_admin_ui.mjs`](../scripts/test_admin_ui.mjs) reproduces both older
successful and failed responses arriving after the current page. Existing
identity-switch and DOM-reset checks still run.

### CR-15 · P2 · Telemetry export buffered unbounded collector responses

**Location:** [`Monitor._post`](../rags/monitoring.py).

The exporter eagerly buffered/decompressed replies and used per-read timeouts.
Two slow-drip replies took approximately 6.25 seconds with a nominal two-second
timeout. Export now uses the shared bounded JSON transport with a **two-second
deadline per HTTP request**, a 16 KiB response cap and `trust_env=False`.
This is not a two-second bound for an entire flush containing multiple requests.
Failure accounting remains best-effort and does not expose payloads or upstream
error text. [`test_monitoring.py`](../tests/test_monitoring.py) verifies header
rejection before decompression, unknown-length body bounds and cancellation.

### CR-16 · P3 · Runtime version values drifted between surfaces

**Location:** [`monitoring`](../rags/monitoring.py), [`api`](../rags/api.py) and
[`cli`](../rags/cli.py).

Telemetry advertised `0.7.0` while the installed distribution and other surfaces
were `0.9.0`. API metadata, CLI version and telemetry now read the installed
distribution's version, avoiding another independent literal. The package
version and dependency lock are unchanged; the documented installation workflow
provides distribution metadata. The telemetry payload regression asserts that
reported version against the installed distribution.

### CR-17 · P2 · Ambient llama authentication broke the Inkling launch contract

**Location:** [`scripts/inkling_local.py`](../scripts/inkling_local.py).

The no-key local profile inherited `LLAMA_API_KEY`, causing its server to require
a credential its generated client configuration did not send. The launcher now
removes that ambient variable together with `LLAMA_ARG_*`, consistently with the
Qwen launcher. Explicit launch configuration and pinned artifacts remain intact.
The launch-environment regression is in
[`test_inkling_local_setup.py`](../tests/test_inkling_local_setup.py).

### CR-18 · P2 · Copied JSON validators disagreed about nonfinite numbers

**Location:** [`client`](../rags/client.py),
[`collection_cli`](../rags/collection_cli.py),
[`collection_api`](../rags/collection_api.py) and OAuth parsing.

The SDK and collection parser copies rejected duplicate keys and literal `NaN`,
but missed exponent overflow such as `1e999`. Baseline OAuth used plain
`json.loads`; it now adopts the same strict decoder. The HTTP SDK accepted that nonfinite response. The CLI
could issue a collection lookup before rejecting malformed input in a later
phase. The existing pure [`model_profiles.strict_json`](../rags/model_profiles.py)
already enforced finite floats and is now reused by all four boundaries. The
CLI no longer imports a private JSON helper from the SDK.

Regressions cover duplicate keys, constants, exponent overflow, decoding errors
and oversized/deep input at the relevant boundaries. The covered records-file
path rejects malformed/nonfinite JSON before collection lookup. Other CLI paths
can perform a read-only collection lookup before validating remaining input;
this change does not make every command's entire validation happen before I/O.
Parser recursion behavior differs by Python runtime;
the deep-input test establishes a sanitized rejection, not a claim that every
runtime raises `RecursionError` at the same nesting depth.

## README and documentation corrections

| Finding | Change | Verification |
| --- | --- | --- |
| The first usable path was buried beneath implementation history. | A short interface chooser, isolated Python install, offline service, SDK example and document example now precede optional models. | Marked examples run from the actual README against a temporary authenticated service. |
| The Compose live-check command omitted its required report argument. | The detailed reference supplies `--report reports/testing/compose-smoke.json`. | The old argument combination failed with exit code 2; the corrected invocation matches the CLI. Actual deployment remains a dedicated CI gate. |
| Moved anchors and configuration inventories could drift. | Updated inbound links, collection vector fields, the `router` provider option and feature/limits/generation authorization reference. Clarified that the standalone and companion Compose environment maps are maintained independently. | A repository Markdown path/anchor verifier plus review against current configuration and schema. |

The README verifier also proves unauthenticated rejection, offline model identity,
expected answers and citations, record counts and restart persistence. It runs
from the wheel's interpreter in CI, so source-tree import success alone cannot
hide a missing package file. It does not download models or equate offline
fixtures with semantic-model performance.

## Architectural observations deliberately retained

These are concrete future change/scaling concerns, rather than unresolved
instances of the defects above.

1. **The retrieval dispatcher coordinates several strategy contracts.**
   `RAGService._retrieve` combines strategy selection, ranking, model loops,
   aggregation and provenance. A new strategy also touches feature and registry
   metadata. Extract a cohesive handler when extending that strategy, with the
   existing tenant/evidence tests as contracts. A blanket dispatcher rewrite
   would obscure which semantics changed during this repair.

2. **Authentication, route capabilities and request models have parallel maps.**
   Collection routing, SDK dispatch, feature scopes and request models must evolve
   together. The review did not demonstrate a current authorization-order or
   route-permission discrepancy. A future shared contract table should preserve
   unknown-route rejection, feature policy and generated API documentation.

3. **SQLite storage classes duplicate connection lifecycle code.**
   Document and asset stores separately manage WAL, busy timeout, commit, rollback
   and close. They have different schema/foreign-key responsibilities. A future
   connection-policy change should cover both; an extraction must retain those
   differences explicitly.

4. **SQLite and PostgreSQL have different scaling paths.** Some SQLite retrieval
   paths materialize historical tenant candidates before Python filtering;
   PostgreSQL pushes selected revisions or allowed IDs into queries. Measure
   document/revision growth before optimizing these paths. Exact vector scans,
   immutable fork payload sharing and O(record-count) mapping copies remain the
   documented contracts; no ANN or Chroma Cloud throughput equivalence is claimed.

5. **Optional caches bound entries rather than resident bytes.** In particular,
   the late-interaction cache's vector matrices vary in size. A memory-budgeted
   eviction policy would make its operational ceiling clearer. The encoder
   construction lock fixes duplicated initialization but does not impose an RSS
   limit or establish a throughput improvement.

6. **Operator callbacks remain trusted and cooperative.** Late read-only tool
   results can be discarded, but Python threads cannot safely kill arbitrary
   writes. Workflow parallel barriers wait for registered callbacks. Operators
   still need downstream I/O deadlines and truthful idempotency declarations.
   Deep snapshots protect checkpoint ownership, not against malicious callbacks.

7. **The research subprocess is a lifecycle boundary, not a complete OS sandbox.**
   Request-local indexes, bounded protocol output and process-group cleanup are
   intentional. Container/host CPU and memory controls remain operational work.
   Process-global upstream registries, telemetry switches and native libraries
   are contained by the worker execution model; direct in-process integrations
   should not claim equivalent multi-tenant isolation.

8. **Broad exception handling is justified at public/dependency boundaries.**
   Provider, optional-model, CLI and research boundaries intentionally sanitize
   credential-bearing or implementation-specific failures. Narrowing every broad
   handler would weaken containment. New checkpoint fallback catches only its
   specific size error so that consistency failures still propagate.

9. **Memory expiration and process-local control state have explicit limits.**
   Observation TTL excludes expired facts from retrieval; it does not physically
   purge storage. Conversation/checkpoint retention remains an operator concern.
   Admin sessions and monitoring are process-local, and export is best-effort;
   this change does not provide a distributed control plane.

10. **The test HTTP guard is narrower than a network sandbox.** It intercepts
    HTTPX requests to nonlocal requested URLs. Other libraries, child processes
    and ambient proxy handling are outside that guard. Explicit resource markers,
    real-resource jobs and `--require-no-skips` remain necessary. Local real-HTTP
    checks remove ambient proxy variables; production proxy behavior is unchanged.

11. **Import and local-interface boundaries are deliberate.** The ASGI module
    creates its application for server entry, while importing `rags.client` stays
    free of app/database setup. `PersistentClient` is a local file-owner interface;
    authenticated HTTP is the boundary for untrusted users. `doctor` performs
    bounded read-only checks, not full database integrity or live-model certification.

12. **Evidence types remain distinct.** Actual encoders, research upstreams,
    deterministic generation fixtures, local wire-contract tests, historical
    Qwen admission measurements and live inference benchmarks answer different
    questions. The frozen Qwen answerability result remains **7/11 quality gates
    passed**, unpromoted and not enabled by default. This review did not alter
    profiles, datasets, recorded quality results, thresholds or promotion status.

## Compatibility and validation

Public routes, tenant/owner ordering, exact source references, storage schemas,
provider/model identities, router/agent/memory defaults and coverage/resource
gates are retained. Intentional corrections reject malformed approval/revision
types and nonfinite JSON, isolate callback mutations, stop oversized transitions,
bound identity-provider/collector replies and refuse unsafe setup overwrites.

Validation results and a source SHA-256 manifest are recorded in
[`testing-results-maintainability.json`](testing-results-maintainability.json).
That file distinguishes local working-source evidence from independent GitHub
CI on the published commit. Complete-package coverage includes all production
Python modules; subprocess/browser/container checks are reported separately.
The floors remain **80% lines and 72% branches**, with no new exclusions.

### Completed local validation

| Check | Result |
| --- | --- |
| Unit contracts | 1,699 passed; zero selected skips |
| Integration contracts without optional external resources | 771 passed; zero selected skips |
| Actual PostgreSQL/pgvector contracts | 55 passed; zero selected skips |
| Additional SQLite HTTP/CLI process and restart contracts | 3 passed; zero selected skips |
| Unique passing cases across these selections | **2,528** |
| Complete-package coverage, all 64 production Python modules | **86.60% lines; 80.53% branches** |
| Isolated wheel import and marked README onboarding | Passed, including authentication, citations and restart persistence |
| Repository Markdown paths/anchors | 48 files; 381 internal links checked |
| Ruff and admin JavaScript checks | Passed; includes same-session and cross-tenant stale-response regressions |
| Frozen profiles, datasets, recorded evidence, dependency and gate contracts | 61 tracked paths unchanged from baseline |

The remaining 25 collected resource-dependent tests are explicitly accounted for
in the JSON inventory and run in dedicated CI environments. They are deselected
from the local core run, not recorded as successful tests or selected skips.
One existing Starlette/httpx deprecation warning remains.

Core coverage is reproduced using the same ordered selections as CI:

```bash
python -m pytest -q -m unit --require-no-skips \
  --cov=rags --cov-branch --cov-config=pyproject.toml \
  --cov-report=json:reports/testing/coverage-unit.json
python -m pytest -q -m "integration and not external" --require-no-skips \
  --cov=rags --cov-branch --cov-config=pyproject.toml --cov-append \
  --cov-report=json:reports/testing/coverage-core.json
# Provision the PostgreSQL test DSNs and vector extension first, as CI does.
python -m pytest -q -m postgres --require-no-skips \
  --cov=rags --cov-branch --cov-config=pyproject.toml --cov-append \
  --cov-report=json:reports/testing/coverage-core.json
python scripts/check_coverage.py --report reports/testing/coverage-core.json
```

The README verifier runs the marked offline examples and checks repository links.
It does not validate external website availability, repeat the network Git
installation command, or preserve unknown external bookmarks to removed README
headings. Package installation is checked with an isolated wheel; real resources
have their own CI gates. The review does not claim a new model-performance measurement.

## File coverage inventory

The following inventory is generated from the reviewed working tree. Every
production module and executable script is listed explicitly; data fixtures,
model weights and dependency source are not included in the source inventory.

### Production Python modules (64)

```text
rags/__init__.py
rags/__main__.py
rags/admin.py
rags/admission.py
rags/admission_eval.py
rags/agents/__init__.py
rags/agents/api.py
rags/agents/config.py
rags/agents/deadline.py
rags/agents/engine.py
rags/agents/memory.py
rags/agents/router.py
rags/agents/state.py
rags/agents/templates.py
rags/agents/workflow.py
rags/answerability.py
rags/api.py
rags/auth.py
rags/backends.py
rags/benchmark.py
rags/chunking.py
rags/cli.py
rags/client.py
rags/collection_api.py
rags/collection_cli.py
rags/collection_filters.py
rags/collection_models.py
rags/collection_search.py
rags/collection_service.py
rags/collection_store.py
rags/config.py
rags/evaluate.py
rags/features.py
rags/file_io.py
rags/limits.py
rags/mcp_server.py
rags/migrate.py
rags/model_benchmark.py
rags/model_profiles.py
rags/monitoring.py
rags/multimodal.py
rags/native.py
rags/postgres.py
rags/provider_transport.py
rags/providers.py
rags/registry.py
rags/research/__init__.py
rags/research/common.py
rags/research/crag_adapter.py
rags/research/crag_web.py
rags/research/graphrag_adapter.py
rags/research/hipporag_adapter.py
rags/research/lightrag_adapter.py
rags/research/raptor_adapter.py
rags/research/selfrag_adapter.py
rags/research/selfrag_runtime.py
rags/research_protocol.py
rags/research_service.py
rags/research_worker.py
rags/retrieval.py
rags/storage.py
rags/tokenization.py
rags/verify_admission.py
rags/vision.py
```

### Administrative assets (3)

```text
rags/admin_static/app.js
rags/admin_static/index.html
rags/admin_static/style.css
```

### Executable scripts (19)

```text
scripts/benchmark_collections.py
scripts/bootstrap_compose.py
scripts/check_compose.py
scripts/check_coverage.py
scripts/check_readme.py
scripts/crag_local.py
scripts/e2e_agent_platform.py
scripts/e2e_crag_v10.py
scripts/e2e_research_v03.py
scripts/e2e_selfrag_v10.py
scripts/inkling_local.py
scripts/install-answerability.sh
scripts/install_research.sh
scripts/local_model.py
scripts/migrate_sqlite_to_postgres.py
scripts/qwen_local.py
scripts/selfrag_local.py
scripts/test_admin_ui.mjs
scripts/verify_admission_runtime.py
```

### CLI templates (8)

```text
rags/templates/anthropic.env
rags/templates/inkling-local.env
rags/templates/inkling.env
rags/templates/litellm.env
rags/templates/offline.env
rags/templates/ollama.env
rags/templates/openai.env
rags/templates/postgres.env
```

### Deployment, packaging and test-contract infrastructure (14)

```text
.github/workflows/ci.yml
.github/workflows/model-benchmark.yml
.env.example
.dockerignore
.gitignore
.python-version
AGENTS.md
Dockerfile
compose.yaml
conftest.py
docker/init-databases.sh
docker/otel-collector.yaml
pyproject.toml
uv.lock
```

### Optional runtime dependency specifications (10)

```text
requirements/answerability.txt
requirements/research-crag.in
requirements/research-crag.txt
requirements/research-full_graphrag.in
requirements/research-full_graphrag.txt
requirements/research-hipporag.in
requirements/research-hipporag.txt
requirements/research-lightrag.txt
requirements/research-raptor.txt
requirements/research-selfrag-convert.txt
```
