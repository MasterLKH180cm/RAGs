# Self-hosted Inkling implementation and validation review

The self-hosted follow-up adds an explicit local Inkling path through direct
generation, the agent router and the model benchmark. The final complete core
and actual PostgreSQL run passed **1,333 tests**, with zero failures, errors or
skips. Actual Inkling inference remains blocked by the available memory of the
execution host. The
[machine-readable record](testing-results-inkling-local.json) binds these
results to the production source and generated evidence hashes.

This review follows [SELF_HOSTED_INKLING_SPEC.md](SELF_HOSTED_INKLING_SPEC.md).
The original [v0.9 validation](VALIDATION_V09.md), its
[test result record](testing-results-v09.json), and both hosted Tinker readiness
reports remain historical evidence. A missing hosted credential does not impose
a Tinker-key requirement on a self-hosted deployment.

## Implemented contracts

| Interface | Local configuration | Authentication and request behavior |
| --- | --- | --- |
| Direct generation | `RAGS_GENERATION_AUTH_MODE=none`, explicit generation URL and `openai` protocol | No Authorization header; no unrelated hosted secret reads; ambient HTTP proxies bypassed |
| Agent router | `RAGS_ROUTER_AUTH_MODE=none`, explicit gateway URL and approved aliases | Every alias explicitly selects `openai` with unset reasoning effort; no Authorization header or ambient proxy use |
| Model benchmark | `--preset inkling-local` | Loopback OpenAI wire semantics, `max_tokens`, unset effort, and no credential lookup by default |
| Authenticated local benchmark | Local preset with `--auth-mode required` | Dedicated `INKLING_LOCAL_API_KEY(_FILE)` by default; an explicit `--key-env` selects another credential |

Both production authentication modes default to `required`. Explicit `none`
accepts only clean literal loopback IP URLs; DNS names, private LAN addresses,
IPv6 mapped/scoped forms, URL credentials, queries and fragments are rejected.
An explicit generation/router credential or secret-file setting conflicts with
`none` and fails validation before the file is read. Router plaintext HTTP keeps
its development/test opt-in. A container's loopback belongs to that container;
a host or service endpoint uses an explicit reachable URL and required
authentication. These boundaries are exercised in
[test_local_model_auth.py](../tests/test_local_model_auth.py).

The packaged [`inkling-local` profile](../rags/templates/inkling-local.env)
selects `http://127.0.0.1:8080/v1` and
`thinkingmachines/Inkling-2-Mini` for direct generation and the router's
`inkling` alias. Direct generation uses a 4,096-unit conservative UTF-8-byte
context budget and reserves 1,024 output units. Embeddings retain their separate
provider and credentials. The local preset retains the default-on router,
agent and memory policy; doctor success establishes configuration readiness.
Runtime installation, model loading and hardware readiness are checked
separately by the [pinned deployment guide](INKLING_LOCAL.md).

The existing bounded transports, local JSON/tool validation, truncation
rejection, tenant authorization, tool permissions and approval rules remain
active. The hosted Tinker path retains its request contract and credential
selection. The benchmark requires a strict deployment manifest for local live
runs, including runtime/model revisions, artifact hash, quantization, context,
server slots, thread count and hardware. Manifest provenance is explicitly
operator supplied; an HTTP model label does not attest to loaded weights.
See [INKLING_BENCHMARK.md](INKLING_BENCHMARK.md) for the comparison contract.

## Pending approvals and upgrade compatibility

The agent execution fingerprint now includes the normalized gateway endpoint
and gateway authentication mode. **Pending or resumable runs created before
this patch have a different fingerprint and cannot resume under the new
contract**, including runs whose endpoint happens to remain unchanged. Start a
new run and review any requested approval again. This is an intentional approval
invalidation; no migration reuses an old approval for the changed contract.

Authorized owners can still read historical runs. Within the patched contract,
changing the gateway endpoint or authentication mode invalidates a pending
approval, while rotating the credential for the same endpoint and mode retains
compatibility. Regression tests also verify that incompatible resumes produce
no tool side effect in
[test_agent_engine_integration.py](../tests/test_agent_engine_integration.py).

## Final verification

The frozen run used Python 3.12.14 and a disposable local PostgreSQL process.
PostgreSQL **17.12** and pgvector **0.8.7** were queried from the executing
service before running the suite. The database cases executed against that
service rather than an injected database adapter.

| Selection or gate | Result |
| --- | ---: |
| Unit tests | 903 passed |
| Integration tests, including actual PostgreSQL | 430 passed |
| Total selected pytest cases | 1,333 passed |
| Ordinary core cases, excluding PostgreSQL | 1,306 passed |
| Actual PostgreSQL cases, included above | 27 passed |
| Explicit external resource deselections | 5 |
| Selected skips, failures and errors | 0 |
| Local preparation helper cases, included above | 18 passed |
| Model benchmark cases, included above | 142 passed |
| Local model authentication cases, included above | 45 passed |
| Complete-package line coverage | 84.9359% |
| Complete-package branch coverage | 77.4530% |
| Required line/branch floors | 80% / 72%, passed |

The PostgreSQL total comprises 23 storage tests and four agent-state tests.
The five deselections are the actual ColBERT, CLIP and BGE encoder tests and the
RAPTOR and LightRAG upstream tests. They are identified individually in the
machine-readable record. The 47.78-second pytest run emitted one existing
Starlette test-client deprecation warning.

Coverage includes all **49 production Python modules** without new omissions,
exclusions or lower thresholds. The separate coverage gate passed. Ruff
`E4/E7/E9/F`, Ruff `B/RUF021`, and `git diff --check` passed. Generated inventory,
JUnit, coverage and service-version artifacts live under the ignored
`reports/testing-self-hosted/` directory; their SHA-256 hashes are retained in
the committed result record. Counts for wheel installation, independent process
acceptance, Chromium, Docker and external model/runtime gates are outside this
local pytest total.

## Review findings corrected before the final run

1. **Local prompt budget:** the profile initially inherited the 16,000-unit
   direct-generation default while the pinned CPU server uses a 4,096-token
   context. The profile now sets `RAGS_CONTEXT_BUDGET=4096`, with an assertion in
   the profile/doctor test. The router retains its own transport bounds; server
   context rejection remains an error rather than a successful model result.
2. **Credential selection:** an authenticated override of the local benchmark
   preset initially inherited the hosted Tinker key name. Local authenticated
   runs now default to `INKLING_LOCAL_API_KEY(_FILE)`, explicit `--key-env`
   overrides remain supported, and the hosted default remains
   `TINKER_API_KEY(_FILE)`. The final 142-case benchmark suite includes the
   dedicated-key, explicit-key and no-lookup regressions.
3. **Download deadline:** a blocking buffered read could defer a total elapsed
   check during a continuously trickling download. The preparation helper now
   uses asynchronous HTTPX streaming under an `asyncio.timeout` deadline,
   validates explicit HTTPS redirects, enforces the pinned size/digest before
   publication and removes incomplete downloads. Its 18-case suite includes
   continuous-trickle cancellation and partial-file cleanup.

## Model and fixture evidence boundary

Actual loopback HTTP tests exercise direct generation, the vision transport
adapter, native router tool selection, local benchmark requests, poisoned proxy
environments, absent Authorization headers, tenant isolation, citations and
truncation rejection. Their responses and documents are synthetic fixtures.
These tests establish integration behavior; they do not measure actual Inkling
latency, throughput, quality or cost, or certify a runtime's model parser. The
pinned CPU recipe makes no vision-support claim.

The separate [local readiness report](../reports/inkling-local-readiness-v09.json)
records 10,451,427,328 bytes of total RAM and 6,828,683,264 bytes available at
inspection. The selected IQ4_NL artifact is 17,627,940,768 bytes. The pinned CPU
runtime was verified and launched for version/help inspection, but the weights
were not downloaded into this insufficient-memory host. Readiness is
`blocked` for `insufficient_available_memory` and `model_weights_missing`, with
zero provider requests and no model inference. The report is a hardware
preflight, and contains no performance or quality result.

No previous admission benchmark was rescored by this change. Its historical
quality limits remain in the original v0.9 review. Future local model results
must retain their exact artifact, runtime, quantization and hardware metadata;
fixture timings and quantized CPU results cannot be relabelled as native hosted
Inkling performance.

## Reproduce the local gates

Install the locked development, PostgreSQL, PDF, MCP and vision extras and
provision a disposable PostgreSQL database with pgvector. Set
`RAGS_TEST_POSTGRES_URL` for that database, then run from the checkout:

```bash
uv run --no-sync python -m pytest -q -m 'not external or postgres' \
  --require-no-skips \
  --test-inventory reports/testing-self-hosted/inventory-complete.json \
  --junitxml reports/testing-self-hosted/junit-complete.xml \
  --cov=rags --cov-branch \
  --cov-report=json:reports/testing-self-hosted/coverage-complete.json

uv run --no-sync python scripts/check_coverage.py \
  --report reports/testing-self-hosted/coverage-complete.json

uv run --no-sync ruff check --select E4,E7,E9,F \
  conftest.py rags tests scripts/inkling_local.py
uv run --no-sync ruff check --select B,RUF021 rags
git diff --check
```

Selected PostgreSQL tests fail if their service is unavailable; they do not
silently skip. Ordinary HTTP tests retain the test-only outbound transport
guard and use injected transports or literal loopback fixtures. The documented
external resource jobs remain separately provisioned gates.
