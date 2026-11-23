# RAGs architecture

This document describes the implemented service. The [v0.4 specification](SPEC_V04.md) adds explicit feature configuration and PostgreSQL/pgvector, extending the [v0.3 specification](SPEC_V03.md) and [v0.2 requirements](SPEC_V02.md); the [v0.3 review](IMPLEMENTATION_REVIEW_V03.md) and historical [v0.2 review](IMPLEMENTATION_REVIEW.md) distinguish actual execution, measured quality and remaining integrations. The [README](../README.md#architecture-and-flows) contains the service, ingestion and retrieval flowcharts.

The current review corrections follow the [data](REVIEW_DATA_FIX_SPEC.md) and
[runtime](REVIEW_RUNTIME_FIX_SPEC.md) contracts. The optional shared generation
gateway follows [LITELLM_SPEC.md](LITELLM_SPEC.md) and its
[deployment guide](LITELLM.md).
The [review report](REVIEW_LITELLM.md) maps these changes to regression evidence
and upgrade requirements; [local results](testing-results-review-litellm.json)
retain their tested source snapshot separately from published CI results.

## Service boundaries

FastAPI authenticates a bearer credential and derives one tenant before calling the document service or original-image service. The read-only stdio MCP bridge calls that HTTP API. The companion [agentic-system](https://github.com/MasterLKH180cm/agentic-system) has its own API, durable workflow store, and worker; the default full stack gives each service a separate PostgreSQL database and non-superuser application role. RAGs owns the native pgvector knowledge index; the workflow store does not contain that index.

Document ingestion and retrieval live in `RAGService`. `Providers` supplies hashing or trained single-vector embeddings and optional text generation. `BackendManager` supplies actual token-matrix late interaction or an explicitly configured external research protocol. Original image assets have a separate service and paired CLIP encoder; they do not masquerade as text chunks.

The optional LiteLLM deployment sits behind the existing typed router and
Agentic's separate model adapter. It has its own process/database, verified HTTPS
and distinct model-restricted service keys. It handles generation through the
immutable alias `gateway/inkling-2-mini-iq4nl-v1`; it owns no RAG source state,
tenant authorization, native-agent approval or conversation history. Its shipped
configuration disables proxy retries, fallback and cache. Default generation
remains extractive and embeddings remain local hashing. Selecting the optional
`litellm` starter does not authorize requests until gateway configuration is
complete; embeddings retain their separate provider and identity.

## Configuration and feature enforcement

`Settings` resolves operational environment configuration once per process. A
selected `RAGS_FEATURES_FILE` supplies a JSON boolean map; `RAGS_FEATURES_JSON`
overrides individual entries. The manifest in `rags/features.py` includes
interfaces, document and original-image operations, provider families, indexes,
chunkers, tokenizers and every retrieval strategy. Unknown keys, non-booleans
and invalid selected files fail startup. An enabled permission still requires
its provider, installed dependencies and compatible index profile.

```mermaid
flowchart TD
    File["Feature file"] --> Resolve["Resolve settings and boolean overrides"]
    Env["Process environment"] --> Resolve
    Request["HTTP or MCP request"] --> Auth["Authenticate and select tenant"]
    Resolve --> Guard["Check operation and required features"]
    Auth --> Guard
    Guard -->|Disabled| Deny["403 feature_disabled"]
    Guard -->|Allowed| Service["Validate profile and bounded operation"]
    Service --> Store["Tenant-scoped storage and providers"]
```

Guards run before optional provider calls or mutations, including caption upload
processing and internal tokenizer use. Original-image download is independently
controlled by `original_images`; internal authorized reindexing still has access
to its source bytes. The shipped MCP bridge uses authenticated capabilities for
its tool listing and rechecks permissions on calls; HTTP remains authoritative.
`/health` and `/capabilities` expose resolved backend metadata and permissions,
without database credentials or provider keys. Feature changes require process
recreation; API capability responses do not constitute model-quality evidence.

The default Compose service uses PostgreSQL 17 and pgvector 0.8.7. Its bootstrap
role creates the database, application role and extension; the runtime role
does not install extensions. Database ports remain private, and provider/feature
settings are explicitly forwarded to the application. `RAGS_DATABASE_URL`
selects PostgreSQL without fallback. With that selector unset, `RAGS_DB_PATH`
retains the explicit SQLite development and legacy-data path.

## Source records and derived indexes

| Storage | Responsibility |
| --- | --- |
| `documents` | Current document identity, source content, and index profile |
| `document_revisions` | Source snapshots, recording time, validity interval, and processing profile |
| `chunks` | Original leaf/parent spans, generated summaries, native PostgreSQL `vector` values, tokenizer metadata, and lineage |
| Sparse index | PostgreSQL GIN full-text index with `ts_rank_cd`; SQLite compatibility uses `chunks_fts` / FTS5 |
| `graph_relations` | Entity relations with original source-chunk evidence IDs |
| `audit` | Lifecycle events committed in the same application database |
| `image_assets` | Tenant-scoped original image bytes, native PostgreSQL `vector` values, revisions, and encoder identity |
| `agent_runs`, `agent_messages`, `agent_observations` | Native-agent checkpoints and execution fingerprints, owner-scoped conversation order, and evidence-linked observations with expiry |

PostgreSQL stores text and original-image embeddings as native `vector` values.
Dense ranking executes exact cosine distance with `<=>` in SQL after tenant,
version and metadata scoping. There is no fixed global vector dimension: the
profile preflight rejects incompatible model/dimension selections before a
query, while explicit reindexing can replace a model profile. No ANN index or
HNSW performance claim is implied. PostgreSQL sparse ranking uses `ts_rank_cd`,
which is not the same ranking formula as SQLite FTS5/BM25.

Document changes and their sparse, hierarchy, graph and audit records commit
atomically. Application SQL uses bound parameters. The SQLite compatibility
backend retains WAL, foreign keys and bounded busy waits. The audit table is an
operational record, not a separate tamper-evident audit system.

The explicit `python -m rags.migrate --sqlite SOURCE` importer reads a source
SQLite database without modifying it, requires an empty PostgreSQL target, and
copies current/historical text, derived records, original images and native-agent
state in one transaction without model calls. Pending checkpoints, fingerprints,
owner/session identity, message sequences and observation expiry are preserved
without approving or resuming runs. Stores with no native-agent tables remain
supported; partial native schemas, unsupported columns and unknown populated
application tables fail before target initialization. All application tables
participate in empty-target checks, locks and count verification. It is an operator-run migration, not an
implicit startup operation; backups and the rollback boundary are documented
in the [README](../README.md#database-migration-and-backups).

### Atomic ingestion and reindexing

Ingestion validates the source and processing configuration, resolves the tokenizer, and constructs exact source spans. The canonical profile includes algorithm/configuration, tokenizer identity, embedding model/resolved endpoint/dimensions, optional embedding-space version, active semantic/context identities, and prompt revision. The unchanged shortcut requires matching source and profile, so a tokenizer or model change cannot silently retain an incompatible index.

Router embedding identity comes from the active router's canonical gateway URL.
`RAGS_EMBEDDING_SPACE_ID` can version the deployment behind a stable endpoint and
alias; it never includes credentials. Endpoint/space changes require reindexing,
including old router profiles that stored only the marker `router`. Empty space
IDs preserve existing direct/offline profiles. Generation aliases used by context,
hierarchy and graph indexes must likewise retain immutable deployment meanings.

Provider calls and derived-index construction happen before the write transaction. A compare-and-swap check verifies that the expected source revision is still current, then commits the source snapshot and derived records atomically. Failed inference or a concurrent change preserves the previous index. Explicit reindexing requires the expected revision. Deleting a document purges its stored revisions and derived evidence as well as its current record.

Additive schema migration marks existing unqualified records as `legacy`; it does not invent offsets/token counts or call a model. Vector retrieval rejects an incompatible or unknown profile and requests reindexing. A source reprocessing operation preserves validity when the source has not changed; a source update creates a new revision.

### Version selection and evidence lineage

Authorization and version selection precede ranking. Within each document and
valid-from interval, reads first choose the newest recorded processing copy,
then test its validity interval and select the latest valid source revision.
Only then do metadata filters apply. A later source update closes every earlier
overlapping processing copy, while read-time selection also protects stores
written before this correction. An expired latest copy cannot expose an older
open copy. Intervals include their start and exclude their end. This supports
valid-time reads and explicit updates; it does not provide automatic upstream
refresh, change-data-capture connectors or a transaction-time query API.

Leaf and parent records retain exact source offsets. Generated contextual prefixes remain separate from source text. Hierarchy summaries store original-leaf lineage and expand into original leaves for citations. Graph traversal uses relations whose evidence IDs are already in the authorized revision scope. Hypothetical HyDE documents and generated summaries are never accepted as original source citations.

## Text processing and prompt budgets

The six ingestion strategies are fixed, recursive, structure, parent-child, semantic, and contextual. Semantic boundaries require a trained encoder; contextual prefixes require a generator. Missing required dependencies produce an error. Structure parsing recognizes text/Markdown blocks and heading paths; it is not a PDF layout parser.

Token spans and source ranges are incremental. Before retaining draft strings
or calling contextual generation, the planner validates a bounded offset plan:
at most 1,024 source records and 8 MiB of retained UTF-8 chunk text, context and
distinct embedding strings. Parent/leaf records share these budgets; generated
context is charged before constructing a combined embedding string. Semantic
splitting keeps its separate 2,048-unit limit. Later hierarchy summaries retain
their own bounds rather than being counted as source-plan records.

Unicode code-point counting is an explicit baseline. Tiktoken resolves a known model or explicit encoding and preserves character boundaries when producing source spans. Stored counts identify their tokenizer and counting unit. Source chunking and generation can use different tokenizers; selecting a correct model encoding remains an operator responsibility for custom compatible endpoints.

Before generation, the service packs the rendered instructions, question, citation headers, separators, and evidence within the configured context budget after reserving output and envelope allowances. The response records included/dropped/truncated sources and counting provenance. An oversized fixed prompt fails rather than dropping the question. Unicode counts and reserved framing estimates are not presented as exact provider token accounting.

## Retrieval execution

| Mechanism | Execution and dependency boundary |
| --- | --- |
| Sparse, dense, naive, hybrid | PostgreSQL full-text or SQLite FTS5 ranking, exact cosine search, direct evidence assembly, and independent rank fusion |
| Parent-child, hierarchy | Persisted parent expansion or recursive vector clusters/summaries with original-leaf lineage |
| Graph | Explicit or validated model-extracted relations; two-hop traversal and label-propagation communities |
| Structured | Typed row/metadata predicates and bounded numeric aggregations; no submitted SQL |
| Temporal | Authorized version selection and recency-aware ranking |
| Fusion, HyDE, contextual | Distinct query-generation, hypothetical-document, or contextual-index mechanisms; required generation/index dependencies are explicit |
| Corrective, adaptive, agentic, self-reflective | Validated routing/planning/evidence review/draft critique, with at most three refinement rounds |
| Multi-vector / ColBERT | Real pretrained query/document token matrices and normalized masked sum-MaxSim |
| Image assets | Paired CLIP text/image vectors, cosine ranking, and original-image citations through `/assets/*` |
| Federated | Operator-defined source IDs and per-tenant credentials, bounded parallel reads, source-qualified IDs, and RRF |

Each federation request has a ten-second whole-call deadline through connection,
headers and body, plus a 2,000,000-byte raw response cap. Any failed branch rejects
the federated result. Bounded worker scheduling means this is a per-source
deadline, not a ten-second bound for every multi-source query.

The native hierarchy, native graph and reflection loop keep their own names. Separate `raptor`, `lightrag`, `full_graphrag` and `hipporag` identifiers execute pinned upstream implementations through the request-local service below. `self_rag` executes the official trained checkpoint through a bounded quantized CPU port; `crag` executes the official trained evaluator and all three correction branches. These two engines preserve the selected answer through the versioned source contract and existing admission/budget gates. See [implementation and evidence boundaries](RESEARCH_COMPLETION.md) and the [upstream ledger](RESEARCH_ENGINES.md).

Model-dependent paths validate structured outputs, permitted choices, and source IDs. Bounded loops record decisions and stopping reasons. There is no arbitrary shell execution or general-purpose tool execution in this RAG service. The companion workflow runtime owns its own tool and approval policies.

## Evidence decisions and profile identity

`EvidenceAdmission` runs on authorized original evidence before finalizing and exposing an answer,
including when `generate=false`. Its result separates successful retrieval from
insufficient evidence, and identifies unassessed ranking, calibrated relevance,
calibrated answerability, explicit model review or exact structured data. A
refusal removes all usable citations. Candidate IDs/counts remain separate
diagnostics and cannot become downstream evidence.

Self-RAG and CRAG generate a candidate answer inside the retrieval worker as part
of their algorithm. That internal candidate is still subject to parent admission
and source-budget checks before it can become the public answer.

Compatibility without a profile is explicitly unassessed. Required policy needs
a configured immutable profile for ordinary passage evidence. The profile
fingerprints actual model/config/tokenizer artifacts, runtime, scoring convention,
prompt/window, calibration digest and threshold. It cannot be selected by a
request. Invalid profiles, unavailable artifacts and malformed scores are
execution errors. Exact structured aggregates carry complete bounded dataset
provenance and do not fabricate rows for a zero result.

BGE computes relevance logits through its actual ONNX cross-encoder. Qwen uses a
pinned local CPU model, a fixed source-only instruction and next-token YES/NO
logit difference. Both use maximum original-passage score and retain candidate
order/set on acceptance. Qwen's single-passage question is a material limitation
for questions requiring multiple passages together; no profile is promoted by
passing protocol tests. Calibration and held-out measurements remain separate.

The process caches only model state and profile identity. Inference is bounded;
source-prefix truncation and full prompt windows are explicit. User evidence and
scores do not become a shared model cache. Federation validates remote decisions
and removes refused branches before rank fusion. A separately admitted branch
may still contribute evidence.

## Request-local research execution

The authenticated research sidecar hosts one fixed engine. RAPTOR, LightRAG, GraphRAG and HippoRAG require the `tokenizer.tiktoken` permission because their provider bridge enforces actual encoder token windows. Self-RAG uses its pinned native tokenizer and model runtime; CRAG uses the evaluator's native tokenizer plus bounded configured generation.
The RAG service sends the original passages already selected for tenant,
metadata and time. The sidecar verifies its credential's tenant and the fixed
engine before starting the worker. A finite semaphore bounds active requests;
capacity exhaustion returns 429 rather than accumulating an unbounded queue.

Each request uses a fixed `python -m rags.research_worker` command, a fresh
working directory and a fresh temporary index. Its environment contains only
runtime and selected provider configuration, not RAG/Agent tenant credential
maps. Source-derived caches/logs are request-local; public weights may be shared.

Self-RAG and CRAG require `authorized-evidence-v2`, which binds a selected answer
to returned original source IDs. The parent never replaces that answer with a
generic generator; failed evidence admission or truncated cited evidence causes
refusal. CRAG may add bounded external snapshots only when the operator enables
the exact hosts on both services and authorizes the tenant on the sidecar. The
snapshot identity binds tenant, literal approved HTTPS URL and content hash;
external IDs cannot overwrite internal source IDs. Web-enabled CRAG rejects
metadata and historical filters before HTTP because current public pages cannot
honor those scope constraints. External pages remain ephemeral and do not become
stored tenant documents. See [the complete web acquisition contract](RESEARCH_COMPLETION.md#source-and-answer-contracts).
A complete wall timeout bounds stdin/output exchange. Output is read through a
bounded pipe; failures expose fixed codes. Completion, timeout and cancellation
terminate the process group and remove the temporary directory.

RAPTOR runs upstream cluster/tree construction and retrieval. It traces selected
summaries through child nodes to authorized original leaves. LightRAG initializes
its upstream stores, extracts/merges a graph and performs hybrid graph retrieval.
It resolves internal chunk records and checks both original identity and content.
Neither generated summaries nor extracted entity descriptions become standalone
citations. The provider bridge checks exact encoder token windows before calls,
as well as generation input/output and call-count bounds.

The client validates exact pinned implementation metadata and bounded diagnostic
fields when present or required. Such metadata is labelled upstream-reported;
actual source/model execution is established by the independent test reports.
No health response or configured URL is treated as algorithm verification.

## External boundaries and authorization

The caller cannot select a tenant, backend URL, or backend credential. Federation and research endpoints come from operator configuration; each tenant has a distinct mapped credential. Requests carry already authorized source passages or source-scoped query parameters. Research v1 responses select IDs only from supplied original leaf records. CRAG v2 may additionally select independently authorized, hash-verified external snapshots; the parent validates their tenant, URL, content and noncolliding identity. Federation IDs are source-qualified to prevent collisions.

HTTP adapters reject redirects and compressed responses and enforce byte limits before buffering. Research requests use one asyncio wall deadline spanning connection, headers and all body reads; the synchronous API facade owns that event loop. Federation retains its separately bounded transport checks. Numeric scores, aggregation values, usage fields and token matrices reject nonfinite values and inappropriate types. Invalid Unicode bearer credentials return unauthorized responses. Provider failures remain execution errors.

All document, archive, parent, graph, and original-asset operations use tenant scope. This service's static key map is not OIDC, document-level ACL management, or a complete public API gateway. External services still need appropriate source handling, retention, and credential isolation; local protocol validation cannot prove their internal behavior.

## Evaluation and operational limits

The historical [36-question synthetic report](evaluation-v02-summary.json) records offline hybrid Recall@3 `0.85` and no-answer refusal recall `0.0`. The v0.3 evaluation separates raw candidates from admitted evidence and records the frozen quality gates independently from contract tests. Record-valid citations establish source identity and span, not entailment. Actual RAPTOR/LightRAG execution uses real embeddings and labelled deterministic generation fixtures; live paid-generation quality remains unverified.

Actual BGE, ColBERT, and paired CLIP checkpoints were exercised separately. These small fixtures verify encoder/adapter execution and selected invariants; they are not domain-quality benchmarks. Ordinary tests also distinguish deterministic control-flow fixtures from real-model inference.

Exact pgvector scans, the serialized application transaction paths, on-demand late-interaction encoding and local federation threads retain workload-dependent scaling limits. SQLite remains a development and legacy-storage option. The service does not currently implement a distributed vector index, automatic streaming refresh, production identity gateway, quotas, or semantic answer-verification guarantees. Scaling and public deployment require those concerns to be addressed explicitly; they are not implied by passing local integration tests.


## v0.9 model capability and checkpoint boundaries

`rags/model_profiles.py` owns the selected provider's supported request fields,
strict JSON decoder and normalized usage. Both `Providers` and `ModelRouter`
consume that contract; an OpenAI-compatible URL alone never implies JSON-schema
support. Tinker/Inkling uses `max_completion_tokens`, optional explicit reasoning
effort, and independently configured embeddings. UTF-8-byte budgeting preserves
Unicode boundaries and advertises its conservative counting provenance.

`rags/provider_transport.py` is the shared direct-provider, router and federation
HTTP boundary. It bounds complete JSON requests before I/O (normally 1 MiB, or
3,000,000 bytes for image encoding), rejects redirects/compression, and streams
bounded raw responses under one cancellable wall deadline through connect,
headers, body and owned cleanup. Direct providers and the router use a 512 KiB
response cap; federation explicitly uses its 2,000,000-byte cap. The synchronous
facade owns and closes its event loop. Router calls disable ambient proxy trust;
an optional bounded nonsymlink `RAGS_ROUTER_CA_FILE` snapshots trusted PEM bytes,
preserves HTTPS hostname verification and adds the CA hash to approval identity.

Multipart intake stays bounded, then offloads extraction, image validation,
captioning, embeddings and storage to an eight-slot process-owned pool.
Cancellation does not release admission until synchronous work finishes; excess
work returns `503 ingestion_capacity` without starting ingestion.

The router bounds the entire request, including tool schemas, rejects compressed
responses, duplicate JSON fields, nonfinite numbers, undeclared or inconsistently
terminated tool calls and malformed envelopes. Cache or reasoning usage is
reported only if actually supplied by the provider. Bounded reasoning continuation text is kept only in the private run checkpoint
and replayed to the model; public reply/checkpoint serialization, agent API
summaries and conversation/observation memory exclude it. This preserves tool
continuity through approval/restart without exposing internal reasoning in the
product interface.

The execution plane checks truncation before any tool action and binds durable
approvals to the complete registered schema/policy/model contract. A changed
contract requires a new run; a retry cannot silently authorize a changed tool.
The versioned v0.9 fingerprint intentionally invalidates v0.8 pending agent
approvals and resumable agent checkpoints; operators must start new runs and
review new approval requests after upgrading.
Read-only history access retains tenant/owner checks without requiring a match
to the current runtime fingerprint.
Existing tenant scopes, compare-and-swap revisions and idempotent-tool rules
remain authoritative. See [SPEC_V09](SPEC_V09.md) for the observed defects and
required regressions.

A terminal native-agent checkpoint and its optional user/assistant history pair
now commit in one database transaction. SQLite's write transaction or a
PostgreSQL session advisory lock allocates adjacent message positions; the run
revision compare-and-swap prevents duplicate finalization. Failure rolls back
both messages and checkpoint. Owner/tenant scope and verified-citation checks
remain in force, with no new schema or rewrite of old conversation messages.
