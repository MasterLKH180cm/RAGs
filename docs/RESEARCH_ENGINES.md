# Research engine integrations

RAGs executes pinned **RAPTOR**, **LightRAG**, **Microsoft GraphRAG** and
**HippoRAG** through isolated authenticated services. The trained **Self-RAG**
checkpoint runs through a bounded quantized CPU port; **CRAG** uses the authors'
trained evaluator and implements all three correction branches. Each has actual
execution evidence; generation fixtures and live-service limits are labelled.

The RAPTOR/LightRAG record below is the original **2026-11-21** verification.
The four additional engines have a separate current
[implementation and execution ledger](RESEARCH_COMPLETION.md), including model
identities, install commands, measured workloads, web-source controls and limits.
The [v0.3 specification](SPEC_V03.md) defines the original contract; the
[completion specification](RESEARCH_COMPLETION_SPEC.md) adds the answer-preserving
v2 contract. The
[architecture](ARCHITECTURE.md#request-local-research-execution) describes its
service boundaries. Model quality, paper benchmark reproduction and paid
generation behavior are separate from the execution checks below.

## Implemented upstream paths

| Engine | Exact upstream identity | Local adapter |
| --- | --- | --- |
| RAPTOR | Official source [`7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`](https://github.com/parthsarthi03/raptor/tree/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767); reported version `git-7da1d48` | [`raptor_adapter.py`](../rags/research/raptor_adapter.py) |
| LightRAG | `lightrag-hku==1.5.7`, official source [`28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`](https://github.com/HKUDS/LightRAG/tree/28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c) | [`lightrag_adapter.py`](../rags/research/lightrag_adapter.py) |
| Microsoft GraphRAG | `graphrag==3.3.0`, official source [`5faaaf4f5685fa2056fa8c3bf9342cb089f2942f`](https://github.com/microsoft/graphrag/tree/5faaaf4f5685fa2056fa8c3bf9342cb089f2942f) | [`graphrag_adapter.py`](../rags/research/graphrag_adapter.py) |
| HippoRAG | Official source version `2.0.0a5`, [`d5c8329422e0a0b834a15874545cb6a74b4f9b26`](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26) | [`hipporag_adapter.py`](../rags/research/hipporag_adapter.py) |
| Trained Self-RAG | Reference inference [`1fcdc420e48f50a7d7ab1ece5494221b93252e99`](https://github.com/AkariAsai/self-rag/tree/1fcdc420e48f50a7d7ab1ece5494221b93252e99), official checkpoint revision `190261383b0779ff66d2f95a73c7ad267d94b820` | [`selfrag_adapter.py`](../rags/research/selfrag_adapter.py), [`selfrag_runtime.py`](../rags/research/selfrag_runtime.py) |
| CRAG | Reference [`de7c2961ae624a1483a138c5798e1f6d0c4fb0e0`](https://github.com/HuskyInSalt/CRAG/tree/de7c2961ae624a1483a138c5798e1f6d0c4fb0e0), individually hashed official evaluator artifacts | [`crag_adapter.py`](../rags/research/crag_adapter.py), [`crag_web.py`](../rags/research/crag_web.py) |

RAPTOR is installed from the official source checkout, not the unrelated PyPI
package named `raptor`. The installer verifies its clean pinned checkout; the
adapter verifies hashes of the upstream implementation files it uses. LightRAG
uses a wheel pinned by version and SHA-256 in its dependency manifest, with
runtime checks of the installed version and relevant source files. Both report
`adapter_version: "v03"`.

### RAPTOR execution and source lineage

The adapter invokes the upstream `ClusterTreeBuilder`, `RAPTOR_Clustering`
UMAP/GMM path and `TreeRetriever`. Each supplied passage becomes one original
leaf. Cluster summaries become internal tree nodes and receive real embeddings.
The configured reduction dimension is three, with at most four summary layers.
The upstream stopping rule leaves corpora of four or fewer passages without
summary layers; the verified 16-passage fixture exercises actual clustering and
summary retrieval.

Selected summaries expand recursively through upstream child relationships.
The adapter verifies the original leaf identity and text before returning a
source ID. Unknown leaves, changed source text and invalid lineage fail the
request. Generated summary text is never returned as an original citation.
Scores are projected reciprocal selected-node ranks, labelled
`reciprocal_selected_node_rank`; they are not confidence probabilities.

This is a bounded integration of the upstream retrieval mechanism. It does not
reproduce the [RAPTOR paper's](https://arxiv.org/abs/2401.18059) full training,
evaluation datasets, model choices or benchmark scores. UMAP/GMM can produce
different tree sizes across executions.

### LightRAG execution and source lineage

The adapter initializes the upstream stores, inserts original passages through
`ainsert_custom_chunks`, and calls `aquery_data` with `QueryParam(mode="hybrid")`.
This insertion API is deprecated but present in the pinned 1.5.7 release. It
executes upstream entity/relation extraction and graph merging; the adapter does
not substitute a prebuilt fixture graph. Storage uses upstream JSON stores,
NanoVectorDB and NetworkX, and is finalized before request cleanup.

Each authorized passage has its own upstream document identity. Retrieved
internal chunk records must map back through `full_doc_id`; both stored and
returned source text must equal the original authorized passage. Unmapped
sources fail. Entity descriptions and relation summaries cannot become source
citations. Scores represent reciprocal hybrid chunk ranks, labelled
`reciprocal_hybrid_chunk_rank`.

The path performs graph-based hybrid retrieval, with no silent naive fallback.
Malformed extraction or keyword output is an execution error. Query and document
embedding callbacks remain distinct. The integration uses bounded extraction,
summary and retrieval settings; it does not establish parity with every upstream
storage backend or the authors' evaluation results.

## RAPTOR/LightRAG providers, isolation and limits

The [shared provider bridge](../rags/research/common.py) requires a configured
trained embedding provider and generator. The worker constructs ordinary RAGs
`Providers`; deterministic generation is injected only by the test harness.
Missing dependencies or invalid model outputs fail explicitly. There is no
runtime hashing, extractive or fixture fallback for these engines.

The verified embedding model is `BAAI/bge-small-en-v1.5` through FastEmbed 0.9.1,
with 384-dimensional vectors and a 512-token native input window, including
special tokens. The bridge counts full inputs with the encoder's tokenizer and
rejects oversized input before inference. It does not silently truncate a source
while claiming to encode the whole passage. Known OpenAI embedding models use
their configured exact counter and a conservative window; other providers must
supply a supported exact counting path. Chunking must fit the selected encoder.

Generation inputs include system/history/user content and reserved framing and
output allowances. The bridge bounds generation and embedding calls and rejects
nonfinite, zero or inconsistent vectors. These resource checks do not establish
semantic correctness of generated summaries or extracted relations.

The [sidecar](../rags/research_service.py), [worker](../rags/research_worker.py)
and [protocol](../rags/research_protocol.py) enforce:

- One operator-selected engine and credential-derived tenant per service.
- At most 64 original passages and 200,000 source characters per request.
- One fixed Python worker command, a new process and a fresh private index for
  each authorized revision snapshot. Source-derived indexes are never reused
  across requests. Public model weights may share a cache.
- A complete process deadline, bounded output, process-group termination and
  temporary-directory cleanup on completion, failure, timeout or cancellation.
- A per-service-process concurrency bound; multiple Uvicorn workers multiply
  that capacity.
- Validation of returned original IDs, finite scores and bounded implementation
  metadata. The caller cannot choose the backend URL or tenant credential.

The RAG API selects tenant, metadata and temporal scope before sending sources.
An update, deletion or `as_of` query therefore supplies a newly selected corpus;
the service cannot retain an older private index. Provider-side retention remains
an operator/provider responsibility. The runtime label
`pinned_upstream_reported` identifies validated *reported* implementation
metadata; actual execution evidence comes from the tests, not from a health
response or configured URL.

## RAPTOR/LightRAG provisioning and reproduction

Use separate environments. The recorded installations were checked on CPython
3.12.14 and Linux x86_64; their incompatible transitive pins should not be merged
into the main service environment.

```bash
bash scripts/install_research.sh raptor .venv-raptor ./upstream-raptor
bash scripts/install_research.sh lightrag .venv-lightrag
```

The [installer](../scripts/install_research.sh) uses Python 3.12 by default
(`RAGS_RESEARCH_BOOTSTRAP_PYTHON` can select its executable), installs the local
RAGs checkout with uv, and runs `uv pip check`. uv 0.13.9 is the tested installer.
Its explicit CPU backend selects the PyTorch index only for PyTorch packages;
the remaining pinned dependencies resolve from PyPI. See the
[RAPTOR manifest](../requirements/research-raptor.txt) and
[LightRAG manifest](../requirements/research-lightrag.txt) for the full recorded
dependency versions. The LightRAG wheel SHA-256 is
`895b81d94ca447afc6290c0e049d339f6caed46fbf29e4324f6db987046c6fcb`.

The [README service example](../README.md#raptor-and-lightrag-services) shows
provider credentials, tenant keys, service startup and the main API backend
configuration. Set `require_pinned_upstream:true` for these services. Keep the
main API's client deadline longer than the sidecar deadline. Production backend
URLs require HTTPS; the localhost example uses development mode.

To run the actual adapter test, select one installed engine:

```bash
RESEARCH_ENGINE=lightrag # or raptor
export RAGS_MODEL_CACHE="$PWD/data/models"
ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 \
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
RAGS_TEST_RESEARCH_UPSTREAMS="$RESEARCH_ENGINE" \
RAGS_RESEARCH_REPORT_PATH="reports/research_${RESEARCH_ENGINE}_v03.json" \
".venv-${RESEARCH_ENGINE}/bin/python" -m pytest -q -s tests/test_research_upstreams.py
```

This downloads public embedding weights when absent. ONNX telemetry is disabled
before runtime initialization. The [fixture provider](../tests/research_fixture_provider.py)
supplies deterministic generation while performing real BGE inference.

To exercise the HTTP boundary and companion Agent chain, install the sibling
Agent service and its MCP dependencies in a separate environment, then run:

```bash
ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 \
".venv-${RESEARCH_ENGINE}/bin/python" scripts/e2e_research_v03.py \
  --engine "$RESEARCH_ENGINE" \
  --agent-python ../.venv/bin/python \
  --report "reports/research_${RESEARCH_ENGINE}_http_v03.json"
```

The default Agent harness is the sibling `agentic-system/tests/e2e_v03.py`;
`--agent-script`, `--model-cache` and `--raptor-source` allow explicit paths.
The harness launches real Uvicorn APIs and the fixed process runner. Its local
OpenAI-compatible HTTP fixture is labelled deterministic generation in every
report. It is not a paid-provider or trained-generation quality test.

## Recorded verification

| Evidence | RAPTOR | LightRAG |
| --- | --- | --- |
| Actual upstream adapter report | [Tree construction/retrieval](../reports/research_raptor_v03.json) | [Graph extraction/hybrid retrieval](../reports/research_lightrag_v03.json) |
| HTTP and independent Agent worker report | [Ten boundary/lifecycle checks passed](../reports/research_raptor_http_v03.json) | [Ten boundary/lifecycle checks passed](../reports/research_lightrag_http_v03.json) |
| Initial HTTP fixture | 16 sources; 25 tree nodes, 9 summaries, 2 summary layers; selected nodes include 7 summaries | 16 sources; 16 graph nodes and 16 edges; retrieved entities, relationships and source chunks |
| Provider evidence | Real BGE embeddings; deterministic generation | Real BGE embeddings; deterministic generation |

The focused adapter suite recorded **33 passed, 2 opt-in checks skipped** without
an upstream selection. Each isolated environment recorded **34 passed, 1 skipped**
when its actual upstream test ran; the other engine's check remained skipped.
The direct reports include package versions and hashes of the BGE model and
tokenizer artifacts. Numeric graph/tree counts describe those executions rather
than a promised deterministic tree shape.

Each HTTP report covers invalid-token and tenant rejection before model calls;
actual engine identity and original source IDs; an independent Agent worker's
terminal REST/SSE and both stdio MCP paths; updates; historical reads; the same
document ID in another tenant; current and historical deletion; an empty
authorized scope; and a real worker deadline with index cleanup. The empty
corpus is explicitly `engine_executed:false`, not counted as algorithm execution.

These reports verify implementation execution, lineage and lifecycle boundaries.
They do **not** establish trained-generation quality, paper benchmark reproduction,
large-corpus capacity, throughput, or universal answer grounding. Admission and
answerability measurements have their own profiles and reports; an upstream
retrieval result is not automatically a calibrated acceptance.

## Completed additional research engines

The previously source-audited prerequisites now have concrete implementations.
Their `adapter_version` is `v10`; this is an adapter cohort, not a change to the
package's `0.9.0` version. Optional environments keep their dependencies out of
the ordinary offline service. The full supported scope and honest limits live in
[RESEARCH_COMPLETION.md](RESEARCH_COMPLETION.md).

### Microsoft GraphRAG

The adapter executes standard LLM entity/relation extraction, graph construction,
Leiden communities, community reporting, embeddings and upstream local
graph-context retrieval. It verifies actual pipeline stages and source lineage,
normalizes validated extraction to the pinned parser's delimiter rules, and
rejects swallowed upstream failures. Real upstream regression checks assert
relationship weights as well as graph existence. Native `graph` remains separate;
global/DRIFT search and FastGraphRAG are outside this implementation.

### HippoRAG

The adapter invokes official indexing, OpenIE, fact selection and graph retrieval,
and observes actual igraph personalized PageRank calls. Original source identities
survive internal text deduplication. A no-fact or dense-only fallback returns an
explicit error instead of being recorded as PPR execution. The isolated source
installation uses pinned CPU dependencies, trained BGE and actual provider
callbacks; no vLLM/GritLM extra is required for this supported configuration.

### Trained Self-RAG

The official trained 7B checkpoint is reproducibly converted from pinned source
shards to Q4_K_M and decoded with a pinned llama.cpp CPU runtime. The short-form,
always-retrieve port measures actual relevance/support/utility token logits and
retains the selected model answer and source IDs. Tests verify the reflection
vocabulary, EOS handling, valid supported answer, and whole-answer abstention for
irrelevant/no-retrieval/contradictory output. See [CPU port details](SELF_RAG_CPU.md).
This does not establish full-precision, adaptive retrieval, long-form or paper
benchmark parity. Native `self_reflective` and Qwen admission remain separate.

### Paper-specific CRAG

The authors' T5 evaluator artifacts are downloaded, hash checked and executed
in FP32 on CPU. The original PopQA thresholds select correct/incorrect/ambiguous
branches. Decomposition preserves source slices; ambiguous results retain both
internal and external evidence. The configured generation bridge returns an
explicit selected answer; actual tests label its deterministic fixtures.

Optional rewritten Serper acquisition uses exact host and tenant allowlists,
public DNS validation, pinned-IP verified TLS, one deadline and bounded content.
The v2 protocol binds each ephemeral web snapshot to tenant, URL and text hash.
All three branches and the real HTTP lifecycle are exercised with the actual
evaluator. TLS transport fixtures are executed, while live Serper and end-to-end
learned generation quality remain unverified. Native `corrective` remains a
distinct mechanism; these execution checks do not reproduce paper benchmarks.
