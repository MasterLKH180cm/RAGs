# RAGs v0.2 implementation review

## Scope and evidence rules

The [v0.2 specification](SPEC_V02.md) was committed before feature implementation, following the requested specification → implementation → self-review sequence. This review records implemented behavior and measured checks against that contract. It does not declare all twenty original research families reproduced.

The [README coverage table](../README.md#twenty-family-coverage) retains every original family. Native mechanisms have explicit names. Required trained models and generators fail closed when unavailable. An external research adapter is an implemented protocol boundary; the named upstream engine remains an unverified external requirement until its actual indexing and retrieval have been exercised.

## Requirement ledger

| Requirement | Implemented behavior and evidence | Remaining boundary |
| --- | --- | --- |
| RAG-01: tokenization | Explicit Unicode/tiktoken backends; exact source spans; CJK, emoji, combining characters, special-token-looking input, and small-budget coverage in [tokenizer tests](../tests/test_tokenization.py) | A configured encoding must match the generation endpoint for model-token claims; Unicode counts remain labelled units |
| RAG-02: six chunkers | Fixed, recursive, structure, parent-child, semantic, and contextual plans, with source/count metadata in [chunking tests](../tests/test_chunking.py) | Semantic requires a trained encoder; contextual requires generation; text structure does not imply PDF layout/OCR support |
| RAG-03: budgeting | Full rendered prompt packing, reserved output/envelope, selected/dropped/truncated source IDs, fixed-prompt rejection, and integration checks in [native tests](../tests/test_native_integration.py) | Provider framing allowance and Unicode counts are labelled; paid-provider exact-count integration is not measured |
| RAG-04: lifecycle | Profile-aware unchanged detection, additive legacy migration, guarded atomic reindex, version snapshots, derived-index cleanup, deletion of history, and cross-tenant checks | SQLite/local deployment; no automatic source refresh or distributed migration controller |
| RAG-05: capabilities | [Registry](../rags/registry.py), tenant-specific capability reporting, distinct native paths, and explicit dependency failures in [API](../tests/test_api.py), [native](../tests/test_native_integration.py), and [backend](../tests/test_backends.py) tests | Named external engines are not installed or validated by the adapter alone |
| RAG-06: structured and temporal | Typed row predicates/aggregations, validity intervals, scoped as-of selection, revision/delete/reindex checks; measured dedicated exact cases are 6/6 structured and 4/4 temporal | No caller SQL; no automatic streaming connector or separate transaction-time query API |
| RAG-07: hierarchy, graph, late interaction | Persisted recursive summary lineage; actual two-hop evidence traversal and communities; numeric masked MaxSim tests; actual ColBERT checkpoint execution | Native hierarchy/graph do not claim RAPTOR/Microsoft GraphRAG/LightRAG/HippoRAG reproduction |
| RAG-08: bounded model workflows | Distinct fusion/HyDE/routing/planning/corrective/self-reflective paths, validated structured responses/citations, bounded iteration and prompt budgets | Deterministic model fixtures establish control flow; live paid-model quality and research parity remain unverified |
| RAG-09: evaluation | [36 labelled synthetic questions](../examples/eval_v02.json), applicable retrieval/answer/refusal/latency metrics, exact structured/time checks, and [measured report](evaluation-v02-summary.json) | Domain quality is unmeasured; semantic entailment, paid usage, and cost are null with reasons |
| RAG-10: real integration | Real TCP RAG and Agent APIs, independent worker, document lifecycle, bounded workflows, and both stdio MCP clients exercised by the companion acceptance script | Local disposable integration; no claim that live paid APIs or external research engines were exercised |

## What is exercised, and what remains external

| Evidence class | Capabilities | Meaning |
| --- | --- | --- |
| Native deterministic execution | Sparse/dense/hash/hybrid/naive; parent-child; recursive hierarchy; explicit graph; structured; temporal; federation validation | Algorithms and lifecycle/security invariants have behavioral tests; hashing is not a semantic model |
| Actual pretrained checkpoint execution | BGE single vectors and semantic boundaries; ColBERT token matrices/late interaction; paired CLIP image/text retrieval | Real encoder inference and selected rankings were executed locally; these are small smoke fixtures |
| Implemented provider-dependent flow | Fusion, HyDE, contextual generation, model graph extraction/summary generation, corrective, adaptive, agentic, self-reflective | Provider interfaces and bounded control flow are implemented; tests use labelled deterministic responses where paid generation would be needed |
| Implemented external protocol, engine unverified | `raptor`, `full_graphrag`, `lightrag`, `hipporag`, `self_rag`, `crag`, and optional `multimodal_native` research backend | Operator must deploy the actual engine and provide per-tenant credentials; protocol validation is not research reproduction |
| Separately unsupported scope | Automatic source-refresh/streaming connectors, audio/video retrieval, image-answer generation, distributed indexing | Remains explicit work, rather than a renamed native substitute |

`colbert` can execute the tested local checkpoint; it may also be configured as an external backend. The local implementation records its actual model identity. Original-image retrieval is available through `/assets/*`; it is separate from the external `multimodal_native` query contract and from caption-only `/images` ingestion.

## Recorded validation

### Ordinary suite and focused integration

The complete ordinary RAG test suite finished with **140 passed, 3 skipped** after the request-body limits were added. The three skips are explicit opt-in actual-model checks; they were exercised separately. A subsequent focused evaluation regression prevents configured external engines from making network calls during the offline evaluator. Static checks use the repository's declared Ruff rule set.

The focused core/backend run also included actual ColBERT inference:

```bash
RAGS_TEST_COLBERT=1 RAGS_MODEL_CACHE=/tmp/rags-model-cache \
python -m pytest tests/test_backends.py tests/test_native_integration.py tests/test_api.py -q
```

Result: **47 passed**. Coverage includes typed filters, historical source isolation, real hierarchy lineage, two-hop graph evidence, reindex atomicity, bounded HTTP parsing, per-tenant federation/backend credentials, citation authorization, invalid Unicode credentials, and nonfinite-value rejection. This is a local result, not a claim that an unpublished GitHub Actions run has completed.

The tokenizer/chunker development gate separately ran **63 focused tests**, actual tiktoken inference, and seeded multilingual small-budget cases. It caught and corrected a BPE overlap/progress edge before integration.

All three README flowcharts passed actual Mermaid 11.17.2 syntax parsing with jsdom 26.1.0. This verifies the diagrams' syntax, not browser visual rendering.

### Actual encoder checks

| Check | Executed model and observed invariant | Limit of this evidence |
| --- | --- | --- |
| Semantic chunking / dense retrieval | `BAAI/bge-small-en-v1.5`; 384-dimensional vectors; actual topic split; Saturn evidence retrieved; [test](../tests/test_semantic_encoder.py) passed | One small synthetic fixture, not a multilingual semantic benchmark |
| ColBERT late interaction | `answerdotai/answerai-colbert-small-v1`; query matrix 32 × 96; finite query/document encodings; session policy ranks first among three documents; [test](../tests/test_backends.py) passed | Confirms the actual checkpoint and sum-MaxSim path, not general retrieval quality |
| Aligned image/text retrieval | `Qdrant/clip-ViT-B-32-text` and `Qdrant/clip-ViT-B-32-vision`; 512 dimensions; red-circle/blue-square text retrieval; image self-match cosine above 0.999; [suite](../tests/test_multimodal.py) finished with 12 passed including real inference | Simple visual fixtures, primarily English text; no image-answer generation |

The semantic smoke completed with **1 passed**. CLIP checks also cover artifact/model-file identities, the actual 77-token text limit, tenant isolation, and deletion. Auth/lifecycle tests using protocol fixtures are distinguished in test code from checkpoint inference. Optional models are cached through `RAGS_MODEL_CACHE`; the encoder extra is constrained to the tested FastEmbed minor version.

### Real HTTP, independent worker, and MCP

The companion [tests/e2e_v02.py](https://github.com/MasterLKH180cm/agentic-system/blob/master/tests/e2e_v02.py) acceptance flow passed with both the Agent SQLite store and PostgreSQL 16.2 store. RAG storage remained SQLite in both runs. The script starts actual TCP APIs, an independent worker process, and stdio MCP clients against disposable data.

The RAG checks include authenticated ingestion, exact source-span citations, prompt-budget accounting, tenant isolation, guarded reindex, historical reads, typed aggregation, deletion of temporal history, and HTTP-backed MCP tool calls. The companion checks include workflow execution, idempotency, bounded agent loops, persisted checkpoints/memory, approval, artifacts, events, and signed webhook replay rejection. This records integration boundaries; the Agent's separate review owns its complete twelve-layer status.

## Measured evaluation and interpretation

The [recorded report](evaluation-v02-summary.json) uses offline hashing and extractive generation on 36 synthetic labelled questions. Different modes have different applicable subsets; the report does not score every mode over all 36 cases. Unsupported modes are reported as unavailable, with unmeasured metrics null.

| Observation | Measured result | Interpretation |
| --- | --- | --- |
| Hybrid retrieval | Recall@3 `0.85`, MRR `0.8`, nDCG `0.8131` | Useful regression baseline on this corpus; not a production benchmark |
| Hybrid no-answer handling | Refusal recall `0.0` | Highest-ranked unrelated passages can still be returned; this is an unresolved quality limit |
| Graph retrieval | Recall@3 approximately `0.1364` | Explicit-edge traversal covers only a narrow portion of this mixed corpus; graph mode is not a general replacement for hybrid |
| Structured exact checks | `6/6` | Typed filtering/aggregation returns expected values on the dedicated fixtures |
| Temporal exact checks | `4/4` | Expected revision selection on dedicated valid-time fixtures |
| Source citation-record validity | `1.0` for measured available modes | Records, source text, revisions, profiles, and spans agree; semantic entailment is not measured |
| Extractive answer exact match | `0.0` | Source excerpts differ from concise reference answers; no semantic-answer score is inferred |
| Provider usage, cost, semantic entailment | `null` with reasons | No paid model execution, pricing, or semantic entailment assessor was fabricated |

Latency is measured on the local test host and excludes paid-model service behavior. The report records p50/p95 and applicable counts. Real encoder smoke fixtures are deliberately not converted into Recall@K quality claims for the 36-question corpus.

## Review findings corrected before handoff

| Finding | Correction and verification |
| --- | --- |
| Advanced mode labels could hide baseline execution | Distinct strategy registry and typed requirement failures; unavailable provider/index/backend paths have explicit tests |
| Processing changes could retain stale source indexes | Canonical profile includes tokenizer/config/model/endpoint/dimensions/prompt identity; unchanged/reindex tests exercise changes |
| Reindex failure or concurrent write could damage active evidence | Provider work precedes transaction; revision compare-and-swap and atomic commit preserve prior source/index |
| Generated summaries or hypothetical text could be promoted to source evidence | Original-leaf lineage and backend authorized-source checks; forged summary/source IDs are rejected |
| Draft answers could contain unauthorized inline citations | Self-reflective drafts validate inline IDs and declared cited IDs, including retrieval-only requests |
| Missing generated text could be mistaken for successful extraction | Model null/empty responses fail explicitly instead of invoking an extractive fallback |
| Context callbacks and generation could use the wrong counting profile | Per-document context budget/tokenizer are forwarded; generation tokenizer resolves independently; processing metadata persists |
| Nonfinite or loosely typed values could enter scores/aggregations | Typed scalar/list predicates, vector/mask checks, finite numeric scores, bounded usage fields, and rejection tests |
| Backend/federation replies could cross tenant or source boundaries | Operator-fixed endpoints, distinct tenant credentials, tenant-scoped catalog, supplied-leaf-only IDs, and namespaced federation results |
| Large or trickled HTTP responses could defeat bounded reads | Identity encoding, pre-buffer byte limits, no redirects, uncoalesced transport iteration, and deadline checks after each chunk |
| Unicode bearer tokens could trigger a comparison exception | Constant-time UTF-8 byte comparison; invalid Unicode credentials return `401` |
| Schema validation could occur after oversized JSON or multipart input was buffered | Pure ASGI intake cap of 2,100,000 bytes and a 30-second total receive deadline; 14 focused tests verify early rejection and downstream-handler isolation |
| Offline evaluation could inherit configured external engine endpoints | Offline evaluation explicitly disables research backends and federation even when the operator environment configures them; regression forbids opening an HTTP client |

These fixes are reflected in the owned core tests and backend adapter tests. Review and testing were bounded to concrete risks and required acceptance gates; no unmeasured external behavior is promoted to verified status.

## Remaining work and operational limits

The ordinary native mechanisms, source lifecycle, explicit token/chunk processing, actual pretrained encoder adapters, and local integration are exercised. The following limits remain material:

- **Named external research engines:** actual RAPTOR, Microsoft GraphRAG, LightRAG, HippoRAG, trained Self-RAG, and paper-specific CRAG deployments still need integration evidence. The adapter alone cannot establish the engine's indexing semantics, authorization, or retrieval quality.
- **Live model workflows:** paid OpenAI/Anthropic generation, generated contextual indexing, and model-led refinement require configured live-provider validation and a representative evaluation set before quality claims.
- **Answer quality:** the recorded hybrid no-answer failure and extractive answer mismatch remain visible. Citation existence checks do not certify grounding or entailment.
- **Scale and identity:** SQLite/full vector scans/on-demand encodings are a small-deployment design. Multi-node indexing, OIDC/document ACLs, quotas, and a production gateway are not implemented by this service.
- **Temporal and media scope:** explicit revision updates do not constitute automatic real-time ingestion. Paired image/text retrieval does not include audio/video or multimodal answer generation.

These are retained in the coverage ledger. They are not treated as completed through aliases, mocked provider output, or a green legacy test suite.
