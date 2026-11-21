# RAGs v0.3 specification: evidence decisions and actual upstream engines

Status: specification and acceptance data prepared before feature implementation.
Baseline: the reviewed v0.2 branch at `78d3d85036b916f4ff4deb1e2abf523f25111a4f`.
Companion: `agentic-system/docs/SPEC_V03.md`.

## 1. Purpose and scope

Continue the original twenty-family design by correcting a measured weakness
and implementing two actual research integrations. The v0.2 hash/extractive
hybrid baseline has refusal recall 0.0; ranked results alone do not establish
relevance or answerability. Its generic research HTTP client does not execute
RAPTOR or LightRAG. The Agent also discards upstream refusal decisions.

This iteration delivers:

1. An explicit query outcome and evidence-admission contract, including pure
   retrieval requests, budget exhaustion, federation, and exact structured zero.
2. An optional actual multilingual cross-encoder with a frozen, independently
   calibrated profile and separate held-out measurements.
3. A bounded authenticated research service executing pinned upstream RAPTOR
   and LightRAG code in request-local worker processes and temporary indexes.
4. Actual HTTP/worker/MCP verification, self-review, and honest documentation.

It does not redefine citation validity as entailment, train a new language
model, or declare all named papers reproduced. Full Microsoft GraphRAG,
HippoRAG, trained Self-RAG, and paper-specific CRAG remain separately identified
upstream work. Their prerequisites and status remain visible in the ledger.

## 2. Query outcome contract (R3-01)

Every successful `/query` response, including `generate=false`, adds:

```json
{
  "outcome": "success",
  "supported": null,
  "refusal_reason": null,
  "admission": {
    "policy_version": "evidence-admission-v1",
    "decision": "unassessed",
    "basis": "ranked_unassessed",
    "profile_id": null,
    "score": null,
    "threshold": null,
    "candidate_count": 3,
    "accepted_count": 3
  }
}
```

- `outcome`: `success` or `insufficient_evidence`. Success describes the evidence
  operation; it is not a universal guarantee of a correct answer.
- `supported`: `true` only for an explicit supporting model assessment or exact
  structured result; `false` for refusal; otherwise `null`. Cross-encoder
  relevance acceptance alone leaves this field `null`.
- `refusal_reason`: `null`, `no_candidates`, `below_admission_threshold`,
  `model_rejected`, or `context_budget_exhausted`.
- Admission decisions: `accept`, `abstain`, `unassessed`.
- Admission bases: `no_candidates`, `ranked_unassessed`, `calibrated_reranker`,
  `model_review`, `structured_exact`, `context_budget`.
- Scores and thresholds are finite numbers or null. Profile IDs and diagnostics
  are bounded. Scores are not represented as probabilities.

Refused responses have `supported=false`, `admission.decision=abstain`, and
`citations=[]`. With `generate=false`, `answer` remains null. No model generation
is invoked after admission refusal. Diagnostic candidate IDs/counts are kept
separate from usable citations; they cannot be consumed as evidence downstream.

Authorization errors, missing configured models/profiles/backends, invalid model
outputs, and transport failures remain explicit execution errors. They must not
be transformed into a successful empty query or domain refusal.

## 3. Evidence admission and profile identity (R3-02)

Operator controls are `RAGS_ADMISSION_POLICY=compatibility|calibrated_required`
and `RAGS_ADMISSION_PROFILE`, a local JSON profile path. Requests cannot change
the model, threshold, endpoint, profile, or required policy.

- Compatibility without a profile reports `unassessed` for ordinary ranked
  evidence. The existing hash demo remains available and honestly labelled.
- A configured profile must be valid and match the loaded scorer; an invalid
  profile never silently disables admission.
- Required mode fails closed when no applicable calibrated profile is present.
  Explicit model support cannot bypass the required calibrated gate.
- Exact structured operations with complete authorized provenance are a separate
  deterministic admission basis and do not need a relevance scorer.

The first actual scorer is FastEmbed `BAAI/bge-reranker-v2-m3-int8`, using the
supported multilingual ONNX checkpoint. The profile records schema version,
model identity, actual model/tokenizer artifact SHA-256 values, runtime/library
identity, returned-score convention, input token/window limits, aggregation,
threshold, language coverage, calibration corpus digest and measured objective.
The profile identifier covers these settings. Missing or mismatched artifacts
invalidate the profile. No model download or calibration is triggered by an
arbitrary request setting.

Admission evaluates the original question and authorized original source
passages. Generated context, hypotheses and summary nodes are not standalone
source evidence. Query-level aggregation is the maximum original-passage score;
acceptance preserves the existing candidate order and set. There is no lexical
overlap veto, hidden reranking, or shared cutoff for hash/cosine/RRF/ColBERT.
Rejecting a query removes all usable citations. Whole-prompt packing still runs
after admission and can independently produce `context_budget_exhausted`.

The scorer bounds its candidate count, text/token windows, input size and CPU
threads. Long-input behavior is recorded in the profile and reproduced during
calibration; truncation must not be hidden. Model/profile loading is cached,
while user evidence and scores must not be reused across tenant or source
revision boundaries.

## 4. Structured and federated evidence (R3-03)

An explicit typed aggregation over an authorized dataset can validly return
zero. Distinguish that from having no authorized dataset. Structured results
add `provenance.datasets`, a nonempty list of at most 20 objects containing
`document_id`, `revision`, and `profile_id`. Aggregation over more datasets fails
explicitly rather than truncating provenance or silently computing a partial
result. An empty authorized scope refuses with `no_candidates`.

The Agent may consume an exact structured value only with valid bounded
provenance. It need not invent a matching source row for `count=0`.

Federation validates and preserves remote decisions. A refused source cannot
contribute usable evidence; another independent accepted source may still do so.
Legacy remote results without the contract remain explicitly unassessed.
Contradictory or malformed typed responses cannot become supported evidence.

## 5. Frozen calibration and evaluation (R3-04)

Before scorer implementation or inference, commit:

- `examples/admission_calibration_v03.json`: 48 labelled cases, 24 positive and
  24 negative, balanced English/Traditional Chinese.
- `examples/admission_holdout_v03.json`: 32 topic-disjoint cases, 16 positive and
  16 negative, balanced English/Traditional Chinese.

Cases include paraphrases without literal overlap, short and multilingual
questions, multi-source evidence, unrelated questions, missing requested fields,
wrong entities/dates, and authorization/version boundaries. Labels, references,
rationales, case IDs and split metadata are never passed to the scorer. Only
questions and passage text enter inference. Both files record content digests.

Choose the threshold once from calibration scores: maximize negative rejection
subject to at least 95% positive retention, with a documented deterministic tie
rule: consider the finite observed scores plus a finite value immediately below
the minimum; accept when `score >= threshold`; maximize negative rejection,
then positive retention, then choose the smallest tied threshold. If the data
or scores cannot produce a valid threshold, calibration reports failure.
Then freeze the profile before opening held-out results. Never adjust a
threshold or edit a fixture using held-out or v0.2 regression labels.

Quality acceptance for the tested profile:

1. Retain at least 15/16 held-out positives and refuse at least 13/16 held-out
   negatives; retain all premarked semantic-only positive cases.
2. On the original 36-question regression, preserve hybrid admitted-evidence
   Recall@3 of at least 0.85 and refuse at least 3/4 no-answer questions.
3. Preserve all six exact structured and four temporal checks.

If a measured gate fails, record the failure and the actual limits. Do not
rewrite these gates or promote a failed profile as verified. Relevance
admission is still not complete answer entailment, even when these gates pass.

The evaluator reports raw candidate retrieval and admitted-evidence metrics
separately; domain outcome drives refusal metrics in both generation modes.
Report false refusals, abstention, refusal precision/recall, per-language and
category results, and applicable sample counts. The offline baseline must not
inherit external providers or a production admission profile. Real-model
evaluation is explicitly selected and never calls paid generation implicitly.

## 6. Actual research service (R3-05)

Continue the existing `authorized-evidence-v1` request shape. An authenticated
sidecar serves one operator-selected engine, `raptor` or `lightrag`; the caller
cannot select arbitrary modules, commands, paths, endpoints, or model settings.
Its token map derives the tenant, which must match the request namespace.

Requests carry at most 64 unique bounded passage IDs and at most 200,000 source
characters, with separate finite body/query limits and a bounded `top_k`.
Oversized requests fail before model/index work. Returned IDs must resolve to
the supplied passages, with finite scores and a bounded response.

Each request runs a fixed Python worker command in a fresh temporary working
directory. Use a finite operator-configured wall deadline, bounded admission
concurrency, and process termination/cleanup on timeout or failure. Do not
retain source-derived indexes or reuse another request's namespace. Model
artifacts may be cached separately. Child error responses and service events
omit source text, prompts, credentials and unbounded upstream exceptions.

The RAG HTTP client gets a validated operator-configured backend deadline for
these workloads. It retains bounded response reads and validates library/version
provenance separately from arbitrary upstream text. Generic adapters without
such provenance remain external/unverified; a configured URL is not proof that
the upstream algorithm ran.

### RAPTOR

Pin official source commit `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767` from
`parthsarthi03/raptor`. The upstream has no installable package metadata: use a
pinned source checkout and an isolated, reproducible CPU dependency environment,
not an unrelated PyPI package called `raptor`.

Use actual `ClusterTreeBuilder` clustering/tree construction and `TreeRetriever`.
Map original authorized chunks to leaf nodes; recursively resolve selected
summary nodes through children to original leaves. Generated summary text must
never masquerade as a source citation. Actual UMAP/GMM-generated summary levels,
retrieval node selection, and leaf lineage are required verification evidence.
Use distinct trained document and query embedding callbacks. Projected leaf
scores must state their convention, such as reciprocal selected-node rank;
upstream layer information is not a numeric confidence or probability.

### LightRAG

Pin `lightrag-hku==1.5.7`, official tag commit
`28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`. Its release wheel SHA-256 is
`895b81d94ca447afc6290c0e049d339f6caed46fbf29e4324f6db987046c6fcb`.

Use actual storage initialization, graph extraction/merge through
`ainsert_custom_chunks`, and `aquery_data` with hybrid graph retrieval. Resolve
returned internal chunks through the upstream store to supplied document IDs
and verify the original passage text. Unknown IDs or changed source content fail
closed. Preconstructed custom graphs or naive retrieval do not satisfy this gate.
Use the upstream asymmetric embedding context to distinguish document and query
encoding when the trained embedding model requires it.

Production generation/embedding callbacks require explicitly configured trained
providers. A deterministic generator is allowed only in labelled tests; it must
not be a hidden runtime fallback. Actual upstream tests use real BGE embeddings
and identify deterministic summary/extraction fixtures separately from live
generator quality. Missing models, packages, or provider configuration are
explicit requirement errors.

## 7. Integration and review gates (R3-06)

- All query contracts, refusal priority, empty scope, finite values, source
  authorization, required-profile policy and structured-zero invariants pass.
- Both upstream engines execute their actual pinned code on CPU, build their
  relevant indexes, and return valid source lineage with real BGE vectors.
- The authenticated sidecar and RAG API exchange actual HTTP requests. Exercise
  wrong tenant/token, update/delete/as-of snapshots, timeout cleanup, malformed
  output, and returned IDs outside the authorized set.
- The companion independent Agent worker preserves refusal through synthesis,
  merges, loops, durable outcomes, SSE, and terminal MCP status/events.
- Repeat the existing meaningful regression and real HTTP/MCP/Compose gates.
  Distinguish actual encoder/upstream execution from controlled model fixtures.
- Self-review the concrete changes, fix findings, update README diagrams and
  coverage/evidence ledgers, and record passing or failing quality measurements.

Commit order: specification plus frozen acceptance data, implementation, actual
verification, bounded self-review corrections, then final evidence documentation.

## Primary references

- https://github.com/parthsarthi03/raptor/tree/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767
- https://github.com/HKUDS/LightRAG/tree/28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c
- https://qdrant.tech/documentation/fastembed/fastembed-rerankers/
- https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX
