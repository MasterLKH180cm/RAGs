# Remaining research RAG implementation contract

Prepared before implementation, against master
`f7b76d847cefca23a1142f66bb201d1dc01931ee` on 2026-11-23.

## Goal and scope

Complete the concrete missing research integrations in the twenty-family
ledger. Native `graph`, `corrective`, and `self_reflective` retain their current
names and behavior. A new identifier, fixture response, configured endpoint,
or missing-resource skip is not execution evidence. The frozen Qwen admission
profile, held-out data, thresholds and quality failures remain unchanged.

The four targets are Microsoft GraphRAG, HippoRAG, trained Self-RAG, and
paper-specific CRAG. Each target must separately report code implementation,
dependency/artifact readiness, actual algorithm execution, and model quality.
If an actual model artifact cannot be obtained or executed, record the exact
blocker and retain fail-closed capability reporting; do not substitute a
different model or mark an unexecuted gate passed.

## RRC-01: pinned Microsoft GraphRAG

- Pin Microsoft GraphRAG 3.3.0, source
  `5faaaf4f5685fa2056fa8c3bf9342cb089f2942f`, and its compatible packages in an
  isolated environment. Verify algorithm source identity before execution.
- Execute the standard upstream indexing pipeline: model entity/relation
  extraction, graph construction, Leiden communities, community reports and
  embeddings. Reject pipeline errors returned as data as well as exceptions.
- Execute upstream local graph-context retrieval. Global and DRIFT search are
  not implied by this selected query mode.
- Resolve retrieved text units through verified document/text-unit lineage to
  exact authorized original passages. Generated reports cannot become citations.
  Identical source texts must preserve all explicitly supplied source identities,
  without mapping to a different tenant or snapshot.
- Report upstream stages and graph/community/source counts, the exact query
  mechanism, provider identities and labelled ranking semantics.

## RRC-02: pinned HippoRAG

- Pin official HippoRAG source version 2.0.0a5 at
  `d5c8329422e0a0b834a15874545cb6a74b4f9b26` in an isolated CPU environment.
- Execute the upstream indexing, OpenIE, fact selection, graph construction and
  personalized PageRank retrieval with explicit bounded provider callbacks.
- Do not silently count the upstream dense fallback as successful PPR execution.
  Invalid extraction/filter output or swallowed provider errors must surface.
- Preserve exact original passage identity and text, including duplicates, and
  close all stores before removing each request's private index.

## RRC-03: trained Self-RAG

- Use the trained Self-RAG checkpoint and reflection tokens, not a prompt-only
  Qwen/ordinary chat substitute. Pin source, runtime, checkpoint and tokenizer
  identities. Investigate a bounded quantized CPU port when native reference
  weights exceed available memory; label that port and its numerical differences.
- Implement bounded short-form retrieval decisions and passage scoring from
  actual reflection-token probabilities. Stable probability normalization,
  finite scores, complete reflection-token vocabulary and context/output bounds
  are required. Record any corrections to reference inference defects explicitly.
- Preserve the selected trained answer and its selected authorized source IDs
  across the sidecar and RAG API. Ordinary generation must not overwrite it.
- An optional versioned grounded-answer result must reject missing/foreign IDs,
  invalid text, incompatible engines, partial output and unknown fields. Evidence
  admission and final context packing remain authoritative: a draft referring
  to removed/truncated evidence cannot bypass those checks.
- Protocol fixtures prove boundary behavior only. Actual checkpoint inference is
  required for trained-execution claims; absence of that evidence remains visible.

## RRC-04: paper-specific CRAG

- Inspect the official evaluator artifact linked by pinned source
  `de7c2961ae624a1483a138c5798e1f6d0c4fb0e0`; do not replace it with untrained T5
  or the unrelated Qwen answerability scorer. A usable installation requires
  immutable artifact identity and an actual evaluator inference check.
- A complete claim requires correct, incorrect and ambiguous decisions,
  internal evidence decomposition/refinement, rewritten external acquisition,
  and combined evidence with source provenance.
- The existing authorized-evidence-v1 contract may never accept arbitrary new
  IDs. Any external acquisition must have a separate explicit authorization and
  provenance contract: operator-controlled targets and credentials, bounded
  transport, validated immutable source snapshots and tenant-scoped identities.
  Missing approved acquisition or evaluator artifacts must fail explicitly.
- An evaluator-only or internal-only implementation is labelled partial. Do not
  bundle unsafe arbitrary-URL fetching from upstream scripts unchanged.

## RRC-05: shared execution and LiteLLM

- Extend the research protocol, sidecar configuration, explicit dispatch map,
  feature checks, installer and capability catalog only for implemented adapters.
  An unknown engine must never fall through to another engine.
- Forward the existing approved LiteLLM generation configuration to the fixed
  worker with dedicated credential/file handling. Only selected provider
  credentials cross the process boundary; API, tenant and unrelated provider
  secrets do not. Existing localhost authentication constraints still apply.
- Routed embeddings require a verified model/token-window contract; unknown
  aliases must not inherit a guessed tokenizer or silently truncated embedding.
- Retain bounded requests, original-source limits, subprocess wall deadlines,
  process-group termination, per-request private directories and cleanup.
- New dependencies remain optional and isolated. Core installation and tests
  remain offline and require no model credentials or downloads.

## RRC-06: verification and delivery

- Add meaningful regressions for new source projection, duplicate content,
  malformed model data, unknown/changed upstream source, incomplete pipelines,
  explicit fallback/error paths, bounds and feature permissions.
- Run real pinned GraphRAG/HippoRAG algorithms with trained BGE embeddings and
  clearly identified deterministic generation fixtures; that verifies execution
  and lineage, not learned generation quality or paper benchmark parity.
- Exercise actual HTTP and fixed subprocess boundaries, independent tenant
  credentials, update/delete and empty-snapshot isolation for added adapters.
- Record real execution timing, provider-call counts and runtime/artifact
  identities. Avoid comparing unlike workloads or claiming production capacity.
- Extend required CI jobs for provisionable new resources; selected resource
  tests must execute without skips. Run affected regressions and existing core,
  PostgreSQL, research, browser and packaging gates before merging.
- Update README, architecture and research ledger to distinguish implemented,
  actually executed, model-quality unverified and externally blocked work.
  Review changes independently and fix findings before final delivery.
