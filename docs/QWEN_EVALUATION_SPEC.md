# Qwen evaluation attribution

## Problem and scope

The frozen Qwen answerability evaluation counts a refusal of any dataset-labelled
answerable question as a `false_refusal`. This remains a useful end-to-end metric,
but it does not identify whether retrieval returned a labelled reference document
before admission. A refusal after a retrieval miss must not be presented as proof
that the admission model rejected available reference evidence.

This change adds document-ID diagnostics to the offline evaluation only. It does
not change retrieval, admission decisions, existing result fields, quality gates,
frozen profiles, thresholds, datasets, or reports. It requires no model inference
and adds no source text to a report. Historical reports remain immutable; a new
full benchmark produces a new report containing the additional diagnostics.

## Per-case contract

Each evaluated measurement gains `reference_diagnostics` with these fields:

| Field | Meaning |
|---|---|
| `reference_document_ids` | Sorted, unique document IDs from the dataset's `relevant_document_ids` labels. |
| `raw_reference_document_ids` | Unique reference IDs present in the existing raw candidate list, in retrieval order. |
| `admitted_reference_document_ids` | Unique reference IDs present in the existing returned-citation list, in citation order. |
| `missing_raw_reference_document_ids` | Sorted reference IDs absent from the raw candidate list. |
| `removed_reference_document_ids` | Reference IDs present in the raw candidate list but absent from returned citations, in retrieval order. |
| `raw_reference_coverage` | `not_applicable` for no reference labels; otherwise `none`, `partial`, or `complete` according to ID coverage. |
| `reference_removal_cause` | `null` when no reference IDs were removed; otherwise the bounded attribution below. |

The comparison uses the same unique document lists, capped at the evaluation's
`k`, as the existing raw/admitted retrieval metrics. Multiple chunks from one
document count once. A partial set of labelled documents is reported explicitly.
Cases with no reference labels are not counted as retrieval misses.

Raw candidates are recorded before admission and generation context packing.
Consequently, a loss of reference IDs is attributed to `admission_refusal` only
when the final outcome is `insufficient_evidence` with refusal reason
`below_admission_threshold` or `model_rejected`. The same outcome with reason
`context_budget_exhausted` is attributed to `context_budget`. Other losses are
`unattributed`; a difference between raw and final IDs alone cannot establish its
cause. Existing outcome, refusal reason, admission basis, and budget fields remain
available alongside these diagnostics.

`admitted_reference_document_ids` describes returned citations, consistent with
the existing `admitted_evidence` metrics. It does not certify that a generation
prompt included every cited chunk. These diagnostics measure document identity,
not relevant text spans, semantic answerability, evidence sufficiency, entailment,
or whether several passages together support an answer. A complete set of
reference IDs is not called sufficient evidence.

## Per-mode aggregate

Each mode report gains `reference_diagnostics` containing:

- `scope = "document_id_coverage_only"` and the number of evaluated `cases`.
- `raw_reference_coverage`: case counts for all four coverage categories.
- `answerable_refusals_by_raw_reference_coverage`: counts of refusals for
  dataset-labelled answerable questions, partitioned by those same categories.
  The sum equals the existing `false_refusals` count; neither is redefined.
- `reference_removals`: the number of affected cases and removed
  `document_occurrences`, plus both counts grouped by `admission_refusal`,
  `context_budget`, and `unattributed`. An occurrence is counted once per case;
  the same reference removed for two different questions contributes two.

All aggregate categories are present with zero counts when empty. Failed or
inapplicable engine executions do not contribute synthetic diagnostic rows.

## Verification

Tests cover no labels, no reference hit, partial and complete reference coverage,
duplicate chunks, preserved references, admission refusal, context-budget
refusal, and loss without an attributable refusal. A real offline Store and
RAGService path with a deterministic scorer fixture verifies that the existing
false-refusal metric and the new decomposition agree for both generation modes.
Fixture scores test reporting/control flow, not model quality.

A read-only regression test recomputes diagnostics from the existing frozen Qwen
report and dataset. For each hybrid generation mode, it must identify three raw
retrieval misses (`semantic-01`, `semantic-03`, `semantic-06`), two answerable
refusals without a retrieved reference (`semantic-03`, `semantic-06`), and three
admission refusals that remove retrieved references (`semantic-05`,
`multi_hop-01`, `multi_hop-02`). Those last three remove five document occurrences.
No frozen evidence is rewritten or used to change a scoring decision.

## Separate model-quality work

The frozen Qwen scorer asks whether each individual passage answers the entire
question, then takes the maximum passage score. Supporting comparisons across
multiple passages requires a separately specified joint-evidence assessor and
new profile identity. Selecting a new model, prompt, aggregation, or threshold
requires calibration-only development and an independent heldout evaluation for
promotion. This reporting improvement does not implement that assessor or change
the frozen profile's failed quality status.
