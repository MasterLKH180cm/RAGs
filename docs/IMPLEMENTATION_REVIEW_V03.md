# RAGs v0.3 implementation and evidence review

This iteration follows [SPEC_V03.md](SPEC_V03.md): specification and fixed data,
implementation, actual execution, independent review, corrections, and evidence
publication. It extends the historical [v0.2 review](IMPLEMENTATION_REVIEW.md).
The twenty-family table remains a coverage ledger, not a claim that every paper
has been reproduced.

The three README flowcharts passed actual Mermaid11.12.1 parsing with
jsdom26.1.0. The [diagram record](diagram-validation-v03.json) fingerprints the
tested diagram sources. This is syntax validation, not visual browser rendering.

## Changes and acceptance evidence

| Requirement | Implemented behavior | Concrete evidence |
| --- | --- | --- |
| R3-01 query decisions | `outcome`, `supported`, finite `admission` and fixed refusal reasons in both generation modes; refused candidates are diagnostic only | [Contract tests](../tests/test_admission.py), [Agent E2E](https://github.com/MasterLKH180cm/agentic-system/blob/feat/spec-driven-v02/tests/e2e_v03.py) |
| R3-02 profiles | Actual BGE relevance and pinned Qwen answerability inference; artifact/runtime/prompt/window identity; required-policy enforcement; bounded inputs and explicit truncation | [Admission](../rags/admission.py), [Qwen scorer](../rags/answerability.py), [window tests](../tests/test_answerability.py), immutable profiles and reports below |
| R3-03 exact/federated evidence | Complete dataset provenance for exact zero, finite aggregates, maximum20 datasets; refused federated sources cannot become citations | Contract and native integration tests |
| R3-04 measured quality | Frozen48 calibration and32 held-out questions; original36-case regression; raw and admitted metrics; no threshold changes after evaluation | [Verifier](../rags/verify_admission.py), [Qwen report](../reports/admission-verification-qwen-v03.json), frozen artifacts below |
| R3-05 actual upstreams | Source-pinned RAPTOR clustering/tree retrieval and LightRAG graph extraction/hybrid retrieval in fixed request-local processes | [RAPTOR algorithm report](../reports/research_raptor_v03.json), [LightRAG algorithm report](../reports/research_lightrag_v03.json) |
| R3-06 real boundaries | Real TCP RAG/sidecar HTTP, tenant isolation, source updates/as-of/deletion, actual timeout cleanup, independent Agent worker and terminal SSE/MCP | [RAPTOR HTTP chain](../reports/research_raptor_http_v03.json), [LightRAG HTTP chain](../reports/research_lightrag_http_v03.json) |

The actual upstream tests use trained BGE embeddings. Summary/entity extraction
uses an explicitly deterministic local generation fixture. The HTTP path still
executes the production provider adapter, authenticated sidecar, fixed worker,
real upstream algorithm, source mapping and Agent runtime. This establishes
those execution paths; it does not measure live trained-generator quality.

## Frozen scoring artifacts and evaluation sequence

The original specification and both datasets were committed before model
scoring. Calibration labels, expected answers, rationales and case IDs never
enter a scorer input. Calibration chooses a threshold using the specified
positive-retention constraint and deterministic tie rule. The profile covers
the resulting threshold and the complete scorer identity.

| Candidate | Frozen profile | Calibration evidence |
| --- | --- | --- |
| BGE multilingual relevance | [admission-bge-v03.json](../profiles/admission-bge-v03.json) | [Calibration](../reports/admission-calibration-v03.json):23/24 positives retained,1/24 negatives refused,11/12 semantic-only positives retained |
| Qwen3-1.7B answerability | [admission-qwen-v03.json](../profiles/admission-qwen-v03.json) | [Calibration](../reports/admission-calibration-qwen-v03.json):23/24 positives retained,18/24 negatives refused,11/12 semantic-only positives retained |

BGE's poor negative rejection motivated the Qwen specification amendment using
calibration results only. An initial BGE held-out command completed before a
stop instruction arrived; its report remained unopened until the Qwen profile
was frozen. That report was not used to choose Qwen, the prompt, threshold or
acceptance criteria. Both profiles and calibration reports have separate
commits before final held-out inspection. The [BGE held-out report](../reports/admission-holdout-v03.json)
is retained as a failed-candidate comparison, not discarded.

The Qwen prompt, pinned model, label tokens, CPU dtype/runtime, complete1024-token
window and threshold remain unchanged during final evaluation. The maximum of
single-passage answerability scores has a known multi-source limitation: no
single source may answer an entire multi-hop question. Preserving the original
ranking on acceptance does not remove that limitation. The frozen fixtures are
single-passage cases; cross-source and lifecycle behavior is separately tested.

### Final measured result: quality gates failed

The [frozen Qwen verification report](../reports/admission-verification-qwen-v03.json)
records **21/21 correctness checks passed and 7/11 quality checks passed**.
Enforced verification exited2 with `promotion_status: failed`. No profile,
prompt, fixture, threshold or acceptance target was changed to improve these results.

| Gate | Result | Decision |
| --- | --- | --- |
| Held-out answerable retention | 15/16; target at least15 | Pass |
| Held-out no-answer rejection | 9/16; target at least13 | Fail |
| Held-out semantic-only retention | 7/8; target8 | Fail |
| Hybrid raw versus admitted Recall@3 | 0.85 versus0.70 in each generation mode | Fail in both modes |
| Original no-answer rejection | 4/4 in each generation mode; target at least3 | Pass |
| Exact structured and temporal regression | 6/6 and4/4 in each generation mode | Pass |

Five of20 answerable hybrid cases were refused. Two had no relevant raw
candidate; the admission gate newly loses useful evidence for `semantic-05`,
`multi_hop-01` and `multi_hop-02`. The held-out false accepts comprise four
missing-field and three wrong-entity cases. The failed semantic paraphrase asks
for a digitized film copy. These failures remain visible in the report.

The verifier runs34 unique applicable questions from the original36-case corpus.
The two graph-only cases remain outside the unchanged hybrid/structured/temporal
mode labels. Per-mode totals overlap. All290 returned citation records passed
record/provenance checks; this does not establish semantic entailment.

The [actual Qwen HTTP-to-Agent report](../reports/agent_admission_http_v03.json)
uses two frozen calibration fixtures solely for the integration contract. The
accepted fixture scores22.875, completes the Agent run, and makes one actual
local fixture-provider synthesis request with one usage record. The missing-field
fixture scores-9.75, persists `insufficient_evidence` and a failed run, and makes
zero synthesis requests or reservations. Required-policy profile snapshots,
source revisions, REST/SSE, both MCP bridges and safe events are verified.
This adds no held-out quality evidence and makes no live-generator quality claim.

### Reproduce the measured gates

```bash
bash scripts/install-answerability.sh ../.venv-admission
# Explicitly download public pinned artifacts before serving requests:
RAGS_MODEL_CACHE="$PWD/data/models" ../.venv-admission/bin/python - <<'PY'
import os
from rags.admission import load_profile
from rags.answerability import AnswerabilityScorer
scorer = AnswerabilityScorer(os.environ["RAGS_MODEL_CACHE"], allow_download=True)
scorer.verify(load_profile("profiles/admission-qwen-v03.json"))
PY
../.venv-admission/bin/python -m rags.verify_admission \
  --profile profiles/admission-qwen-v03.json \
  --holdout examples/admission_holdout_v03.json \
  --regression examples/eval_v02.json \
  --cache-dir data/models \
  --report reports/my-frozen-profile-verification.json --enforce-quality
```

The report destination must be new; the verifier refuses to overwrite evidence.
Default enforcement exits0 only when correctness and quality both pass,1 for an
execution/contract failure, and2 for a quality failure. The optional
`--no-enforce-quality` mode checks runtime correctness while retaining every
failed quality target and `promotion_status:failed`. Ordinary CI instead runs
the small actual-model contract and recorded-evidence audit described below.
Its green runtime job does not approve a failed profile for deployment. No
scoring profile is enabled by default.

### CI resource boundary and recorded evidence

The complete local benchmark performed 472 forward passes: 32 held-out passages
and 440 candidate scores across regression modes. The actual Agent CI contract
took 140.627 seconds for six scorer calls, versus 18.606 seconds locally. Its
logits were 22.625 and -9.5, compared with local 22.875 and -9.75. The frozen
threshold 10.75 produced the same two decisions. Runtime and artifact identity
does not promise bitwise numerical equality or identical speed across CPUs.

The [first PR full replay](https://github.com/MasterLKH180cm/RAGs/actions/runs/44862953958)
was superseded before completing its model evaluation. It supplies no new full
quality measurement. A committed [specification amendment](SPEC_V03.md#9-ci-resource-and-evidence-amendment)
then separated ordinary CI from the expensive full benchmark without changing
the model, threshold, fixtures or quality targets.

Ordinary CI executes the same two preselected calibration contract cases with
the actual pinned scorer. The [runtime/evidence helper](../scripts/verify_admission_runtime.py)
also verifies SHA-256 values for the frozen full report, profile, calibration,
holdout and original regression files; checks row coverage and dataset/profile
identity; and recomputes every recorded correctness and quality gate. Its
`recorded_quality` object is explicitly historical. Audit success is not a
new quality result and cannot promote the failed profile. The pure evidence
audit can be reproduced without model inference:

```bash
python scripts/verify_admission_runtime.py --audit-only
```

The manual `full_admission_benchmark` workflow option retains the complete
`--enforce-quality` command and all original gates, allows up to 360 CPU minutes,
and uploads the result even when quality enforcement fails. A scoring/retrieval
change or a promotion proposal requires a fresh complete benchmark. No automatic
retries, profile selection or threshold changes are introduced.

## Review findings and corrections

| Concrete finding | Correction |
| --- | --- |
| RAG could return ranked citations while declaring insufficient evidence | Centralized typed admission removes usable citations and skips later generation in both query modes |
| A required calibrated gate could be bypassed by a positive model critic | Required profile validation and scoring precede acceptance; separate model review cannot waive the gate |
| A critic removing every candidate could lose the explicit refusal reason | Model rejection retains priority over a plain empty-corpus reason |
| Invalid/contradictory remote assessments could become fused evidence | Strict enums, counts, profile fields, finite scores and refusal consistency before federation |
| An exact zero could be confused with absent source scope | Nonempty bounded dataset provenance and strict finite aggregate validation |
| Concurrent cold requests could create duplicate large model instances | Process-wide cache construction lock plus per-model inference serialization |
| Upstream libraries could leak progress/source text through stdout/errors | Worker redirects non-protocol output; service emits fixed bounded error codes |
| Large worker output could fill a temporary spool before checking size | Bounded incremental stdout pipe; terminate on the first over-limit read |
| Killing an over-limit child could deadlock while its stdout reader was paused | Kill all process-group writers, drain remaining pipes with `communicate`, then reap; regression forces pipe backpressure and bounds cleanup |
| A worker leader could exit while leaving a descendant process | Always terminate the process group, including after leader completion; real fork/process checks |
| Per-read HTTP timeouts could exceed the declared total budget | One asyncio deadline spans connect, headers and body; actual slow-header/stalled-body TCP regression |
| Very large JSON integers could escape numeric validation | Explicit overflow rejection before scores become usable evidence |
| Native embedding truncation could be hidden behind full-source citations | Count the actual encoder input window and reject oversize inputs before encoding |
| RAPTOR's small-sample UMAP neighbor count can be invalid | Use the upstream stop condition for small leaf sets; test both clustering and small-input behavior |
| A configured URL or empty health probe could be described as verified upstream execution | Catalog distinguishes configuration, upstream-reported metadata and separate actual execution evidence |

The companion Agent review covers refused-source quarantine, exact-observation
packing, old-checkpoint re-evaluation, complete RAG deadlines, durable failure,
and terminal REST/SSE/MCP consistency. Provider bodies and source/query text are
not copied into the new event metadata.

Cross-review also rejected malformed federated citation provenance and legacy
support values other than boolean/null before fusion. The final local suite
passed280 tests with5 opt-in model/environment skips; those external paths have
separate actual execution evidence. Ruff and `git diff --check` passed.

## Remaining boundaries

- Microsoft GraphRAG, HippoRAG, trained reflection-token Self-RAG and paper-specific
  CRAG remain unimplemented upstream integrations. Their concrete prerequisites
  and acceptance evidence are in [RESEARCH_ENGINES.md](RESEARCH_ENGINES.md).
- Real BGE/upstream execution does not establish live summarization, extraction,
  answer generation or production domain quality.
- Frozen synthetic quality targets are retained even when a profile fails.
  Citation-record integrity is distinct from answer entailment.
- Research indexes are rebuilt per request, prioritizing authorized revision
  isolation over throughput. They are not a distributed incremental index.
- SQLite, full vector scans, static tenant credentials, explicit revision updates
  and separate image retrieval retain the v0.2 deployment/media limits.
