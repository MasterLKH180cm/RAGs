# User-oriented README and whole-repository maintainability review

Status: specification committed before implementation.
Baseline: `6b1f97357ce934a75bfae3102ef8e72512d3da16`.
Requested on 2026-11-26: make the README user friendly and conduct a complete
code bad-smell review of the current RAGs repository.

## Outcomes

1. A reader can choose between collection records, chunked documents and agents,
   install the application, and obtain a first authenticated result without
   configuring a paid model or reading historical implementation notes.
2. Existing operational, authentication, model, migration and quality information
   remains available in linked documentation. Repository links and anchors remain
   valid after moving material out of the README.
3. A review report records the covered files, concrete findings, severity,
   evidence, fixes and deliberately retained architecture tradeoffs. Static
   complexity counts guide inspection; they do not establish correctness.
4. Confirmed defects and consequential maintainability problems receive bounded
   fixes and relevant regression coverage. Changes remain reviewable in one PR,
   with the existing complete-package and actual-resource CI gates preserved.

## README information architecture

Keep the public README in the repository's existing English language. Lead with
what the service does and which interface to use. Put a credential-free local
installation and a working first result before optional model instructions.
Make required versus optional components explicit and use the implemented CLI
and SDK rather than hypothetical package/distribution commands.

Provide a short guide map for collections, documents, model providers, agents,
the dashboard, production PostgreSQL, authentication, migration, testing and
research engines. Keep quality limitations next to the capability they qualify.
Move detailed reference/history out of the entry page with correct relative
links. Preserve or update inbound anchors. Explain expected output and common
startup/authentication/provider/index problems in user terms.

Verify installation/import and the actual README onboarding path in an isolated
directory, including persistent data and an authenticated query. Check internal
Markdown links and fragments. Do not introduce a separate test for every prose
change; cover executable contracts and broken-navigation risks.

## Review coverage

Review all production Python modules under `rags/`, the shipped administrative
HTML/CSS/JavaScript, executable setup/benchmark/E2E scripts, CLI templates,
database/deployment configuration, workflows and test-contract infrastructure.
Read associated tests for each finding and inspect the test suite for isolation,
fixtures and assertions that could conceal an actual defect. Upstream vendored
model libraries and downloaded weights are outside this repository review.

Areas of emphasis:

- Tenant/owner authorization and feature checks before storage or provider work.
- Duplicated contracts and invariants that can drift across transports or stores.
- Excessive responsibilities, nonlocal control flow and avoidable coupling.
- Hidden configuration/import effects and lifecycle/resource ownership.
- Exception handling, cancellation, timeout and bounded input/output behavior.
- Atomicity, concurrent updates, immutable source references and migration.
- Provider/model identity, exact source preservation and honest capability output.
- Validation quality, test independence, explicit resource selection and CI gates.
- Documentation examples, packaging and deployment configuration consistency.

Classify confirmed behavior faults separately from maintainability risks. Every
actionable finding needs a concrete location, triggering condition or change
scenario, impact and a narrowly justified remedy. A long function, a broad
exception handler at a process boundary or a dictionary alone is not sufficient
evidence of a defect. Retain intentional boundaries where abstraction would hide
semantics or add unnecessary indirection.

## Implementation boundaries

Preserve public routes and SDK/CLI contracts unless a demonstrated bug requires
an explicit documented correction. Any shared-contract extraction must preserve
authorization order, unknown-route rejection, defaults, errors and API docs.
Keep provider execution outside write transactions and preserve exact original
source citations, record versions and tenant/owner scope.

Do not change retrieval/scoring formulas, frozen Qwen profiles/datasets/results,
quality targets or promotion status as part of cosmetic cleanup. Do not lower
coverage floors, add exclusions, weaken resource gates or relabel failures.
Router, agent and memory defaults stay enabled; external calls still require the
configured provider/gateway and appropriate permissions.

Do not perform repository-wide formatting or replace working algorithms merely
to improve a complexity statistic. Prefer cohesive helpers, explicit contracts
and lifecycle ownership only where they resolve a demonstrated problem.

## Validation and delivery

Run focused regressions for each behavior fix, required static checks, explicit
unit/integration suites, actual PostgreSQL/pgvector tests, relevant HTTP/CLI
onboarding and the complete-package line/branch coverage gate. The ordinary
offline suites must not acquire model credentials or make external model calls.
Retain the actual encoder/research/browser/deployment CI jobs and verify the
tested PR head before merge, then verify the merged master run.

Deliver the revised README, a complete review report with file coverage and
residual observations, reproducible validation results, and the PR/CI links.
Any GitHub comment, if one is needed, must contain exactly `@codex review`.
