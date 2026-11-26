/* Browser DOM smoke/regression without external packages.
   Exercises actual packaged client logic in a restricted fake DOM and mocked
   same-origin API. In particular, no prior tenant response may survive logout
   or an in-flight A->B identity transition. */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const dom = new Map();
const timers = new Set();

class Element {
  constructor(name = "") {
    this.name = name;
    this.textContent = "";
    this.children = [];
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.type = "text";
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.events = {};
    this.open = false;
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const yes = force === undefined ? !classes.has(name) : force;
        if (yes) classes.add(name);
        else classes.delete(name);
        return yes;
      },
    };
  }
  replaceChildren(...children) {
    this.textContent = "";
    this.children = [...children];
  }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, fn) { this.events[name] = fn; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  reset() {
    if (this.name === "#query-form") {
      element("#query-question").value = "";
      element("#query-generate").checked = false;
    }
    if (this.name === "#add-form") {
      for (const name of ["#add-title", "#add-content", "#add-id"]) element(name).value = "";
    }
    if (this.name === "#delete-form") element("#delete-confirm").value = "";
  }
}
function element(selector) {
  if (!dom.has(selector)) dom.set(selector, new Element(selector));
  return dom.get(selector);
}
const agentNav = element("#test-agents-nav");
agentNav.dataset.view = "agents";
const documentsNav = element("#test-documents-nav");
documentsNav.dataset.view = "documents";
const settingsNav = element("#test-settings-nav");
settingsNav.dataset.view = "settings";
const document = {
  body: { dataset: { theme: "dark" } },
  hidden: false,
  querySelector: element,
  querySelectorAll: (selector) => selector === ".nav-item" ? [agentNav, documentsNav, settingsNav] : [],
  createElement: (name) => new Element(name),
  addEventListener() {},
};
const localStorage = {
  getItem: () => null,
  setItem: (name) => assert.equal(name, "rags-admin-theme", "Only theme may be persisted"),
};

let current = null;
let deferredBootstrap, deferredLoginSession;
let initialSessionPending = true, firstLoginSessionPending = true;
let deferred = null;
let makeNextQueryDeferred = false;
let queryRequests = 0;
let deferDocuments = false;
const documentRequests = [];
let deferModels = false;
const modelRequests = [];
const modelCatalog = (identity) => ({
  embeddings: [{id: "bge-small-zh", label: "BGE 中文 <script>literal</script>", provider: "fastembed",
    model: "BAAI/bge-small-zh-v1.5", languages: ["zh"], dimensions: 512, max_input_tokens: 512,
    execution: "local", cost_category: "local_compute", input_policy: "exact", enabled: false,
    dependency: {installed: false}, limitations: ["Validate your corpus."]}],
  tokenizers: [
    {id: "auto", label: "Automatic"}, {id: "embedding", label: "Embedding counter"},
    {id: "unicode", label: "Characters"}, {id: "utf8_bytes", label: "UTF-8 bytes"},
    {id: "huggingface", label: "HF", requires_artifact: true},
  ],
  registered_artifacts: [{id: "zh-count", backend: "huggingface"}],
  approved_profiles: [{id: "model-profile-" + identity, model: "approved-" + identity}],
});
const session = () => ({
  csrf: "csrf-" + current,
  tenant: "tenant-" + current,
  permissions: ["rags:admin", "rags:read", "rags:query"],
  expires_at: "2027-01-01T00:00:00Z",
});
const reply = (status, body = {}) => ({
  status, ok: status >= 200 && status < 300,
  async json() { return body; },
});
const queryResult = (identity) => ({
  answer: "PRIVATE " + identity + " only",
  outcome: "supported",
  citations: [{title: "SECRET " + identity, chunk_id: "source-ID-" + identity, source: "private-" + identity,
    text: "Proprietary evidence of tenant " + identity, score: .987}],
});
async function fetch(url, options = {}) {
  const method = options.method || "GET";
  if (url === "/admin/api/session" && method === "POST") {
    const token = options.headers.Authorization.split(" ")[1];
    current = token.slice(-1);
    return reply(200, {status: "authenticated"});
  }
  if (url === "/admin/api/session" && method === "DELETE") {
    current = null;
    return reply(204);
  }
  if (url === "/admin/api/session" && method === "GET") {
    // Ignore abort deliberately: an already delivered anonymous response must
    // not invalidate an overlapping authenticated login in the newer epoch.
    if (initialSessionPending) {
      initialSessionPending = false;
      return new Promise(resolve => {
        deferredBootstrap = () => resolve(reply(401, {detail: {code: "session_required"}}));
      });
    }
    if (firstLoginSessionPending && current) {
      firstLoginSessionPending = false;
      return new Promise(resolve => { deferredLoginSession = () => resolve(reply(200, session())); });
    }
    return current ? reply(200, session()) : reply(401, {detail: {code: "session_required"}});
  }
  if (!current) return reply(401);
  if (url === "/admin/api/overview") return reply(200, {
    counts: {documents: 1, chunks: 3, revisions: 2, images: 0},
    activity: [{action: "upsert", document_id: "secret-" + current,
      timestamp: "2026-11-23T00:00:00Z"}],
    telemetry: {series: [], requests_5m: 1, errors_5m: 0, error_rate_5m: 0,
      otel: {enabled: false, exported: 0, dropped: 0, export_failures: 0}},
  });
  if (url.startsWith("/admin/api/documents?")) {
    if (deferDocuments) return new Promise(resolve => documentRequests.push({url, resolve}));
    return reply(200, {
      total: 1, offset: 0, items: [{id: "secret-" + current, title: "Private " + current,
        source: "private-source", revision: "rev", updated_at: "2026-11-23T00:00:00Z"}],
    });
  }
  if (url === "/admin/api/operations") return reply(200, {
    auth_mode: "api_key", storage: "sqlite", embedding_provider: "local",
    generation_provider: "extractive", permissions: ["rags:admin"], features: {},
    embedding: {provider: "local", model: "offline-hash-v1",
      readiness: {enabled: true, dependency: {installed: true}, runtime_validation: "not_run"}},
    source_tokenizer: {backend: "unicode", count_kind: "fallback_units"},
    generation_tokenizer: {backend: "utf8_bytes", count_kind: "conservative_bytes"},
  });
  if (url === "/admin/api/models") {
    assert.equal(method, "GET", "Model selection must never mutate live configuration");
    const identity = current;
    if (deferModels) return new Promise(resolve => modelRequests.push({identity, resolve}));
    modelRequests.push({identity});
    return reply(200, modelCatalog(identity));
  }
  if (url === "/admin/api/agents/capabilities") return reply(200, {
    models: [], tools: [], router_ready: false,
  });
  if (url === "/admin/api/agents/workflows") return reply(200, {
    templates: [{name: "evidence_review", title: "Evidence review",
      description: "Registered review graph", version: "v1",
      nodes: [
        {name: "retrieve", kind: "then", approval: false},
        {name: "quality_gate", kind: "branch", approval: false},
        {name: "cross_checks", kind: "parallel", approval: false},
        {name: "human_approval", kind: "then", approval: true},
        {name: "finalize", kind: "then", approval: false},
      ]}],
  });
  if (url === "/admin/api/agents/workflows/evidence_review/runs" && method === "POST") {
    return reply(201, {run_id: "workflow-" + current, kind: "workflow",
      revision: 3, status: "waiting_for_approval",
      state: {cursor: 3, input: {question: "private-" + current},
        outputs: {retrieve: {citations: [{chunk_id: "secret-" + current}]}}}});
  }
  if (url.startsWith("/admin/api/agents/workflows/runs/") && url.endsWith("/resume") &&
      method === "POST") {
    return reply(200, {run_id: "workflow-" + current, revision: 5,
      status: "completed", state: {cursor: 5,
        outputs: {retrieve: {citations: [{chunk_id: "secret-" + current}]}}}});
  }
  if (url === "/admin/api/query" && method === "POST") {
    queryRequests++;
    const identity = current;
    if (makeNextQueryDeferred) {
      makeNextQueryDeferred = false;
      return new Promise((resolve) => { deferred = () => resolve(reply(200, queryResult(identity))); });
    }
    return reply(200, queryResult(identity));
  }
  throw new Error("unexpected dashboard API request: " + method + " " + url);
}
const sandbox = {
  document, localStorage, fetch,
  URLSearchParams, AbortController, Intl, Date, Number, String, Object,
  Promise, console,
  setTimeout(fn) { const obj = {fn}; timers.add(obj); return obj; },
  clearTimeout(obj) { timers.delete(obj); },
  setInterval() { return 0; },
  window: {scrollTo() {}},
};
const source = fs.readFileSync(new URL("../rags/admin_static/app.js", import.meta.url), "utf8");
vm.runInNewContext(source, sandbox, {filename: "app.js", timeout: 2000});

async function microtasks() {
  for (let i = 0; i < 12; i++) await new Promise((resolve) => setImmediate(resolve));
}
async function trigger(selector, type = "click") {
  const handler = element(selector).events[type];
  assert.equal(typeof handler, "function", "Missing " + type + " on " + selector);
  await handler({preventDefault() {}});
  await microtasks();
}
async function login(identity) {
  element("#admin-token").value = "admin-" + identity;
  await trigger("#login-form", "submit");
  assert.equal(element("#workspace").hidden, false);
  assert.equal(element("#tenant-sidebar").textContent, "tenant-" + identity);
}
function snapshot(node, seen = new Set()) {
  if (!node || typeof node !== "object" || seen.has(node)) return "";
  seen.add(node);
  return String(node.textContent || "") + " " + String(node.value || "") + " " +
    (node.children || []).map((c) => snapshot(c, seen)).join(" ");
}
function assertNoOldTenant() {
  const sensitive = ["PRIVATE A", "SECRET A", "Proprietary evidence of tenant A", "secret-A",
    "source-ID-A", "workflow-A", "model-profile-A", "approved-A"];
  for (const value of sensitive) {
    for (const [key, node] of dom.entries()) {
      assert.equal(snapshot(node).includes(value), false,
        "Previous tenant data persisted in " + key + ": " + value);
    }
  }
}

await microtasks();
element("#admin-token").value = "admin-A";
const overlappingLogin = element("#login-form").events.submit({preventDefault() {}});
await microtasks();
assert.equal(typeof deferredBootstrap, "function");
assert.equal(typeof deferredLoginSession, "function");
deferredBootstrap();
await microtasks();
deferredLoginSession();
await overlappingLogin;
await microtasks();
assert.equal(element("#workspace").hidden, false, "Late bootstrap 401 must not undo a newer login");
assert.equal(element("#tenant-sidebar").textContent, "tenant-A");
assert.equal(element("#login-error").hidden, true);
assert.equal(element("#login-submit").disabled, false, "Successful login must restore its submit control");
await trigger("#logout-button");
assert.equal(element("#login-submit").disabled, false, "Logout must permit another login");

await login("A");
element("#query-question").value = "Confidential source request A";
await trigger("#query-form", "submit");
assert.match(element("#query-answer").textContent, /PRIVATE A/);
assert.match(snapshot(element("#query-citations")), /\[source-ID-A\]/,
  "Evidence card must show the exact answer citation ID");

assert.equal(element("#query-output").hidden, false);
await trigger("#logout-button");
assert.equal(element("#workspace").hidden, true);
assertNoOldTenant();
await login("B");
assert.equal(element("#query-output").hidden, true);
assertNoOldTenant();

await trigger("#logout-button");
await login("A");
makeNextQueryDeferred = true;
element("#query-question").value = "Another confidential A query";
const hanging = element("#query-form").events.submit({preventDefault() {}});
await microtasks();
assert.equal(typeof deferred, "function", "Expected in-flight query");
await trigger("#logout-button");
await login("B");
deferred(); // Mock ignores abort: epoch check must still reject the late response.
await hanging;
await microtasks();
assertNoOldTenant();
assert.equal(element("#query-output").hidden, true);
assert.equal(element("#query-submit").disabled, false);
assert.equal(element("#add-submit").disabled, false);
assert.equal(element("#query-submit").textContent, "開始檢索 ↗");
assert.ok(queryRequests >= 2);
await trigger("#test-agents-nav");
await microtasks();
assert.equal(element("#workflow-template").children.length, 1);
assert.equal(element("#workflow-graph").children.length, 5);
element("#workflow-template").value = "evidence_review";
element("#workflow-question").value = "Records policy evidence";
element("#workflow-mode").value = "sparse";
element("#workflow-topk").value = "3";
await trigger("#workflow-form", "submit");
assert.equal(element("#workflow-approval").hidden, false);
assert.equal(element("#workflow-run-id").textContent, "執行 ID：workflow-B");
assert.equal(element("#workflow-graph").children.length, 5);
await trigger("#workflow-approve");
assert.equal(element("#workflow-approval").hidden, true);
assert.equal(element("#workflow-status").textContent, "completed");
await trigger("#logout-button");
assertNoOldTenant();
await login("A");
assert.equal(element("#workflow-actions").hidden, true);
assert.equal(element("#workflow-graph").children.length, 0,
  "Workflow graph must not survive switching identity");
// Deliberately deliver document requests out of order within one login. A
// previous page's success or failure must not replace the latest page.
deferDocuments = true;
for (const olderStatus of [200, 500]) {
  const first = documentRequests.length;
  await trigger("#test-documents-nav");
  await trigger("#page-next");
  const [older, latest] = documentRequests.slice(first);
  assert.ok(older && latest);
  const offset = Number(new URLSearchParams(latest.url.split("?")[1]).get("offset"));
  latest.resolve(reply(200, {total: 80, offset, items: [
    {id: "latest-page", title: "Latest page", source: "manual", revision: "rev"},
  ]}));
  await microtasks();
  const visible = snapshot(element("#documents-body"));
  const range = element("#document-range").textContent;
  assert.match(visible, /latest-page/);
  older.resolve(reply(olderStatus, {total: 80, offset: 0, items: [
    {id: "obsolete-page", title: "Obsolete page", source: "manual", revision: "rev"},
  ]}));
  await microtasks();
  assert.equal(snapshot(element("#documents-body")), visible,
    "An obsolete response must not replace the latest document page or its error state");
  assert.equal(element("#document-range").textContent, range);
}
deferDocuments = false;
// Selection preview uses the shared catalog, distinguishes configured from
// validated, and performs only read requests even when users change choices.
await trigger("#test-settings-nav");
assert.match(element("#setting-embedding").textContent, /offline-hash-v1/);
assert.match(element("#setting-model-readiness").textContent, /尚未執行模型驗證/);
assert.match(snapshot(element("#model-choice")), /<script>literal<\/script>/,
  "Labels are displayed as literal text, never interpreted as HTML");
assert.equal(element("#generation-tokenizer-choice").children.some(c => ["auto", "embedding"].includes(c.value)), false);
assert.equal(element("#source-tokenizer-choice").children.some(c => c.value === "huggingface:zh-count"), true);
const beforePreview = modelRequests.length;
element("#model-choice").value = "bge-small-zh";
await trigger("#model-choice", "change");
element("#generation-tokenizer-choice").value = "utf8_bytes";
await trigger("#generation-tokenizer-choice", "change");
assert.match(element("#model-selection-command").textContent, /rags config plan.*--embedding 'bge-small-zh'/);
assert.match(element("#model-selection-command").textContent, /--generation-tokenizer 'utf8_bytes'/);
assert.match(snapshot(element("#model-selection-notes")), /目前部署已停用/);
assert.match(snapshot(element("#model-selection-notes")), /重新產生向量/);
assert.equal(modelRequests.length, beforePreview, "Changing a preview has no API side effects");

// A catalog loaded under the previous tenant must never repopulate choices or
// approved profile names after logout/login, even if its response arrives late.
deferModels = true;
await trigger("#test-settings-nav");
const pendingModel = modelRequests.at(-1);
assert.equal(typeof pendingModel.resolve, "function");
await trigger("#logout-button");
assert.equal(element("#model-choice").children.length, 0);
assert.equal(element("#model-choice").disabled, true);
assert.equal(element("#approved-model-profiles").children.length, 0);
await login("B");
pendingModel.resolve(reply(200, modelCatalog("A")));
await microtasks();
assertNoOldTenant();
assert.equal(element("#model-choice").children.length, 0);
await trigger("#logout-button");
console.log("Admin UI selection preview, identity switch and stale request isolation: passed");
