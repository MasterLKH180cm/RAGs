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
const document = {
  body: { dataset: { theme: "dark" } },
  hidden: false,
  querySelector: element,
  querySelectorAll: () => [],
  createElement: (name) => new Element(name),
  addEventListener() {},
};
const localStorage = {
  getItem: () => null,
  setItem: (name) => assert.equal(name, "rags-admin-theme", "Only theme may be persisted"),
};

let current = null;
let deferred = null;
let makeNextQueryDeferred = false;
let queryRequests = 0;
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
  citations: [{title: "SECRET " + identity, source: "private-" + identity,
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
  if (url.startsWith("/admin/api/documents?")) return reply(200, {
    total: 1, offset: 0, items: [{id: "secret-" + current, title: "Private " + current,
      source: "private-source", revision: "rev", updated_at: "2026-11-23T00:00:00Z"}],
  });
  if (url === "/admin/api/operations") return reply(200, {
    auth_mode: "api_key", storage: "sqlite", embedding_provider: "local",
    generation_provider: "extractive", permissions: ["rags:admin"], features: {},
  });
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
  const sensitive = ["PRIVATE A", "SECRET A", "Proprietary evidence of tenant A", "secret-A"];
  for (const value of sensitive) {
    for (const [key, node] of dom.entries()) {
      assert.equal(snapshot(node).includes(value), false,
        "Previous tenant data persisted in " + key + ": " + value);
    }
  }
}

await microtasks();
await login("A");
element("#query-question").value = "Confidential source request A";
await trigger("#query-form", "submit");
assert.match(element("#query-answer").textContent, /PRIVATE A/);
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
assert.ok(queryRequests >= 2);
console.log("Admin UI identity switch, DOM reset and stale request isolation: passed");
