/* Resume Depth — frontend logic. SSE contract, event names and payload shapes
   are unchanged from the backend (app.py): step, role, check, done, error. */

/* Where the FastAPI backend lives. config.js sets window.API_BASE when the page
   is hosted apart from the API; left unset it stays empty and every call is
   same-origin, which is what `uvicorn app:app` serves locally. */
const API = String(window.API_BASE || "").replace(/\/+$/, "");

const $ = (id) => document.getElementById(id);
const ORDER = ["experience", "evidence", "growth", "completeness"];

const META = {
  experience: {
    icon: "ph-briefcase", color: "var(--c-experience)",
    why: "The system reads the project content to see what work they have really done, and the dates to see how long they did it for.",
  },
  evidence: {
    icon: "ph-seal-check", color: "var(--c-evidence)",
    why: "We check that the skills they claim actually show up in the work they described.",
  },
  growth: {
    icon: "ph-trend-up", color: "var(--c-growth)",
    why: "Do the job titles rise over time? Junior to Senior to Lead shows someone trusted with more. A flat line for eight years does not.",
  },
  completeness: {
    icon: "ph-list-checks", color: "var(--c-completeness)",
    why: "Are the basic parts there - a summary, a skill list, education, certificates. Missing pieces cost the candidate nothing to fix.",
  },
};
ORDER.forEach((k) => (META[k].title = k[0].toUpperCase() + k.slice(1)));

const TONE = {
  pass: { cls: "strong", label: "Strong", icon: "ph-check-circle", color: "var(--strong)", tint: "var(--strong-t)", border: "var(--strong-b)" },
  warn: { cls: "needs", label: "Needs work", icon: "ph-warning-circle", color: "var(--needs)", tint: "var(--needs-t)", border: "var(--needs-b)" },
  fail: { cls: "weak", label: "Weak", icon: "ph-x-circle", color: "var(--weak)", tint: "var(--weak-t)", border: "var(--weak-b)" },
  error: { cls: "neutral", label: "Skipped", icon: "ph-minus-circle", color: "var(--faint)", tint: "var(--paper-2)", border: "var(--line)" },
};

// A backwards timeline used to come back tagged "rising" under a trend-up
// arrow. ground_growth() can now return "declining", so the icon has to follow
// the word rather than always pointing up.
const TRAJECTORY_ICON = {
  rising: "ph-trend-up",
  declining: "ph-trend-down",
  steady: "ph-arrow-right",
  flat: "ph-arrow-right",
  unclear: "ph-question",
};

const STRENGTH = {
  strong: { key: "proven", cls: "chip-proven", label: "Proven", icon: "ph-check" },
  moderate: { key: "mentioned", cls: "chip-mentioned", label: "Mentioned", icon: "ph-check" },
  claimed: { key: "unproven", cls: "chip-unproven", label: "Unproven", icon: "ph-x" },
};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* wraps quoted resume phrases in <mark>, escaping every segment independently */
function highlightQuotes(raw) {
  const parts = String(raw ?? "").split(/"([^"]{3,160})"/);
  return parts.map((part, i) => (i % 2 === 1 ? `&ldquo;<mark>${esc(part)}</mark>&rdquo;` : esc(part))).join("");
}

function show(id) {
  ["viewUpload", "viewReport", "viewError"].forEach((v) => $(v).classList.toggle("hidden", v !== id));
}

/* keeps the sticky rail pinned directly under the sticky nav + report topbar,
   whose combined height varies with content (e.g. long role names wrapping) */
function syncStickyOffsets() {
  const nav = document.querySelector(".nav");
  const topbar = document.querySelector(".report-topbar");
  const root = document.documentElement.style;
  if (nav) root.setProperty("--nav-h", nav.offsetHeight + "px");
  if (topbar) root.setProperty("--topbar-h", topbar.offsetHeight + "px");
}
window.addEventListener("resize", syncStickyOffsets);
window.addEventListener("load", syncStickyOffsets);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(syncStickyOffsets);

const CIRC = 2 * Math.PI * 59;

/* ---------------- backend banner ---------------- */
fetch(API + "/api/backend").then((r) => r.json()).then((b) => {
  $("backendNote").innerHTML = b.local
    ? `Analysed locally by <strong>${esc(b.model)}</strong> &mdash; nothing leaves this machine.`
    : `Analysed by <strong>${esc(b.model)}</strong> on <strong>${esc(b.host)}</strong> &mdash; your resume is sent there and is not stored after the report is generated.`;
  $("navSub").textContent = `${b.model} · ${b.host}`;
}).catch(() => {});

/* ---------------- file pickup ---------------- */
let currentFile = null;

const drop = $("drop");
drop.onclick = () => $("file").click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0]);
};
$("file").onchange = (e) => { if (e.target.files[0]) analyse(e.target.files[0]); };
$("btnSelect").onclick = (e) => e.stopPropagation();

$("btnNew").onclick = () => location.reload();
$("btnNew2").onclick = () => location.reload();
$("btnBack").onclick = () => location.reload();
$("btnPrint").onclick = () => window.print();
$("btnExport").onclick = () => window.print();
$("btnExport2").onclick = () => window.print();
$("btnHowItWorks").onclick = () => {
  const grid = document.querySelector(".preview-grid");
  if (grid) grid.scrollIntoView({ behavior: "smooth", block: "start" });
};

/* ---------------- pipeline ---------------- */
function buildPipeline() {
  const el = $("pipeline");
  el.innerHTML = ORDER.map((k, i) => `
    ${i > 0 ? `<div class="rail-connector" id="pc-${k}"><i></i></div>` : ""}
    <button class="rail-node" id="pl-${k}" type="button" data-state="waiting"
            style="--node-color:${META[k].color}" data-key="${k}">
      <span class="rail-node-row">
        <span class="rail-tile"><i class="ph ${META[k].icon}" aria-hidden="true"></i></span>
        <span class="rail-info">
          <span class="rail-name">${META[k].title}</span>
          <span class="rail-state"><span class="txt">waiting</span></span>
        </span>
      </span>
    </button>`).join("");

  el.querySelectorAll(".rail-node").forEach((btn) => {
    btn.onclick = () => {
      const row = $("row-" + btn.dataset.key);
      if (!row) return;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.classList.add("flash");
      setTimeout(() => row.classList.remove("flash"), 1300);
    };
  });
}

function setNodeState(key, state, score, status) {
  const node = $("pl-" + key);
  if (!node) return;
  node.dataset.state = state;
  const txt = node.querySelector(".rail-state .txt");
  if (state === "waiting") txt.textContent = "waiting";
  if (state === "running") txt.textContent = "running";
  if (state === "done") {
    const tone = TONE[status] || TONE.warn;
    node.style.setProperty("--verdict-color", tone.color);
    txt.innerHTML = `<span class="rail-score num">${score}</span>`;
  }
  if (state === "running") {
    const connector = $("pc-" + key);
    if (connector) requestAnimationFrame(() => (connector.querySelector("i").style.height = "100%"));
  }
}

/* ---------------- rows ---------------- */
function buildRows() {
  $("rows").innerHTML = "";
}

/* only the currently-running check's row exists in the DOM — queued checks
   have no visible loader until it is actually their turn */
function addRow(k) {
  const idx = ORDER.indexOf(k) + 1;
  const holder = document.createElement("div");
  holder.innerHTML = `
    <div class="row" id="row-${k}" data-key="${k}" style="--row-color:${META[k].color}">
      <div class="card-head">
        <div class="card-head-left">
          <span class="icon-tile" style="--card-color:${META[k].color}"><i class="ph ${META[k].icon}" aria-hidden="true"></i></span>
          <div class="card-head-text">
            <span class="card-name">${META[k].title}</span>
            <span class="card-eyebrow">Check 0${idx} of 0${ORDER.length}</span>
          </div>
        </div>
        <div class="card-head-right" id="chr-${k}">${leftSkeleton(k)}</div>
      </div>
      <div class="card-meter"><i id="mt-${k}"></i></div>
      <div class="card-body" id="cb-${k}">${midSkeleton(k)}</div>
    </div>`;
  $("rows").appendChild(holder.firstElementChild);
}

function leftSkeleton(k) {
  return `<span class="queued-tag" id="qt-${k}"><i class="ph-bold ph-circle-dashed" aria-hidden="true"></i>Queued</span>`;
}

function midSkeleton(k) {
  return `
    <p class="matters">${META[k].why}</p>
    <div class="sk sk-line w1"></div>
    <div class="sk sk-line w2"></div>
    <div class="sk sk-line w3"></div>
    <div class="sk sk-line w4"></div>`;
}

function markRowRunning(k) {
  const tag = $("qt-" + k);
  if (!tag) return;
  tag.className = "queued-tag running";
  tag.style.color = META[k].color;
  tag.innerHTML = `<span class="spinner" aria-hidden="true"></span>Analysing`;
}

function fillRow(c) {
  const k = c.key;
  const tone = TONE[c.status] || TONE.warn;
  const failed = c.status === "error";
  const x = c.extra || {};
  const row = $("row-" + k);
  row.style.setProperty("--verdict-color", tone.color);
  row.style.setProperty("--verdict-tint", tone.tint);
  row.style.setProperty("--verdict-border", tone.border);

  $("chr-" + k).innerHTML = failed ? `
    <span class="verdict-pill">${tone.label}</span>
    <i class="ph-bold status-icon" aria-hidden="true"></i>` : `
    <span class="verdict-pill">${tone.label}</span>
    <div class="card-score"><span class="n num" id="sc-${k}">0</span><span class="d">/100</span></div>
    <i class="ph-bold status-icon" aria-hidden="true"></i>`;

  const tagChips = [
    x.total_years ? tagChip("ph-clock", x.total_years) : "",
    x.trajectory ? tagChip(TRAJECTORY_ICON[String(x.trajectory).toLowerCase()] || "ph-trend-up", x.trajectory) : "",
  ].join("");

  const drawerId = "drawer-" + k;
  const hasBreakdown = !!(x.skills || x.present || x.missing);
  const hasDrawer = !!(c.reasoning || hasBreakdown);

  $("cb-" + k).innerHTML = `
    <p class="matters">${META[k].why}</p>
    <div class="quote-block">${failed ? esc(c.verdict) : `&ldquo;${esc(c.verdict)}&rdquo;`}</div>
    <div class="found-line"><p>${esc(c.detail)}</p>${tagChips}</div>
    ${hasDrawer ? `
      <div class="divider">
        <button class="toggle" id="tg-${k}" type="button" aria-expanded="false" aria-controls="${drawerId}">
          <span class="lbl">Show full justification</span><i class="ph-bold ph-caret-down chev" aria-hidden="true"></i>
        </button>
      </div>
      <div class="drawer" id="${drawerId}"><div><div class="drawer-inner">
        ${c.reasoning ? `<div class="why-panel"><span class="label">Why this score</span>${highlightQuotes(c.reasoning)}</div>` : ""}
        ${breakdown(k, x)}
      </div></div></div>` : ""}`;

  if (!failed) {
    const scoreEl = $("sc-" + k);
    countUp(scoreEl, c.score, 900);
    requestAnimationFrame(() => { $("mt-" + k).style.width = c.score + "%"; });
  }

  const toggle = $("tg-" + k);
  if (toggle) toggle.onclick = () => {
    const open = row.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.querySelector(".lbl").textContent = open ? "Hide full justification" : "Show full justification";
    syncExpandAllLabel();
  };

  row.querySelectorAll(".filt-btn").forEach((btn) => btn.onclick = () => {
    const want = btn.dataset.filter;
    row.querySelectorAll(".filt-btn").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    row.querySelectorAll(".chip-wrap .chip").forEach((chip) => {
      chip.classList.toggle("dimmed", want !== "all" && chip.dataset.status !== want);
    });
  });

  const icon = $("chr-" + k).querySelector(".status-icon");
  icon.className = `ph-bold ${tone.icon} status-icon ${tone.cls}`;
  requestAnimationFrame(() => icon.classList.add("show"));
}

function tagChip(icon, text) {
  return `<span class="tag-chip"><i class="ph ${icon}" aria-hidden="true"></i>${esc(text)}</span>`;
}

function breakdown(key, x) {
  if (key === "evidence" && Array.isArray(x.skills) && x.skills.length) {
    const rank = { strong: 0, moderate: 1, claimed: 2 };
    const skills = x.skills.slice().sort((a, b) => rank[a.strength] - rank[b.strength]);
    const counts = { strong: 0, moderate: 0, claimed: 0 };
    skills.forEach((s) => counts[s.strength] !== undefined && counts[s.strength]++);
    const total = skills.length;

    const bar = `
      <div class="stackbar" role="img" aria-label="${counts.strong} proven, ${counts.moderate} mentioned, ${counts.claimed} unproven">
        ${counts.strong ? `<i class="seg-strong" style="width:${(counts.strong / total) * 100}%"></i>` : ""}
        ${counts.moderate ? `<i class="seg-moderate" style="width:${(counts.moderate / total) * 100}%"></i>` : ""}
        ${counts.claimed ? `<i class="seg-claimed" style="width:${(counts.claimed / total) * 100}%"></i>` : ""}
      </div>`;

    const filters = `
      <div class="filter-row">
        <span class="flabel">Filter</span>
        ${[["all", "All", total], ["strong", "Proven", counts.strong], ["moderate", "Mentioned", counts.moderate], ["claimed", "Unproven", counts.claimed]]
          .filter((f) => f[2] > 0)
          .map(([k, l, n], i) => `<button class="filt-btn" type="button" data-filter="${k}" aria-pressed="${i === 0}">${l} ${n}</button>`).join("")}
      </div>`;

    const chips = skills.map((s) => {
      const st = STRENGTH[s.strength] || STRENGTH.moderate;
      return `<span class="chip ${st.cls}" data-status="${s.strength}" title="${esc(s.note || st.label)}">
        <i class="ph-bold ${st.icon}" aria-hidden="true"></i>${esc(s.name)}</span>`;
    }).join("");

    return `${bar}${filters}<div class="chip-wrap">${chips}</div>`;
  }

  if (key === "completeness") {
    const present = x.present || [], missing = x.missing || [];
    const list = (items, cls, label, icon) => items.length ? `
      <div class="section-list ${cls === "chip-present" ? "present" : "absent"}">
        <span class="list-label">${label}</span>
        <div class="chip-wrap">${items.map((m) => `<span class="chip ${cls}"><i class="ph-bold ${icon}" aria-hidden="true"></i>${esc(m)}</span>`).join("")}</div>
      </div>` : "";
    return `${list(present, "chip-present", "Present", "ph-check")}${list(missing, "chip-absent", "Not found in this resume", "ph-x")}`;
  }

  return "";
}

function syncExpandAllLabel() {
  const rows = ORDER.map((k) => $("row-" + k)).filter((r) => r && r.querySelector(".toggle"));
  if (!rows.length) return;
  const allOpen = rows.every((r) => r.classList.contains("open"));
  const btn = $("expandAllBtn");
  btn.innerHTML = allOpen
    ? `Collapse all <i class="ph-bold ph-caret-up" aria-hidden="true"></i>`
    : `Expand all <i class="ph-bold ph-caret-down" aria-hidden="true"></i>`;
  btn.dataset.state = allOpen ? "open" : "closed";
}

$("expandAllBtn").onclick = () => {
  const opening = $("expandAllBtn").dataset.state !== "open";
  ORDER.forEach((k) => {
    const row = $("row-" + k), toggle = $("tg-" + k);
    if (!row || !toggle) return;
    row.classList.toggle("open", opening);
    toggle.setAttribute("aria-expanded", String(opening));
    toggle.querySelector(".lbl").textContent = opening ? "Hide full justification" : "Show full justification";
  });
  syncExpandAllLabel();
};

/* ---------------- count-up ---------------- */
function countUp(el, target, duration) {
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(target * eased);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ---------------- run ---------------- */
let doneCount = 0;
const results = [];

async function analyse(file) {
  currentFile = file;
  doneCount = 0;
  results.length = 0;

  buildPipeline();
  buildRows();
  show("viewReport");
  requestAnimationFrame(syncStickyOffsets);

  $("roleChip").classList.add("loading");
  $("roleChip").innerHTML = `<span class="sk sk-pill" aria-hidden="true"></span>`;
  $("fileChip").innerHTML = `<i class="ph ph-file" aria-hidden="true"></i>${esc(file.name)}`;
  $("dialNumber").classList.add("loading");
  $("overall").textContent = "—";
  $("overallStatus").textContent = "Analysing…";
  $("dialArc").style.stroke = "var(--faint)";
  $("dialArc").style.strokeDasharray = CIRC;
  $("dialArc").style.strokeDashoffset = CIRC;
  $("actionBox").classList.remove("show");
  $("pipelineCounter").textContent = "0 of 4 complete";
  $("btnPrint").hidden = true;
  $("btnNew").hidden = false;
  $("expandAllBtn").dataset.state = "closed";

  setNodeState(ORDER[0], "running");
  addRow(ORDER[0]);
  markRowRunning(ORDER[0]);

  const body = new FormData();
  body.append("file", file);

  let res;
  try {
    res = await fetch(API + "/api/analyze", { method: "POST", body });
  } catch {
    return fail("Could not reach the server. Is <code>uvicorn app:app</code> still running?");
  }

  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done: fin } = await reader.read();
    if (fin) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = (chunk.match(/^event: (.+)$/m) || [])[1];
      const raw = (chunk.match(/^data: (.+)$/m) || [])[1];
      if (ev && raw) handle(ev, JSON.parse(raw));
    }
  }
}

function handle(ev, d) {
  if (ev === "error") return fail(esc(d.message));

  if (ev === "step") $("overallStatus").textContent = d.message.replace(/\.\.\.$/, "…");

  if (ev === "role") {
    $("roleChip").classList.remove("loading");
    $("roleChip").innerHTML = `<i class="ph ph-user" aria-hidden="true"></i>${esc(d.seniority)} ${esc(d.role)}`;
    $("overallStatus").textContent = `Scoring as ${d.seniority} ${d.role}…`;
    requestAnimationFrame(syncStickyOffsets);
  }

  if (ev === "check") {
    results.push(d);
    setNodeState(d.key, "done", d.score, d.status);
    fillRow(d);
    doneCount++;
    $("pipelineCounter").textContent = `${doneCount} of ${ORDER.length} complete`;

    const next = ORDER[ORDER.indexOf(d.key) + 1];
    if (next) {
      addRow(next);
      setNodeState(next, "running");
      markRowRunning(next);
      $("overallStatus").textContent = `Checking ${META[next].title}…`;
    }
  }

  if (ev === "done") finish(d);
}

function finish(d) {
  $("btnPrint").hidden = false;
  $("dialNumber").classList.remove("loading");
  $("roleChip").classList.remove("loading");

  const tone = TONE[d.status] || TONE.warn;
  $("dialArc").style.stroke = tone.color;
  requestAnimationFrame(() => {
    $("dialArc").style.strokeDashoffset = CIRC - (d.score / 100) * CIRC;
  });
  countUp($("overall"), d.score, 950);

  $("scoreCard").style.setProperty("--score-tint", tone.tint);
  $("verdictWord").style.setProperty("--verdict-color", tone.color);
  $("verdictWord").textContent = tone.label;

  const strong = results.filter((r) => r.status === "pass").length;
  const needs = results.filter((r) => r.status === "warn").length;
  const weak = results.filter((r) => r.status === "fail").length;
  const parts = [
    strong ? `${strong} strong` : "",
    needs ? `${needs} needs work` : "",
    weak ? `${weak} weak` : "",
  ].filter(Boolean);
  $("overallStatus").textContent = parts.length ? parts.join(" · ") : "Analysis complete";

  if (d.action) {
    $("actionText").textContent = d.action;
    $("actionBox").classList.add("show");
  }
}

function fail(html) {
  show("viewError");
  $("errText").innerHTML = html;
  $("btnNew").hidden = false;
}
