"use strict";

// =====================================================================
// Per-dataset live dashboard.
//   Landing (no ?dataset=)  -> dataset picker from index/datasets.json
//   ?dataset=<challenge>    -> that dataset only: overview chart (from the
//                              PRECOMPUTED per-run `curve` in index/<ch>.json,
//                              so we never fetch 100+ results.tsv on load),
//                              paginated run cards, capped/paginated scratchpad.
//   Per-run detail stays lazy: results.tsv + per-iter observation.json load
//   only when a run card is opened.
// =====================================================================

async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`);
  return await r.json();
}
async function fetchOptional(path) {
  try {
    const r = await fetch(path, { cache: "no-cache" });
    return r.ok ? r : null;
  } catch { return null; }
}

function parseTSV(s) {
  const lines = s.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const header = lines[0].split("\t");
  return lines.slice(1).map(line => {
    const cells = line.split("\t");
    const row = {};
    header.forEach((h, i) => row[h] = cells[i]);
    return row;
  });
}
function parseJSONL(s) {
  return s.split(/\r?\n/).filter(Boolean)
          .map(l => { try { return JSON.parse(l); } catch { return null; } })
          .filter(Boolean);
}

function fmtNum(n, digits = 3) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}
// "mean ± std" when a finite, non-zero std is present, else just the mean.
function fmtMeanStd(mean, std, digits = 3) {
  const m = fmtNum(mean, digits);
  if (std === null || std === undefined || isNaN(std) || Number(std) === 0) return m;
  return m + " ± " + fmtNum(std, digits);
}
function badge(text, cls = "") {
  const span = document.createElement("span");
  span.className = `badge ${cls}`;
  span.textContent = text;
  return span;
}
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

// ---------- colours -----------------------------------------------------

const CHART_COLOURS = [
  "#155799", "#159957", "#c9810c", "#a83232", "#6a2477", "#117a40",
  "#1f6f8b", "#b5402a", "#5c3a96", "#2e7d32", "#7b4b1e", "#3a4a8c",
];
function colourFor(i) { return CHART_COLOURS[i % CHART_COLOURS.length]; }
const _runColours = new Map();
function colourForRun(slug) { return _runColours.get(slug) || "#888"; }
function seedRunColours(runs) {
  _runColours.clear();
  runs.forEach((r, i) => _runColours.set(r.slug, colourFor(i)));
}

// ---------- routing -----------------------------------------------------

function currentDataset() {
  return new URLSearchParams(location.search).get("dataset");
}

// ---------- series labelling: solver + optimization scheme --------------
// The overview chart overlays one line per run. Label each by SOLVER and the
// OPTIMIZATION SCHEME it was produced with (agentic autoresearch vs TPE search),
// not the run-id (which is identical across a dataset and thus useless).
function schemeOf(r) {
  if (r.scheme) return r.scheme;                       // explicit index field, if present
  const s = String(r.slug || "");
  if (s.includes("claude-agentic")) return "agentic";
  if (s.includes("calibrated-tpe")) return "TPE";
  if (s.includes("-fair-")) return "fair-baseline";
  return "";                                            // early / ad-hoc runs: no clean scheme
}
function solverOf(r) {
  return r.solver || r.name || String(r.slug || "").replace(/-search-\d.*$/, "");
}
function seriesLabel(r) {
  const sc = schemeOf(r);
  const solver = solverOf(r);
  return sc ? `${solver} · ${sc}` : solver;
}

// ---------- overview chart (from precomputed curves) --------------------

let _overviewChart = null;
function renderOverviewChart(runs) {
  const container = document.getElementById("overview-charts");
  const empty = document.getElementById("overview-empty");
  if (empty) empty.remove();
  if (!container) return;
  if (_overviewChart) { _overviewChart.destroy(); _overviewChart = null; }
  container.innerHTML = "";
  if (typeof Chart === "undefined") {           // CDN still loading — retry
    container.innerHTML = `<p class="muted">loading chart…</p>`;
    setTimeout(() => renderOverviewChart(runs), 300);
    return;
  }
  const datasets = [];
  runs.forEach((r, i) => {
    const curve = (r.curve || []).filter(p => p[1] != null);
    if (curve.length === 0) return;
    datasets.push({
      label: seriesLabel(r),
      data: curve.map(p => ({ x: p[0], y: p[1] })),
      borderColor: colourForRun(r.slug), backgroundColor: colourForRun(r.slug) + "33",
      tension: 0.15, pointRadius: 1.5, borderWidth: 2, spanGaps: true,
    });
  });
  if (datasets.length === 0) {
    container.innerHTML = `<p class="muted">No iterations with headroom yet.</p>`;
    return;
  }
  const wrap = el("div", { class: "chart-wrap overview-group" });
  const canvas = document.createElement("canvas");
  wrap.appendChild(canvas);
  container.appendChild(wrap);
  _overviewChart = new Chart(canvas.getContext("2d"), {
    type: "line", data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "linear", title: { display: true, text: "iteration" },
             ticks: { stepSize: 1, precision: 0 } },
        y: { title: { display: true, text: "best headroom (running max)" },
             min: 0, max: 1 },
      },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 10 } } },
        tooltip: {
          itemSort: (a, b) => b.parsed.y - a.parsed.y,   // best (highest headroom) at top
          callbacks: {
            title: (it) => (it.length ? `iter ${it[0].parsed.x}` : ""),
            label: (it) => `${it.dataset.label}: ${it.parsed.y.toFixed(3)}`,
          },
        },
      },
    },
  });
}

// ---------- run-detail chart (unchanged) --------------------------------

function bestSoFar(rows) {
  const out = [];
  let best = -Infinity;
  for (const r of rows) {
    if (detectHallucination(r).length > 0) {
      out.push({ iter: parseInt(r.iter, 10), best: best === -Infinity ? null : best });
      continue;
    }
    const h = parseFloat(r.headroom);
    if (!isNaN(h) && h > best) best = h;
    out.push({ iter: parseInt(r.iter, 10), best: best === -Infinity ? null : best });
  }
  return out;
}

let _runChart = null;
function renderRunChart(rows) {
  if (typeof Chart === "undefined") return;
  const canvas = document.getElementById("run-chart");
  if (!canvas) return;
  if (_runChart) { _runChart.destroy(); _runChart = null; }
  const points = rows.map(r => ({
    x: parseInt(r.iter, 10), y: parseFloat(r.val_score),
    status: (r.status || "").toLowerCase(),
  })).filter(p => !isNaN(p.x) && !isNaN(p.y));
  const headroom = rows.map(r => ({
    x: parseInt(r.iter, 10), y: parseFloat(r.headroom),
  })).filter(p => !isNaN(p.x) && !isNaN(p.y));
  const best = bestSoFar(rows).map(p => ({ x: p.iter, y: p.best }));
  const colour = (s) =>
      s === "keep" ? "#159957" : s === "discard" ? "#a83232"
    : s === "crash" ? "#6a2477" : s === "timeout" ? "#c9810c" : "#888";
  _runChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets: [
      { label: "val_score", data: points.map(p => ({ x: p.x, y: p.y })),
        borderColor: "#155799", backgroundColor: "#15579933",
        pointBackgroundColor: points.map(p => colour(p.status)),
        pointBorderColor: points.map(p => colour(p.status)),
        pointRadius: 4, tension: 0.15, borderWidth: 2 },
      { label: "running-best val_score", data: best, borderColor: "#159957",
        borderDash: [6, 4], pointRadius: 0, tension: 0, borderWidth: 2 },
      { label: "headroom", data: headroom, borderColor: "#c9810c",
        backgroundColor: "#c9810c33", pointRadius: 2, tension: 0.15,
        borderWidth: 1.5, yAxisID: "yHr" },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "linear", title: { display: true, text: "iteration" },
             ticks: { stepSize: 1, precision: 0 } },
        y: { title: { display: true, text: "val_score" }, suggestedMin: 0 },
        yHr: { position: "right", title: { display: true, text: "headroom" },
               grid: { drawOnChartArea: false }, suggestedMin: 0, suggestedMax: 1 },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } },
                 tooltip: { mode: "index" } },
    },
  });
}

// ---------- pagination helper -------------------------------------------

function renderPaginated(container, items, renderFn, pageSize) {
  container.innerHTML = "";
  if (items.length === 0) {
    container.appendChild(el("p", { class: "muted" }, "Nothing to show."));
    return;
  }
  const more = el("button", { class: "load-more", type: "button" });
  let shown = 0;
  const showNext = () => {
    for (const it of items.slice(shown, shown + pageSize)) {
      container.insertBefore(renderFn(it), more);
    }
    shown = Math.min(shown + pageSize, items.length);
    if (shown >= items.length) more.remove();
    else more.textContent = `Load more (${items.length - shown} more)`;
  };
  container.appendChild(more);
  more.addEventListener("click", showNext);
  showNext();
}

// ---------- landing (dataset picker) ------------------------------------

async function loadLanding() {
  document.getElementById("dataset-view").hidden = true;
  const landing = document.getElementById("landing");
  landing.hidden = false;
  const grid = document.getElementById("dataset-grid");
  grid.innerHTML = `<p class="muted">loading datasets…</p>`;
  let data;
  try { data = await fetchJSON("runs/index/datasets.json"); }
  catch (e) { grid.innerHTML = `<p class="muted">Couldn't load index/datasets.json (${e.message}). Run scripts/rebuild_runs_index.py.</p>`; return; }
  grid.innerHTML = "";
  for (const d of data.datasets) grid.appendChild(renderDatasetCard(d));
}

function renderDatasetCard(d) {
  const card = el("a", { class: "dash-dataset-card", href: `?dataset=${d.challenge}` });
  card.appendChild(el("h3", {}, d.label || d.challenge));
  if (d.thumbnail) {
    const img = el("img", { src: d.thumbnail, loading: "lazy", decoding: "async",
                            alt: `${d.label} champion`, class: "dataset-thumb" });
    img.addEventListener("error", () => img.remove(), { once: true });
    card.appendChild(img);
  }
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "runs"), el("span", {}, String(d.n_runs))));
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "iterations"), el("span", {}, String(d.n_iterations))));
  // champion_score is the board's ranking metric per dataset basis (test_hr_mean,
  // mean over the 5 test patients, for Mayo; val headroom for breast/demo — set
  // basis-aware in build_registry); SSIM shown beside it. "—" when no run cleared
  // baseline (champion_score null).
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "champion hr"), el("span", {}, fmtNum(d.champion_score))));
  if (d.champion_ssim != null)
    card.appendChild(el("div", { class: "stat" },
      el("span", { class: "label" }, "champion SSIM"), el("span", {}, fmtNum(d.champion_ssim, 4))));
  return card;
}

// ---------- dataset view ------------------------------------------------

async function loadDataset(ch) {
  document.getElementById("landing").hidden = true;
  const view = document.getElementById("dataset-view");
  view.hidden = false;
  document.getElementById("dataset-title").textContent = ch;
  const grid = document.getElementById("runs-grid");
  grid.innerHTML = `<p class="muted">loading ${ch} runs…</p>`;
  let index;
  try { index = await fetchJSON(`runs/index/${ch}.json`); }
  catch (e) { grid.innerHTML = `<p class="muted">Couldn't load index/${ch}.json (${e.message}).</p>`; return; }
  const runs = (index.runs || []).slice()
    .sort((a, b) => (b.started || "").localeCompare(a.started || ""));
  document.getElementById("dataset-title").textContent =
    `${index.label || ch} — ${runs.length} runs`;
  seedRunColours(runs);
  renderOverviewChart(runs);
  renderPaginated(grid, runs, renderRunCard, 12);
  await loadScratch(ch);

  if (location.hash.startsWith("#run/")) {
    const slug = location.hash.slice("#run/".length);
    if (slug) showRun(slug);
  }
}

function thumb(src, alt) {
  if (!src) return null;
  const img = el("img", { src, alt, loading: "lazy", decoding: "async",
                          "data-zoomable": "1", class: "run-thumb",
                          title: "Click to enlarge" });
  img.addEventListener("error", () => img.remove(), { once: true });
  return img;
}

function shortModel(m) {
  if (!m) return "—";
  return String(m).replace(/^claude-/, "").replace(/^anthropic\//, "");
}

function renderRunCard(r) {
  const card = el("a", { class: "dash-run-card", href: `#run/${r.slug}`,
    onclick: (ev) => { ev.preventDefault(); showRun(r.slug); } });
  const runColour = colourForRun(r.slug);
  card.style.borderLeft = `4px solid ${runColour}`;
  const header = el("h3", {}, r.name || r.short_id || r.slug);
  if (r.status) header.appendChild(badge(r.status, r.status));
  card.appendChild(header);
  card.appendChild(el("div", { class: "dash-run-slug muted" }, r.slug));
  // val + test thumbnails (test only when present).
  const v = thumb(r.val_image, `${r.slug} val`);
  const t = thumb(r.test_image, `${r.slug} test`);
  if (v || t) {
    const strip = el("div", { class: "run-thumbs" });
    if (v) strip.appendChild(el("figure", { class: "run-thumb-fig" }, v,
                                el("figcaption", {}, "validation")));
    if (t) strip.appendChild(el("figure", { class: "run-thumb-fig" }, t,
                                el("figcaption", {}, "held-out test")));
    card.appendChild(strip);
  }
  const stat = (lbl, val) => card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, lbl), el("span", {}, val)));
  stat("started", (r.started || "—").slice(0, 10));
  stat("iterations", String(r.n_iterations ?? "—"));
  // Test datasets (Mayo, held-out test set): report the per-patient TEST
  // mean ± std (n=5) — NEVER validation. Datasets without a held-out test set
  // (demo_dl, breast_ct) report their single-patient val metrics as before.
  if (r.metric_basis === "test") {
    stat("test SSIM (n=5)", fmtMeanStd(r.test_ssim_mean, r.test_ssim_std, 4));
    stat("test hr (n=5)", fmtMeanStd(r.test_hr_mean, r.test_hr_std));
  } else {
    stat("best val (SSIM)", fmtNum(r.best_score, 4));
    stat("best headroom", fmtNum(r.best_headroom));
  }
  return card;
}

// ---------- per-run detail (lazy; unchanged behaviour) ------------------

async function showRun(slug) {
  const section = document.getElementById("run-detail");
  section.hidden = false;
  document.getElementById("run-detail-title").textContent = slug;
  const meta = document.getElementById("run-detail-meta");
  const iters = document.getElementById("run-detail-iters");
  const stages = document.getElementById("run-detail-stages");
  meta.textContent = "loading…"; iters.innerHTML = ""; stages.innerHTML = "";
  let manifest;
  try { manifest = await fetchJSON(`runs/${slug}/manifest.json`); }
  catch (e) { meta.textContent = `Couldn't load manifest: ${e.message}`; return; }
  renderManifest(meta, manifest);

  const finalDiv = document.getElementById("run-detail-final");
  finalDiv.hidden = true;
  const finalResp = await fetchOptional(`runs/${slug}/final.json`);
  if (finalResp) {
    const fin = await finalResp.json();
    finalDiv.hidden = false;
    finalDiv.innerHTML = `<strong>Run finalised.</strong>
      Stopped on <code>${fin.stop_reason || "?"}</code> after ${fin.n_iterations ?? "?"} iterations.
      Best iter <code>${fin.best_iter ?? "?"}</code>
      (val=${fmtNum(fin.best_val_score)}, hr=${fmtNum(fin.best_headroom)}).<br>
      <strong>Test set:</strong> score=${fmtNum(fin.final_test_score)},
      headroom=${fmtNum(fin.final_test_headroom)}.
      ${fin.notes ? `<br><em>${fin.notes}</em>` : ""}`;
  }

  const resultsResp = await fetchOptional(`runs/${slug}/results.tsv`);
  if (resultsResp) {
    const rowsAll = parseTSV(await resultsResp.text());
    renderRunChart(rowsAll);
    for (const row of rowsAll.slice().reverse()) iters.appendChild(await renderIteration(slug, row));
  } else {
    iters.appendChild(el("p", { class: "muted" }, "No results.tsv yet."));
  }

  const stagesResp = await fetchOptional(`runs/${slug}/stages.tsv`);
  if (stagesResp) {
    const rows = parseTSV(await stagesResp.text());
    if (rows.length === 0) stages.appendChild(el("p", { class: "muted" }, "No stage checks yet."));
    else for (const row of rows.reverse()) stages.appendChild(renderStageRow(row));
  } else {
    stages.appendChild(el("p", { class: "muted" }, "No stage runs yet (every 30 iterations)."));
  }
  section.scrollIntoView({ behavior: "smooth" });
}

function renderManifest(meta, m) {
  meta.innerHTML = "";
  const add = (k, v) => {
    meta.appendChild(el("span", { class: "label" }, k));
    meta.appendChild(el("span", {}, v === undefined || v === null ? "—" : String(v)));
  };
  add("challenge", m.challenge); add("started", m.started);
  add("agent", m.agent || "—"); add("model", m.model || "—");
  add("status", m.status || "running");
  if (m.notes) add("notes", m.notes);
}

async function renderIteration(slug, row) {
  const iterN = row.iter || row.iter_n || "?";
  const iterId = `iter-${String(iterN).padStart(4, "0")}`;
  const status = (row.status || (row.kept === "true" ? "keep" : row.kept === "false" ? "discard" : "")).toLowerCase();
  const statusClass = status === "keep" ? "kept" : status === "discard" ? "discard"
                    : status === "crash" ? "crash" : status === "timeout" ? "timeout" : "";
  const halluc = detectHallucination(row);
  const isHallucinated = halluc.length > 0;
  const identityTags = [];
  if (row.agent) identityTags.push(el("span", { class: "tag tag-agent", title: "agent" }, row.agent));
  if (row.model) identityTags.push(el("span", { class: "tag tag-model", title: "model" }, shortModel(row.model)));
  const scoreCell = el("span", { class: "iter-score" },
    `val=${fmtNum(row.val_score)} hr=${fmtNum(row.headroom)}`);
  if (isHallucinated) {
    scoreCell.style.textDecoration = "line-through"; scoreCell.style.opacity = "0.7";
    scoreCell.title = "Hallucinated metric — " + halluc.join("; ");
  }
  const badges = el("span", { class: "iter-badges" }, badge(status || "?", statusClass));
  if (isHallucinated) {
    const hb = badge("hallucinated", "hallucinated");
    hb.title = "Contract violation: " + halluc.join("; ");
    badges.appendChild(hb);
  }
  const summary = el("summary", {},
    el("span", { class: "iter-id" }, iterId), scoreCell,
    el("span", { class: "iter-rationale" }, row.rationale || row.description || ""),
    el("span", { class: "iter-identity" }, ...identityTags), badges,
    el("span", { class: "iter-link" }, row.commit || ""));
  const details = el("details", {
    class: "dash-iter" + (isHallucinated ? " dash-iter-hallucinated" : "") }, summary);
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loaded) return;
    details.dataset.loaded = "1";
    const body = el("div", { class: "dash-iter-body" }, el("p", { class: "muted" }, "loading…"));
    details.appendChild(body);
    let obs = null;
    try { obs = await fetchJSON(`runs/${slug}/iterations/${iterId}/observation.json`); } catch {}
    body.innerHTML = "";
    const compPath = `runs/${slug}/iterations/${iterId}/comparison.png`;
    const compDiv = el("div", { class: "compare" });
    if (status === "crash" || status === "timeout") {
      compDiv.appendChild(failedRunPlaceholder(status, `${slug} · ${iterId}`));
    } else {
      compDiv.appendChild(comparisonImg(compPath, `${slug} · ${iterId} comparison`, status, `${slug} · ${iterId}`));
    }
    body.appendChild(compDiv);
    if (obs && obs.rationale) body.appendChild(el("div", { class: "rationale-full" }, obs.rationale));
    if (obs && obs.advice_for_others) body.appendChild(el("p", { class: "scratch-advice" }, "Advice for others: " + obs.advice_for_others));
    if (obs) body.appendChild(el("pre", {}, JSON.stringify(obs, null, 2)));
  });
  return details;
}

function renderStageRow(row) {
  return el("div", { class: "dash-iter" },
    el("summary", {},
      el("span", { class: "iter-id" }, `stage @ iter ${row.iter || ""}`),
      el("span", { class: "iter-score" }, `stage_val=${fmtNum(row.stage_val_score)} hr=${fmtNum(row.stage_headroom)}`),
      el("span", { class: "iter-rationale" }, `gap iter-stage = ${fmtNum(row.gap)}`),
      badge(row.verdict || "—", row.verdict === "ok" ? "kept" : row.verdict === "overfit" ? "discard" : "")));
}

// ---------- per-dataset scratchpad --------------------------------------

async function loadScratch(ch) {
  const root = document.getElementById("scratch-list");
  const section = document.getElementById("scratchpad");
  root.innerHTML = "";
  const resp = await fetchOptional(`runs/scratch/${ch}.jsonl`);
  if (!resp) { section.hidden = true; return; }
  const entries = parseJSONL(await resp.text());
  if (entries.length === 0) { section.hidden = true; return; }
  section.hidden = false;
  renderPaginated(root, entries, renderScratchCard, 30);
}

function renderScratchCard(e) {
  const runColour = colourForRun(e.run_id);
  const status = (e.status || "").toLowerCase();
  const halluc = detectHallucination(e);
  const isHallucinated = halluc.length > 0;
  const body = el("div", { class: "scratch-body" });
  const metaText = [e.ts || "", e.run_id ? ` · ${e.run_id}` : "",
    (e.iter !== undefined) ? ` · iter ${e.iter}` : "", e.agent ? ` · ${e.agent}` : "",
    e.model ? ` · ${shortModel(e.model)}` : "", e.change_class ? ` · ${e.change_class}` : "",
    e.kept !== undefined ? ` · ${e.kept ? "keep" : "discard"}` : ""].join("");
  const metaRow = el("div", { class: "scratch-meta" }, metaText);
  if (isHallucinated) {
    const b = badge("hallucinated", "hallucinated");
    b.title = "Contract violation: " + halluc.join("; ");
    metaRow.appendChild(document.createTextNode(" ")); metaRow.appendChild(b);
  }
  body.appendChild(metaRow);
  if (e.rationale) body.appendChild(el("p", { class: "scratch-rationale" }, e.rationale));
  const scores = [];
  if (e.val_score !== undefined) scores.push(`val=${fmtNum(e.val_score)}`);
  if (e.headroom !== undefined) scores.push(`headroom=${fmtNum(e.headroom)}`);
  if (e.delta_vs_best !== undefined) scores.push(`Δ=${fmtNum(e.delta_vs_best)}`);
  if (e.params_M !== undefined) scores.push(`params=${fmtNum(e.params_M, 2)} M`);
  if (e.train_n !== undefined) scores.push(`train_n=${e.train_n}`);
  if (scores.length) body.appendChild(el("div", { class: "scratch-meta" }, scores.join(" · ")));
  if (e.advice_for_others) {
    const advice = el("p", { class: "scratch-advice" },
      el("span", { class: "scratch-advice-label" }, "Advice for others:"), " ", e.advice_for_others);
    advice.style.borderLeftColor = runColour; advice.style.color = runColour;
    body.appendChild(advice);
  }
  const thumbDiv = el("div", { class: "scratch-thumb" });
  let imgPath = e.comparison_image;
  if (!imgPath && e.run_id && (e.iter !== undefined)) {
    imgPath = `runs/${e.run_id}/iterations/iter-${String(e.iter).padStart(4, "0")}/comparison.png`;
  }
  const label = [e.run_id, e.iter !== undefined ? `iter ${e.iter}` : ""].filter(Boolean).join(" · ");
  if (status === "crash" || status === "timeout") thumbDiv.appendChild(failedRunPlaceholder(status, label));
  else if (imgPath) thumbDiv.appendChild(comparisonImg(imgPath, label, status, label));
  else thumbDiv.appendChild(failedRunPlaceholder("unknown", label));
  const card = el("div", { class: "scratch-card" + (isHallucinated ? " scratch-card-hallucinated" : "") }, body, thumbDiv);
  card.style.borderLeft = `4px solid ${isHallucinated ? "#a83232" : runColour}`;
  return card;
}

// ---------- hallucination detection (DL-Sparse-View contract) -----------

const _CONTRACT = { train_n: 400, val_n: 100 };
function detectHallucination(obj) {
  if (!obj) return [];
  // Only the dl_sparse_view challenge pins train_n=400/val_n=100. Other
  // datasets legitimately use smaller subsets, so skip the check for them.
  const slug = obj.run_id || "";
  if (slug && !slug.startsWith("dl-sparse-view")) return [];
  const reasons = [];
  const status = (obj.status || (obj.kept === "true" || obj.kept === true ? "keep" : "")).toString().toLowerCase();
  const tn = obj.train_n;
  if (tn !== undefined && tn !== null && tn !== "" && Number(tn) !== _CONTRACT.train_n)
    reasons.push(`train_n=${tn} (contract: ${_CONTRACT.train_n})`);
  const vn = obj.val_n;
  if (vn !== undefined && vn !== null && vn !== "" && Number(vn) !== _CONTRACT.val_n)
    reasons.push(`val_n=${vn} (contract: ${_CONTRACT.val_n})`);
  const rat = String(obj.rationale || "").toLowerCase();
  if ((rat.match(/robust\s+(3x3\s+)?ssim/) || rat.match(/custom\s+(training|ssim)/)) && status === "keep")
    reasons.push("custom metric/training (rationale)");
  return reasons;
}

// ---------- failed-run placeholder + lazy image -------------------------

function failedRunPlaceholder(status, label = "") {
  const s = (status || "").toLowerCase();
  let title = "run failed", icon = "✕";
  if (s === "crash") { title = "run crashed"; icon = "✕"; }
  else if (s === "timeout") { title = "timed out"; icon = "⧗"; }
  else if (s === "running") { title = "still running"; icon = "…"; }
  else if (s === "") { title = "no image"; icon = "?"; }
  return el("div", { class: `failed-run failed-${s || "unknown"}` },
    el("div", { class: "failed-run-icon" }, icon),
    el("div", { class: "failed-run-title" }, title),
    label ? el("div", { class: "failed-run-label" }, label) : null);
}

function comparisonImg(src, alt, status = "", label = "") {
  const img = el("img", { src, alt: alt || "comparison", loading: "lazy",
    decoding: "async", "data-zoomable": "1", title: "Click to enlarge" });
  img.addEventListener("error", () => img.replaceWith(failedRunPlaceholder(status || "unknown", label)), { once: true });
  return img;
}

// ---------- lightbox ----------------------------------------------------

function openLightbox(src, caption) {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  const cap = document.getElementById("lightbox-cap");
  const newtab = document.getElementById("lightbox-newtab");
  if (!box || !img) return;
  img.src = src; img.alt = caption || "comparison";
  if (cap) cap.textContent = caption || "";
  if (newtab) newtab.href = src;
  box.hidden = false; box.setAttribute("aria-hidden", "false");
  setTimeout(() => { const c = document.getElementById("lightbox-close"); if (c) c.focus(); }, 0);
}
function closeLightbox() {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  if (!box) return;
  box.hidden = true; box.setAttribute("aria-hidden", "true");
  if (img) img.src = "";
}
document.addEventListener("click", (ev) => {
  const t = ev.target;
  if (!(t instanceof Element)) return;
  const zoomable = t.closest("[data-zoomable]");
  if (zoomable && zoomable.tagName === "IMG") {
    ev.preventDefault();
    openLightbox(zoomable.getAttribute("src"), zoomable.getAttribute("alt") || "");
    return;
  }
  if (t.closest("#lightbox-close")) { closeLightbox(); return; }
  if (t.closest("#lightbox-newtab")) return;
  const box = document.getElementById("lightbox");
  if (box && !box.hidden && !t.closest(".lightbox-figure")) closeLightbox();
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { const box = document.getElementById("lightbox"); if (box && !box.hidden) closeLightbox(); }
});

// ---------- boot --------------------------------------------------------

(async function () {
  const lf = document.getElementById("last-fetched");
  if (lf) lf.textContent = new Date().toLocaleString();
  const ds = currentDataset();
  if (ds) await loadDataset(ds);
  else await loadLanding();
})();
