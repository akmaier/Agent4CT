"use strict";

// Fetch JSON / text / TSV from a relative path; throw on non-2xx.
async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`);
  return await r.json();
}
async function fetchText(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`);
  return await r.text();
}
async function fetchOptional(path) {
  try {
    const r = await fetch(path, { cache: "no-cache" });
    if (!r.ok) return null;
    return r;
  } catch (e) {
    return null;
  }
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

// JSONL is one JSON object per line.
function parseJSONL(s) {
  return s.split(/\r?\n/)
          .filter(Boolean)
          .map(l => {
            try { return JSON.parse(l); }
            catch { return null; }
          })
          .filter(Boolean);
}

function fmtNum(n, digits = 3) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(digits);
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

// ---------- charts ------------------------------------------------------

let _overviewCharts = [];
let _runChart = null;

// "dl-sparse-view-iter-20260514-01" -> "dl-sparse-view-iter".
function slugPrefix(slug) {
  return slug.replace(/-\d{8}-\d{2}$/, "");
}
// First two hyphen-segments of the slug-prefix, used as the chart-group key.
// "dl-sparse-view-iter" -> "dl-sparse"; "demo-dl-sparse-view" -> "demo-dl";
// "mayo-ldct" -> "mayo-ldct".
function chartGroupKey(slug) {
  const p = slugPrefix(slug).split("-");
  return p.slice(0, 2).join("-");
}

// Distinguishable colours per run.
const CHART_COLOURS = [
  "#155799", "#159957", "#c9810c", "#a83232",
  "#6a2477", "#117a40", "#1f6f8b", "#b5402a",
  "#5c3a96", "#2e7d32",
];
function colourFor(i) { return CHART_COLOURS[i % CHART_COLOURS.length]; }

async function loadResults(slug) {
  const resp = await fetchOptional(`runs/${slug}/results.tsv`);
  if (!resp) return [];
  return parseTSV(await resp.text());
}

function bestSoFar(rows) {
  const out = [];
  let best = -Infinity;
  for (const r of rows) {
    const h = parseFloat(r.headroom);
    if (!isNaN(h) && h > best) best = h;
    out.push({ iter: parseInt(r.iter, 10), best: best === -Infinity ? null : best });
  }
  return out;
}

async function renderOverviewCharts(runs) {
  if (typeof Chart === "undefined") return; // CDN still loading
  const container = document.getElementById("overview-charts");
  if (!container) return;
  for (const c of _overviewCharts) c.destroy();
  _overviewCharts = [];
  container.innerHTML = "";

  // Group by chart-group key (first two hyphen-segments of slug-prefix).
  const groups = new Map();
  for (const r of runs) {
    const k = chartGroupKey(r.slug);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  if (groups.size === 0) {
    container.innerHTML = `<p class="muted">No runs yet.</p>`;
    return;
  }
  // Sort groups: real challenges before demos (lexicographic puts "demo-"
  // first naturally, but flip so the active work is on top).
  const groupKeys = [...groups.keys()].sort((a, b) => {
    if (a.startsWith("demo-") && !b.startsWith("demo-")) return 1;
    if (!a.startsWith("demo-") && b.startsWith("demo-")) return -1;
    return a.localeCompare(b);
  });

  for (const key of groupKeys) {
    const section = document.createElement("div");
    section.innerHTML = `
      <h3 class="overview-group-title">${key}-*</h3>
      <div class="chart-wrap overview-group"><canvas></canvas></div>
    `;
    container.appendChild(section);
    const canvas = section.querySelector("canvas");
    await renderOneOverviewChart(canvas, groups.get(key));
  }
}

async function renderOneOverviewChart(canvas, runs) {
  const datasets = [];
  for (let i = 0; i < runs.length; i++) {
    const r = runs[i];
    const rows = await loadResults(r.slug);
    if (rows.length === 0) continue;
    const series = bestSoFar(rows);
    datasets.push({
      label: r.slug,
      data: series.map(p => ({ x: p.iter, y: p.best })),
      borderColor: colourFor(i),
      backgroundColor: colourFor(i) + "33",
      tension: 0.15,
      pointRadius: 2,
      borderWidth: 2,
      spanGaps: true,
    });
  }
  if (datasets.length === 0) {
    canvas.parentElement.innerHTML =
      `<p class="muted" style="margin: 0;">No iterations yet.</p>`;
    return;
  }
  const chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "iteration" },
          ticks: { stepSize: 1, precision: 0 },
        },
        y: {
          title: { display: true, text: "best headroom (running max)" },
          min: 0, max: 1,    // fixed 0..1 so all curves use the same vertical scale
        },
      },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12 } },
        tooltip: {
          callbacks: {
            title: (items) => `iter ${items[0].parsed.x}`,
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(3)}`,
          },
        },
      },
    },
  });
  _overviewCharts.push(chart);
}

function renderRunChart(rows) {
  if (typeof Chart === "undefined") return;
  const canvas = document.getElementById("run-chart");
  if (!canvas) return;
  if (_runChart) { _runChart.destroy(); _runChart = null; }

  const points = rows.map(r => ({
    x: parseInt(r.iter, 10),
    y: parseFloat(r.val_score),
    status: (r.status || "").toLowerCase(),
  })).filter(p => !isNaN(p.x) && !isNaN(p.y));

  const headroom = rows.map(r => ({
    x: parseInt(r.iter, 10),
    y: parseFloat(r.headroom),
  })).filter(p => !isNaN(p.x) && !isNaN(p.y));

  const best = bestSoFar(rows).map(p => ({ x: p.iter, y: p.best }));

  const colour = (s) =>
      s === "keep"    ? "#159957"
    : s === "discard" ? "#a83232"
    : s === "crash"   ? "#6a2477"
    : s === "timeout" ? "#c9810c"
    : "#888";

  _runChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "val_score",
          data: points.map(p => ({ x: p.x, y: p.y })),
          borderColor: "#155799",
          backgroundColor: "#15579933",
          pointBackgroundColor: points.map(p => colour(p.status)),
          pointBorderColor: points.map(p => colour(p.status)),
          pointRadius: 4,
          tension: 0.15,
          borderWidth: 2,
        },
        {
          label: "running-best val_score",
          data: best,
          borderColor: "#159957",
          borderDash: [6, 4],
          pointRadius: 0,
          tension: 0,
          borderWidth: 2,
        },
        {
          label: "headroom",
          data: headroom,
          borderColor: "#c9810c",
          backgroundColor: "#c9810c33",
          pointRadius: 2,
          tension: 0.15,
          borderWidth: 1.5,
          yAxisID: "yHr",
          hidden: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "iteration" },
          ticks: { stepSize: 1, precision: 0 },
        },
        y: {
          title: { display: true, text: "val_score" },
          suggestedMin: 0,
        },
        yHr: {
          position: "right",
          title: { display: true, text: "headroom" },
          grid: { drawOnChartArea: false },
          suggestedMin: 0, suggestedMax: 1,
        },
      },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12 } },
        tooltip: { mode: "index" },
      },
    },
  });
}

// ---------- runs grid ---------------------------------------------------

async function loadRunsIndex() {
  const grid = document.getElementById("runs-grid");
  const empty = document.getElementById("runs-empty");
  let index;
  try {
    index = await fetchJSON("runs/runs-index.json");
  } catch (e) {
    empty.textContent = `Couldn't load runs/runs-index.json (${e.message}). Run the harness once.`;
    return;
  }
  const runs = (index.runs || []).slice().sort((a, b) =>
    (b.started || "").localeCompare(a.started || ""));
  if (runs.length === 0) {
    empty.textContent = "No runs yet. Start one with `agent4ct-record --new-run`.";
    return;
  }
  empty.remove();
  for (const r of runs) {
    grid.appendChild(renderRunCard(r));
  }
  const emptyOverview = document.getElementById("overview-empty");
  if (emptyOverview) emptyOverview.remove();
  await renderOverviewCharts(runs);
}

function renderRunCard(r) {
  const card = el("a", {
    class: "dash-run-card",
    href: `#run/${r.slug}`,
    onclick: (ev) => { ev.preventDefault(); showRun(r.slug); },
  });
  const header = el("h3", {}, r.slug);
  const statusBadge = r.status ? badge(r.status, r.status) : null;
  if (statusBadge) header.appendChild(statusBadge);
  card.appendChild(header);
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "challenge"),
    el("span", {}, r.challenge || "—")));
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "started"),
    el("span", {}, (r.started || "—").slice(0, 10))));
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "iterations"),
    el("span", {}, String(r.n_iterations ?? "—"))));
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "best val"),
    el("span", {}, fmtNum(r.best_score))));
  card.appendChild(el("div", { class: "stat" },
    el("span", { class: "label" }, "best headroom"),
    el("span", {}, fmtNum(r.best_headroom))));
  return card;
}

// ---------- per-run detail ----------------------------------------------

async function showRun(slug) {
  const section = document.getElementById("run-detail");
  section.hidden = false;
  document.getElementById("run-detail-title").textContent = slug;
  const meta = document.getElementById("run-detail-meta");
  const iters = document.getElementById("run-detail-iters");
  const stages = document.getElementById("run-detail-stages");
  meta.textContent = "loading…";
  iters.innerHTML = "";
  stages.innerHTML = "";

  let manifest;
  try {
    manifest = await fetchJSON(`runs/${slug}/manifest.json`);
  } catch (e) {
    meta.textContent = `Couldn't load manifest: ${e.message}`;
    return;
  }
  renderManifest(meta, manifest);

  // Final result banner if present.
  const finalDiv = document.getElementById("run-detail-final");
  finalDiv.hidden = true;
  const finalResp = await fetchOptional(`runs/${slug}/final.json`);
  if (finalResp) {
    const fin = await finalResp.json();
    finalDiv.hidden = false;
    finalDiv.innerHTML = `<strong>Run finalised.</strong>
      Stopped on <code>${fin.stop_reason || "?"}</code> after
      ${fin.n_iterations ?? "?"} iterations.
      Best iter <code>${fin.best_iter ?? "?"}</code>
      (val=${fmtNum(fin.best_val_score)}, hr=${fmtNum(fin.best_headroom)}).
      <br>
      <strong>Test set:</strong>
      score=${fmtNum(fin.final_test_score)},
      headroom=${fmtNum(fin.final_test_headroom)}.
      ${fin.notes ? `<br><em>${fin.notes}</em>` : ""}`;
  }

  const resultsResp = await fetchOptional(`runs/${slug}/results.tsv`);
  let rowsAll = [];
  if (resultsResp) {
    rowsAll = parseTSV(await resultsResp.text());
    renderRunChart(rowsAll);
    for (const row of rowsAll.slice().reverse()) {
      iters.appendChild(await renderIteration(slug, row));
    }
  } else {
    iters.appendChild(el("p", { class: "muted" }, "No results.tsv yet."));
  }

  const stagesResp = await fetchOptional(`runs/${slug}/stages.tsv`);
  if (stagesResp) {
    const rows = parseTSV(await stagesResp.text());
    if (rows.length === 0) {
      stages.appendChild(el("p", { class: "muted" }, "No stage checks yet."));
    } else {
      for (const row of rows.reverse()) {
        stages.appendChild(renderStageRow(row));
      }
    }
  } else {
    stages.appendChild(el("p", { class: "muted" },
      "No stage runs yet (every 30 iterations)."));
  }

  section.scrollIntoView({ behavior: "smooth" });
}

function renderManifest(meta, m) {
  meta.innerHTML = "";
  const add = (k, v) => {
    meta.appendChild(el("span", { class: "label" }, k));
    meta.appendChild(el("span", {}, v === undefined || v === null ? "—" : String(v)));
  };
  add("challenge", m.challenge);
  add("started", m.started);
  add("agent", m.agent || "—");
  add("model", m.model || "—");
  add("status", m.status || "running");
  if (m.notes) add("notes", m.notes);
}

async function renderIteration(slug, row) {
  const iterN = row.iter || row.iter_n || "?";
  const iterId = `iter-${String(iterN).padStart(4, "0")}`;
  const status = (row.status || (row.kept === "true" ? "keep" : row.kept === "false" ? "discard" : "")).toLowerCase();
  const statusClass = status === "keep" ? "kept"
                    : status === "discard" ? "discard"
                    : status === "crash" ? "crash"
                    : status === "timeout" ? "timeout"
                    : "";

  const summary = el("summary", {},
    el("span", { class: "iter-id" }, iterId),
    el("span", { class: "iter-score" },
      `val=${fmtNum(row.val_score)} hr=${fmtNum(row.headroom)}`),
    el("span", { class: "iter-rationale" }, row.rationale || row.description || ""),
    badge(status || "?", statusClass),
    el("span", { class: "iter-link" }, row.commit || ""));

  const details = el("details", { class: "dash-iter" }, summary);
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loaded) return;
    details.dataset.loaded = "1";
    const body = el("div", { class: "dash-iter-body" }, el("p", { class: "muted" }, "loading…"));
    details.appendChild(body);
    let obs = null;
    try {
      obs = await fetchJSON(`runs/${slug}/iterations/${iterId}/observation.json`);
    } catch (_) { /* fine, no observation */ }
    body.innerHTML = "";
    const compPath = (obs && obs.comparison_image)
      ? `runs/${slug}/iterations/${iterId}/${(obs.comparison_image.split("/").pop())}`
      : `runs/${slug}/iterations/${iterId}/comparison.png`;
    const compResp = await fetchOptional(compPath);
    const compDiv = el("div", { class: "compare" });
    if (compResp) {
      compDiv.appendChild(el("img", { src: compPath, alt: `${iterId} comparison` }));
    } else {
      compDiv.appendChild(el("p", { class: "nocompare" }, "(no comparison image)"));
    }
    body.appendChild(compDiv);
    if (obs && obs.rationale) {
      body.appendChild(el("div", { class: "rationale-full" }, obs.rationale));
    }
    if (obs && obs.advice_for_others) {
      body.appendChild(el("p", { class: "scratch-advice" },
        "Advice for others: " + obs.advice_for_others));
    }
    if (obs) {
      body.appendChild(el("pre", {}, JSON.stringify(obs, null, 2)));
    }
  });
  return details;
}

function renderStageRow(row) {
  return el("div", { class: "dash-iter" },
    el("summary", {},
      el("span", { class: "iter-id" }, `stage @ iter ${row.iter || ""}`),
      el("span", { class: "iter-score" },
        `stage_val=${fmtNum(row.stage_val_score)} hr=${fmtNum(row.stage_headroom)}`),
      el("span", { class: "iter-rationale" },
        `gap iter-stage = ${fmtNum(row.gap)}`),
      badge(row.verdict || "—",
            row.verdict === "ok" ? "kept" :
            row.verdict === "overfit" ? "discard" : "")));
}

// ---------- shared scratchpad -------------------------------------------

async function loadScratch() {
  const root = document.getElementById("scratch-list");
  const resp = await fetchOptional("runs/observations.jsonl");
  if (!resp) {
    root.innerHTML = `<p class="muted">No scratchpad yet — runs/observations.jsonl missing.</p>`;
    return;
  }
  const text = await resp.text();
  const entries = parseJSONL(text);
  if (entries.length === 0) {
    root.innerHTML = `<p class="muted">Scratchpad is empty.</p>`;
    return;
  }
  // Newest last in file; reverse to show newest first.
  for (const e of entries.reverse()) {
    root.appendChild(renderScratchCard(e));
  }
}

function renderScratchCard(e) {
  const body = el("div", { class: "scratch-body" });
  body.appendChild(el("div", { class: "scratch-meta" },
    [
      e.ts || "",
      e.run_id ? ` · ${e.run_id}` : "",
      (e.iter !== undefined) ? ` · iter ${e.iter}` : "",
      e.change_class ? ` · ${e.change_class}` : "",
      e.kept !== undefined ? ` · ${e.kept ? "keep" : "discard"}` : "",
    ].join("")));
  if (e.rationale) {
    body.appendChild(el("p", { class: "scratch-rationale" }, e.rationale));
  }
  const scores = [];
  if (e.val_score !== undefined) scores.push(`val=${fmtNum(e.val_score)}`);
  if (e.headroom !== undefined) scores.push(`headroom=${fmtNum(e.headroom)}`);
  if (e.delta_vs_best !== undefined) scores.push(`Δ=${fmtNum(e.delta_vs_best)}`);
  if (e.params_M !== undefined) scores.push(`params=${fmtNum(e.params_M, 2)} M`);
  if (e.train_n !== undefined) scores.push(`train_n=${e.train_n}`);
  if (scores.length > 0) {
    body.appendChild(el("div", { class: "scratch-meta" }, scores.join(" · ")));
  }
  if (e.advice_for_others) {
    body.appendChild(el("p", { class: "scratch-advice" },
      "Advice for others: " + e.advice_for_others));
  }

  const thumb = el("div", { class: "scratch-thumb" });
  if (e.comparison_image) {
    // Path is relative to docs/ (e.g. "runs/<slug>/iterations/iter-NNNN/comparison.png")
    thumb.appendChild(el("img", { src: e.comparison_image, alt: "comparison" }));
  } else {
    thumb.appendChild(el("div", { class: "nocompare" }, "no image"));
  }
  return el("div", { class: "scratch-card" }, body, thumb);
}

// ---------- boot --------------------------------------------------------

(async function () {
  document.getElementById("last-fetched").textContent =
    new Date().toLocaleString();
  await loadRunsIndex();
  await loadScratch();

  // If the URL has a #run/<slug> anchor, open it.
  if (location.hash.startsWith("#run/")) {
    const slug = location.hash.slice("#run/".length);
    if (slug) showRun(slug);
  }
})();
