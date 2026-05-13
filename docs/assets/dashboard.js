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

// Strip the leading "claude-" or trailing "-sonnet-4.5" noise to make
// model strings fit a card. e.g. "claude-sonnet-4.5" -> "sonnet-4.5".
function shortModel(m) {
  if (!m) return "—";
  return String(m).replace(/^claude-/, "").replace(/^anthropic\//, "");
}

// Build a comparison <img> that:
//   * loads lazily so 30 thumbnails don't all hit the wire on first paint;
//   * falls back to a placeholder if the file 404s instead of leaving a
//     broken-image icon (some recordings have a comparison_image field
//     pointing at a file that hasn't been pushed yet, or there's no PNG
//     for crash/timeout iters);
//   * gets data-zoomable so the lightbox click handler picks it up.
function comparisonImg(src, alt, placeholderText = "(no comparison image)") {
  const img = el("img", {
    src,
    alt: alt || "comparison",
    loading: "lazy",
    decoding: "async",
    "data-zoomable": "1",
    title: "Click to enlarge",
  });
  img.addEventListener("error", () => {
    const ph = el("p", { class: "nocompare" }, placeholderText);
    img.replaceWith(ph);
  }, { once: true });
  return img;
}

// ---------- lightbox ----------------------------------------------------
//
// Each comparison PNG rendered into the dashboard (iter-detail body or
// scratchpad thumbnail) gets data-zoomable="1" so the lightbox click
// handler picks it up. We use event delegation on document so the
// handler covers images that are appended later.
function openLightbox(src, caption) {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  const cap = document.getElementById("lightbox-cap");
  const newtab = document.getElementById("lightbox-newtab");
  if (!box || !img) return;
  img.src = src;
  img.alt = caption || "comparison";
  if (cap) cap.textContent = caption || "";
  if (newtab) newtab.href = src;
  box.hidden = false;
  box.setAttribute("aria-hidden", "false");
  // Defer focus so the browser doesn't scroll the page first.
  setTimeout(() => {
    const close = document.getElementById("lightbox-close");
    if (close) close.focus();
  }, 0);
}
function closeLightbox() {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  if (!box) return;
  box.hidden = true;
  box.setAttribute("aria-hidden", "true");
  if (img) img.src = "";
}

document.addEventListener("click", (ev) => {
  const t = ev.target;
  if (!(t instanceof Element)) return;
  // Open: any image flagged zoomable.
  const zoomable = t.closest("[data-zoomable]");
  if (zoomable && zoomable.tagName === "IMG") {
    ev.preventDefault();
    openLightbox(zoomable.getAttribute("src"),
                 zoomable.getAttribute("alt") || "");
    return;
  }
  // Close: any click on the backdrop, the close button, or outside the
  // figure inside the open lightbox.
  if (t.closest("#lightbox-close")) {
    closeLightbox();
    return;
  }
  // Don't treat the new-tab anchor click as close.
  if (t.closest("#lightbox-newtab")) return;
  const box = document.getElementById("lightbox");
  if (box && !box.hidden && !t.closest(".lightbox-figure")) {
    closeLightbox();
  }
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    const box = document.getElementById("lightbox");
    if (box && !box.hidden) closeLightbox();
  }
});

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
  // Identity strip: agent · model · short-id (the date-ordinal slug tail).
  // Multi-agent runs surface their *current* recorder here so it's obvious
  // which subagent's iteration last touched the run.
  const identity = el("div", { class: "dash-run-identity" },
    el("span", { class: "tag tag-agent", title: "agent" }, r.agent || "—"),
    el("span", { class: "tag tag-model", title: "model" }, shortModel(r.model)),
    el("span", { class: "tag tag-id",    title: "short-id" }, r.short_id || "—"));
  card.appendChild(identity);
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

  // Identity tags (only render when the iteration actually has them — old
  // pre-2026-05-14 rows that pre-date the agent/model columns just omit them).
  const identityTags = [];
  if (row.agent) {
    identityTags.push(el("span", { class: "tag tag-agent", title: "agent" }, row.agent));
  }
  if (row.model) {
    identityTags.push(el("span", { class: "tag tag-model", title: "model" }, shortModel(row.model)));
  }
  const identityCell = identityTags.length
    ? el("span", { class: "iter-identity" }, ...identityTags)
    : el("span", { class: "iter-identity" });

  const summary = el("summary", {},
    el("span", { class: "iter-id" }, iterId),
    el("span", { class: "iter-score" },
      `val=${fmtNum(row.val_score)} hr=${fmtNum(row.headroom)}`),
    el("span", { class: "iter-rationale" }, row.rationale || row.description || ""),
    identityCell,
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
    // Build the comparison-image path from the iteration directory itself
    // — not from the cross-run observation's comparison_image field, which
    // is fragile (concurrent writers, schema drift). For crash/timeout
    // iters the file simply 404s and `comparisonImg` swaps in a "no
    // image" placeholder.
    const compPath = `runs/${slug}/iterations/${iterId}/comparison.png`;
    const compDiv = el("div", { class: "compare" },
      comparisonImg(compPath, `${slug} · ${iterId} comparison`));
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
      e.agent ? ` · ${e.agent}` : "",
      e.model ? ` · ${shortModel(e.model)}` : "",
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
  // Reconstruct the path from run_id + iter rather than trusting
  // observation.comparison_image (older entries can have a stale path,
  // empty string, or just be missing for crash/timeout iters).
  let imgPath = e.comparison_image;
  if (!imgPath && e.run_id && (e.iter !== undefined)) {
    const iterId = `iter-${String(e.iter).padStart(4, "0")}`;
    imgPath = `runs/${e.run_id}/iterations/${iterId}/comparison.png`;
  }
  if (imgPath) {
    const altParts = [];
    if (e.run_id) altParts.push(e.run_id);
    if (e.iter !== undefined) altParts.push(`iter ${e.iter}`);
    thumb.appendChild(comparisonImg(imgPath,
                                    altParts.join(" · "), "no image"));
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
