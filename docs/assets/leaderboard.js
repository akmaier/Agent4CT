"use strict";
// =====================================================================
// leaderboard.js — fills every <div data-leaderboard="<challenge>"> mount on a
// page from the registry's leaderboard.json (built by scripts/build_registry.py).
//
// Markdown boards carry PROSE ONLY now; the numbers live in exactly one place
// (the registry) and are rendered here at view time, so nothing a human types
// can go stale. Renders ALL solvers via table.js (ranked first, excluded dimmed
// below) — there is no slice / top-N anywhere.
//
// Path resolution: a mount may set data-json to an explicit URL; otherwise we
// try a small set of relative candidates so the same script works from
// docs/leaderboards/*.html (../runs/...) and docs/*.html (runs/...).
// =====================================================================

(function () {
  var CANDIDATES = [
    "../runs/index/leaderboard.json",   // docs/leaderboards/*.html
    "runs/index/leaderboard.json",      // docs/*.html (index, dashboard)
    "/Agent4CT/docs/runs/index/leaderboard.json",
  ];

  function fetchFirst(urls) {
    var i = 0;
    function tryNext() {
      if (i >= urls.length) return Promise.reject(new Error("leaderboard.json not found"));
      var url = urls[i++];
      return fetch(url, { cache: "no-cache" }).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json().then(function (j) {
          return { data: j, url: url };
        });
      }).catch(tryNext);
    }
    return tryNext();
  }

  function imgBaseFor(jsonUrl) {
    // image paths in the registry are "runs/<slug>/...". The board page resolves
    // them relative to the leaderboard.json location: strip "runs/index/leaderboard.json".
    return jsonUrl.replace(/runs\/index\/leaderboard\.json$/, "");
  }

  function render(mounts, payload) {
    var board = payload.data.datasets || {};
    var base = imgBaseFor(payload.url);
    mounts.forEach(function (mount) {
      var ch = mount.getAttribute("data-leaderboard");
      var lb = board[ch];
      mount.innerHTML = "";
      if (!lb || !lb.rows || !lb.rows.length) {
        mount.appendChild(document.createTextNode(
          "No leaderboard data for " + ch + " yet."));
        return;
      }
      var ranked = lb.rows.filter(function (r) { return !r.excluded_reason; }).length;
      // pending-test (no final.json yet) is distinct from below-baseline — don't
      // lump the two into one "below" count on the test board.
      var pending = lb.rows.filter(function (r) {
        return r.excluded_reason === "pending-test"; }).length;
      var below = lb.rows.length - ranked - pending;
      var tail = "";
      if (pending) tail += ", " + pending + " pending test-scoring";
      if (below) tail += ", " + below + " below baseline";
      var caption = document.createElement("p");
      caption.className = "lb-caption muted";
      caption.innerHTML =
        "<strong>" + lb.rows.length + " solvers</strong> — ranked by <strong>" +
        (lb.ranking_metric || "headroom") + "</strong> (" + (lb.tiebreak || "val_ssim") +
        " tiebreak). " + ranked + " above baseline" + tail +
        " (dimmed rows shown for completeness). Built from the registry — no hand-typed numbers.";
      mount.appendChild(caption);
      mount.appendChild(
        window.A4CTable.renderLeaderboardTable(lb.rows,
          { imgBase: base, metricBasis: lb.metric_basis || "val",
            testN: lb.test_n || null }));
    });
  }

  // Compact champions summary (for the front pages): one row per dataset =
  // that dataset's leaderboard rank-1. Driven by the SAME registry data, so it
  // can't disagree with the boards. Renders into <div data-champions>.
  function renderChampions(mount, payload) {
    var board = payload.data.datasets || {};
    var base = imgBaseFor(payload.url);
    var order = ["breast_ct", "breast_ct_noise", "demo_dl", "mayo_ldct"];
    var labels = {
      breast_ct: "Breast-CT", breast_ct_noise: "BreastCT-Noise",
      demo_dl: "Demo-DL", mayo_ldct: "Mayo-LDCT",
    };
    var linkBase = mount.getAttribute("data-link-base") || "leaderboards/";
    mount.innerHTML = "";
    var table = document.createElement("table");
    table.className = "lb-table";
    table.innerHTML =
      "<thead><tr><th>Dataset</th><th>Champion</th><th class='lb-num'>SSIM</th>" +
      "<th class='lb-num'>hr</th><th>Board</th></tr></thead>";
    var tb = document.createElement("tbody");
    order.forEach(function (ch) {
      var lb = board[ch];
      if (!lb || !lb.rows) return;
      var r1 = lb.rows.filter(function (r) { return r.rank === 1; })[0];
      // Headline SSIM/hr follow the board's basis: test mean (n=5) for Mayo, val
      // otherwise — so the champions row agrees with the board it links to.
      var isTest = (lb.metric_basis === "test");
      var ssim = r1 ? (isTest ? r1.test_ssim_mean : r1.val_ssim) : null;
      var hr = r1 ? (isTest ? r1.test_hr_mean : r1.headroom) : null;
      var tr = document.createElement("tr");
      tr.className = "lb-row";
      var href = linkBase + ch + ".html";
      tr.innerHTML =
        "<td class='lb-solver'>" + (labels[ch] || ch) + "</td>" +
        "<td>" + (r1 ? r1.solver_name : "—") + "</td>" +
        "<td class='lb-num'>" + window.A4CTable.fmtSSIM(ssim) + "</td>" +
        "<td class='lb-num lb-hr'>" + window.A4CTable.fmtHr(hr) + "</td>" +
        "<td><a href='" + href + "'>" + ch + "</a></td>";
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    mount.appendChild(table);
  }

  function boot() {
    var champ = document.querySelector("[data-champions]");
    var mounts = Array.prototype.slice.call(
      document.querySelectorAll("[data-leaderboard]"));
    if (!mounts.length && !champ) return;
    if (champ && !champ.innerHTML.trim())
      champ.innerHTML = "<p class=\"muted\">loading champions…</p>";
    mounts.forEach(function (m) {
      if (!m.innerHTML.trim()) m.innerHTML = "<p class=\"muted\">loading leaderboard…</p>";
    });
    // allow a per-mount explicit data-json override on the first mount
    var explicit = (mounts[0] || champ).getAttribute("data-json");
    var urls = explicit ? [explicit].concat(CANDIDATES) : CANDIDATES;
    fetchFirst(urls).then(function (payload) {
      if (mounts.length) render(mounts, payload);
      if (champ) renderChampions(champ, payload);
    }).catch(function (e) {
      var msg = "<p class=\"muted\">Couldn't load the leaderboard (" +
        e.message + "). Run scripts/build_registry.py.</p>";
      mounts.forEach(function (m) { m.innerHTML = msg; });
      if (champ) champ.innerHTML = msg;
    });
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
