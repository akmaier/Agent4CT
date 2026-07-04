"use strict";
// =====================================================================
// table.js — a pure leaderboard table renderer over registry views.
//
// Builds an HTML <table> from an array of row objects + a column spec. There is
// NO .slice() and NO row cap anywhere in this file: every row passed in is
// rendered. Excluded (below-baseline / discarded / non-finite) rows are rendered
// DIMMED and BELOW the ranked rows — never dropped — so every solver always
// shows and "top-N" is structurally unexpressible.
//
// Shared by leaderboard.js (the 3 dataset boards) and any other registry table.
// Plain global functions (no module system) so it works as a <script> include.
// =====================================================================

(function (global) {
  function _fmt(x, digits) {
    if (x === null || x === undefined || (typeof x === "number" && !isFinite(x)))
      return "—";
    return Number(x).toFixed(digits);
  }
  function fmtSSIM(x) { return _fmt(x, 4); }
  function fmtHr(x) { return _fmt(x, 4); }
  function fmtPSNR(x) { return _fmt(x, 2); }
  function fmtRMSE(x) {
    if (x === null || x === undefined || (typeof x === "number" && !isFinite(x)))
      return "—";
    return Number(x).toExponential(2);
  }
  function fmtTime(x) {
    if (x === null || x === undefined || (typeof x === "number" && !isFinite(x)))
      return "—";
    return Math.round(Number(x)).toString();
  }
  function fmtParams(pm) {
    // params_M is in MILLIONS. Show 3 dp for >=0.001 M, else the raw integer
    // param count (a handful of params reads better than "0.000").
    if (pm === null || pm === undefined || (typeof pm === "number" && !isFinite(pm)))
      return "—";
    if (pm >= 0.001) return Number(pm).toFixed(3);
    return String(Math.round(pm * 1e6));
  }
  function fmtMeanStd(mean, std, f) {
    // "mean ± std" when a finite, non-zero std is present, else just the mean.
    const m = f(mean);
    if (std === null || std === undefined ||
        (typeof std === "number" && !isFinite(std)) || std === 0) return m;
    return m + " ± " + f(std);
  }

  function _el(tag, attrs, children) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") e.className = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) {
      if (c == null) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return e;
  }

  // The metric columns depend on the board's basis (set by build_registry as
  // lb.metric_basis, passed via opts.metricBasis):
  //   "test" (Mayo): hr · SSIM · PSNR · RMSE as mean ± std over the 5 held-out
  //                  test patients (n=5), and the board is ranked by test hr.
  //   "val"  (others): the single-patient val (L277) SSIM · hr · PSNR · RMSE,
  //                  SSIM/PSNR/RMSE carrying the per-slice std.
  // params(M) · <metrics> · time(s) · best iter · comparison on both.
  function _metricCols(basis, testN) {
    if (basis === "test") {
      // Held-out test-set size varies per dataset (Mayo n=5 patients, breast
      // n=200 cases). Fall back to "n=5" only if the board omits test_n (legacy).
      var nsuf = " (n=" + (testN || 5) + ")";
      return [
        { label: "hr" + nsuf, cls: "lb-num lb-hr",
          val: function (r) { return fmtMeanStd(r.test_hr_mean, r.test_hr_std, fmtHr); } },
        { label: "SSIM" + nsuf, cls: "lb-num",
          val: function (r) { return fmtMeanStd(r.test_ssim_mean, r.test_ssim_std, fmtSSIM); } },
        { label: "PSNR dB" + nsuf, cls: "lb-num",
          val: function (r) { return fmtMeanStd(r.test_psnr_mean, r.test_psnr_std, fmtPSNR); } },
        { label: "RMSE" + nsuf, cls: "lb-num",
          val: function (r) { return fmtMeanStd(r.test_rmse_mean, r.test_rmse_std, fmtRMSE); } },
      ];
    }
    return [
      { label: "SSIM", cls: "lb-num",
        val: function (r) { return fmtMeanStd(r.val_ssim, r.val_ssim_std, fmtSSIM); } },
      { label: "hr", cls: "lb-num lb-hr",
        val: function (r) { return fmtMeanStd(r.headroom, r.headroom_std, fmtHr); } },
      { label: "PSNR (dB)", cls: "lb-num",
        val: function (r) { return fmtMeanStd(r.val_psnr, r.val_psnr_std, fmtPSNR); } },
      { label: "RMSE", cls: "lb-num",
        val: function (r) { return fmtMeanStd(r.val_rmse, r.val_rmse_std, fmtRMSE); } },
    ];
  }

  function renderLeaderboardTable(rows, opts) {
    opts = opts || {};
    const imgBase = opts.imgBase || "";
    const cols = _metricCols(opts.metricBasis || "val", opts.testN);
    const table = _el("table", { class: "lb-table" }, []);
    const headRow = [
      _el("th", { class: "lb-rank" }, ["#"]),
      _el("th", {}, ["Solver"]),
      _el("th", { class: "lb-num" }, ["params (M)"]),
    ];
    cols.forEach(function (c) { headRow.push(_el("th", { class: "lb-num" }, [c.label])); });
    headRow.push(_el("th", { class: "lb-num" }, ["time (s)"]));
    headRow.push(_el("th", {}, ["best iter"]));
    headRow.push(_el("th", {}, ["comparison"]));
    table.appendChild(_el("thead", {}, [_el("tr", {}, headRow)]));
    const tbody = _el("tbody", {}, []);

    // EVERY row, in the order given (ranked first, excluded dimmed below). No
    // slicing — the loop walks the full array.
    rows.forEach(function (r) {
      const excluded = !!r.excluded_reason;
      const tr = _el("tr", { class: excluded ? "lb-row lb-excluded" : "lb-row" }, []);
      // rank cell: number for ranked, the exclusion reason for excluded
      const rankCell = excluded
        ? _el("td", { class: "lb-rank lb-rank-excl", title: r.excluded_reason }, ["—"])
        : _el("td", { class: "lb-rank" }, [String(r.rank)]);
      tr.appendChild(rankCell);

      const nameCell = _el("td", { class: "lb-solver" }, [r.solver_name || r.solver_key || "?"]);
      if (excluded) {
        nameCell.appendChild(document.createTextNode(" "));
        nameCell.appendChild(_el("span", { class: "lb-tag-excl", title:
          "Below baseline / discarded — shown for completeness, not ranked" },
          [r.excluded_reason]));
      }
      tr.appendChild(nameCell);

      tr.appendChild(_el("td", { class: "lb-num" }, [fmtParams(r.params_M)]));
      cols.forEach(function (c) {
        tr.appendChild(_el("td", { class: c.cls }, [c.val(r)]));
      });
      tr.appendChild(_el("td", { class: "lb-num" }, [fmtTime(r.elapsed_s)]));

      const iterCell = _el("td", {}, []);
      iterCell.appendChild(document.createTextNode(
        r.best_iter != null ? ("iter-" + r.best_iter) : "—"));
      tr.appendChild(iterCell);

      const imgCell = _el("td", { class: "lb-img" }, []);
      if (r.image) {
        const href = imgBase + r.image;
        const a = _el("a", { href: href, target: "_blank", rel: "noopener" }, []);
        const img = _el("img", { src: href, loading: "lazy", decoding: "async",
          alt: (r.solver_name || "") + " comparison", class: "lb-thumb" }, []);
        img.addEventListener("error", function () { a.textContent = "(image)"; }, { once: true });
        a.appendChild(img);
        imgCell.appendChild(a);
      } else {
        imgCell.appendChild(document.createTextNode("—"));
      }
      tr.appendChild(imgCell);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  global.A4CTable = {
    renderLeaderboardTable: renderLeaderboardTable,
    fmtParams: fmtParams, fmtSSIM: fmtSSIM, fmtHr: fmtHr,
    fmtPSNR: fmtPSNR, fmtRMSE: fmtRMSE, fmtTime: fmtTime,
  };
})(window);
