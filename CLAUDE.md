# Required first step in every turn

**Before any tool call**, verify you have read **[`README.md`](README.md)**
in your CURRENT context. After context compaction your file-level memory
is gone — the resume summary does NOT include canonical recipes. If you
cannot quote from `README.md`, Read it NOW. README.md points onward to
`solver_plan.md` and `docs/findings.md`; follow that chain.

## Hard rules

1. **CT-image vision is unreliable.** The built-in vision module is not
   trained on CT / sinogram / medical imagery. Multiple wrong "visual"
   conclusions have already been drawn from CT slices in user-guided
   debug sessions.
   - **Autonomous agentic autoresearch:** vision is fine as a coarse
     sanity check (a number is still the source of truth).
   - **User-guided sessions:** do NOT draw conclusions from CT slices.
     Report quantitative stats (means, stds, RMSE, SSIM, profiles)
     and let the user inspect the image.

2. **Every reported result needs an image.** No claim of "solver X
   achieved hr=Y on dataset Z" without a supporting figure path.

3. **Do not write artefacts to `/tmp`.** The user cannot browse `/tmp`
   from Finder. Save into the repo (e.g. `results/`, `docs/runs/`,
   `docs/_debug/`) or under `~/Documents/...` so the path is reachable
   from the user's file browser.
