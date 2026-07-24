"""make_submission.py — build the two submission packages from paper/tex/.

  paper/build/medphys_anon/   double-anonymized package for Medical Physics
                              (no names/affiliations/acknowledgments anywhere in
                              the manuscript or supplement, no repository URLs)
                              + title_page.pdf carrying the identifying material,
                              per the journal's de-identifying checklist.
  paper/build/arxiv/          full, non-anonymized package for arXiv, with the
                              repository links intact and a pre-built .bbl.

The sources in paper/tex/ are NEVER modified: every transform is applied to a
copy inside the build directory. Run:  python3 scripts/make_submission.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "paper" / "tex"
FIGS = REPO / "paper" / "figures"
BUILD = REPO / "paper" / "build"

SUPPORT = ["refs.bib", "USG.cls", "WileyNJD-AMA.bst",
           "NJDapacite.sty", "NJDnatbib.sty", "mla.sty"]
# The Wiley class \RequirePackage{lettersp}; the file ships as LETTERSP.STY.
# macOS is case-insensitive so it resolves locally, but arXiv builds on Linux,
# where it MUST be lowercase. Stage it under the name LaTeX actually asks for.
CASED = {"LETTERSP.STY": "lettersp.sty"}
# USG.cls pulls the Wiley logos from images/ and its main font from Fonts/Stix.
SUPPORT_DIRS = ["images", "Fonts"]

# The sentence the author asked for wherever code links used to be.
RELEASE = ("will be published as open source upon acceptance of the manuscript")


# ---------------------------------------------------------------- anonymizing
def strip_front_matter(s: str) -> str:
    """Replace the author/affiliation/corresponding block with anonymous stand-ins."""
    s = re.sub(r"^\\author\[[^\]]*\]\{[^}]*\}\n", "", s, flags=re.M)
    s = re.sub(r"^\\address\[[^\]]*\]\{.*?\}\}\n\n?", "", s, flags=re.M | re.S)
    s = re.sub(r"^\\corres\{.*?\}\n", "", s, flags=re.M)
    s = re.sub(r"^\\authormark\{.*?\}\n", "\\\\authormark{Anonymous \\\\textsc{et al.}}\n",
               s, flags=re.M)
    # one anonymous author + affiliation so the class still typesets a header
    s = s.replace("\\authormark{Anonymous",
                  "\\author[1]{Anonymous Author}\n\n"
                  "\\address[1]{\\orgname{Institution withheld for review}}\n\n"
                  "\\authormark{Anonymous", 1)
    return s


def strip_backmatter(s: str) -> str:
    """Remove Acknowledgments and Conflicts of Interest (they move to the title page)."""
    s = re.sub(r"\\bmsubsection\*\{Acknowledgments\}.*?(?=\\bmsubsection\*|\Z)",
               "", s, flags=re.S)
    s = re.sub(r"\\bmsubsection\*\{Conflicts of Interest\}.*?(?=\\bmsubsection\*|\Z)",
               "", s, flags=re.S)
    return s


def delink(s: str) -> str:
    """Remove every repository/dashboard URL and promise release on acceptance."""
    # main text, contribution (v)
    s = re.sub(
        r"\(v\) We release all 26 methods, our compact solver, and the agentic loop as open source \(\\url\{[^}]*\}\)\.",
        f"(v) All 26 methods, our compact solver, and the agentic loop {RELEASE}.", s)
    # main text, Data and Code Availability
    s = re.sub(
        r"is open source at \\url\{[^}]*\}, with the leaderboards and per-iteration provenance on the live dashboard at \\url\{[^}]*\}\.",
        f"together with the leaderboards and per-iteration provenance, {RELEASE}.", s)
    s = s.replace(
        "and the evaluation scripts --- is together with",
        "and the evaluation scripts --- together with")
    # supplement S8
    s = re.sub(
        r"\\textbf\{Data and code\.\} The code, the 26 solvers and the compact recombination, and the per-iteration records are in the public repository, with the leaderboards on the project page:\s*\\begin\{center\}.*?\\end\{center\}\s*",
        "\\\\textbf{Data and code.} The code, the 26 solvers and the compact "
        "recombination, the per-iteration records, and the leaderboards "
        f"{RELEASE}. ", s, flags=re.S)
    s = s.replace("The dashboard lists the boards", "The released dashboard lists the boards")
    s = s.replace("in the public repository,", "in the released repository,")
    s = s.replace("are in the repository.", "will be in the released repository.")
    s = s.replace("in the repository.", "in the released repository.")
    s = s.replace("The repository also contains", "The released repository also contains")
    s = s.replace("correspond to the repository at commit", "correspond to the code at commit")
    # belt and braces: no bare URL/name may survive
    s = re.sub(r"\\url\{https?://[^}]*(github|akmaier)[^}]*\}", "", s)
    return s


TITLE_PAGE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\begin{document}
\pagestyle{empty}

\begin{center}
{\Large\bfseries Agentic Autoresearch for CT Reconstruction}\\[1.2em]
{\itshape Title page (identifying information, withheld from the review copy)}
\end{center}

\vspace{1em}
\noindent\textbf{Authors and affiliations}\\[0.4em]
Andreas Maier\textsuperscript{1,5},
Lucas Kachelrie\ss\textsuperscript{1,2},
Siming Bayer\textsuperscript{1},
Yixing Huang\textsuperscript{3},
Yan Xia\textsuperscript{2},
Amber Simpson\textsuperscript{4},
Moritz Zaiss\textsuperscript{2,5}

\vspace{0.8em}
\noindent
\textsuperscript{1}Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg (FAU), Erlangen, Germany\\
\textsuperscript{2}Universit\"atsklinikum Erlangen (UKER), Erlangen, Germany\\
\textsuperscript{3}Peking University, Beijing, China\\
\textsuperscript{4}University of Alberta, Edmonton, Canada\\
\textsuperscript{5}Department Artificial Intelligence in Biomedical Engineering (AIBE), Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg (FAU), Erlangen, Germany

\vspace{1.2em}
\noindent\textbf{Corresponding author}\\[0.4em]
Andreas Maier, andreas.maier@fau.de

\vspace{1.2em}
\noindent\textbf{Acknowledgments}\\[0.4em]
We thank the organizers of the AAPM Low-Dose CT Grand Challenge (Mayo Clinic),
Emil Sidky, and the organizers of the Breast CT Challenge for making their data
available.

\vspace{1.2em}
\noindent\textbf{Conflicts of Interest}\\[0.4em]
The authors declare no relevant conflicts of interest.

\end{document}
"""


ANON_README = """Medical Physics - double-anonymized submission package
==========================================================

  main.pdf / main.tex           manuscript, de-identified
  supplement.pdf / .tex         supplementary material, de-identified
  title_page.pdf / .tex         the identifying material, kept separate:
                                authors, affiliations, corresponding author,
                                Acknowledgments, Conflicts of Interest
  figures/ images/ Fonts/       assets (Wiley logos + Stix font come from the class)
  USG.cls *.sty *.bst refs.bib  class and bibliography support

De-identification applied (per the journal's de-identifying checklist):
  - author block, affiliations and corresponding author replaced by
    "Anonymous Author / Institution withheld for review"; running head is
    "Anonymous et al."
  - Acknowledgments and Conflicts of Interest removed from the manuscript and
    moved to title_page
  - every repository / dashboard URL removed. Code availability now reads
    "will be published as open source upon acceptance of the manuscript"
  - PDFs carry no Author/Title/Subject/Keywords metadata and no XMP packet

Retained deliberately: reference-list citations to the authors' own prior work,
written in the third person. That is standard for double-anonymized review; the
checklist bars names in the manuscript, not citations to published literature.

Regenerate with:  python3 scripts/make_submission.py
"""

ARXIV_README = """arXiv submission package (full, non-anonymized)
================================================

  main.pdf / main.tex         manuscript, with authors and repository links
  supplement.pdf / .tex       supplementary material
  main.bbl / supplement.bbl   pre-built bibliographies, so arXiv need not run
                              BibTeX
  figures/ images/ Fonts/     assets
  USG.cls *.sty *.bst         Wiley class and support

arXiv is a non-commercial preprint server, which Medical Physics explicitly
permits. Two obligations follow: declare the preprint at submission, and after
acceptance update this posting with a link to the published article.

Build note: this compiles with XeLaTeX and loads a local font via fontspec
(Fonts/Stix). If arXiv's TeX Live struggles with the Wiley class or that font,
submit main.pdf directly - arXiv accepts PDF-only submissions.
"""


# ------------------------------------------------------------------- building
def stage(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in SUPPORT:
        src = TEX / f
        if src.exists():
            shutil.copy2(src, dst / f)
    for src_name, dst_name in CASED.items():
        src = TEX / src_name
        if src.exists():
            shutil.copy2(src, dst / dst_name)
    for d in SUPPORT_DIRS:
        src = TEX / d
        if src.is_dir():
            shutil.copytree(src, dst / d, dirs_exist_ok=True)
    figs = dst / "figures"
    figs.mkdir(exist_ok=True)
    for f in FIGS.glob("*.pdf"):
        shutil.copy2(f, figs / f.name)


def fix_graphicspath(s: str) -> str:
    # keep images/ on the path: the class pulls its logos from there
    return s.replace(r"\graphicspath{{../figures/}{./images/}}",
                     r"\graphicspath{{figures/}{images/}}")


def compile_tex(d: Path, stem: str) -> int:
    for _ in range(2):
        subprocess.run(["latexmk", "-xelatex", "-bibtex", "-interaction=nonstopmode",
                        f"{stem}.tex"], cwd=d, capture_output=True)
    pdf = d / f"{stem}.pdf"
    if not pdf.exists():
        return 0
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


IDENTIFYING = ("maier", "kachelrie", "bayer", "yixing", "xia", "simpson", "zaiss",
               "erlangen", "peking", "alberta", "fau", "aibe", "akmaier")


def check_pdf_metadata(pdf: Path) -> str:
    """Verify the PDF carries no identifying document metadata.

    The XeLaTeX build writes only Creator/Producer (toolchain strings), never an
    Author/Title field, so there is normally nothing to strip. We therefore
    VERIFY rather than rewrite: re-encoding through ghostscript to 'scrub' fields
    that are already absent would only risk degrading the figures.
    """
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    bad = [ln for ln in out.splitlines()
           if any(t in ln.lower() for t in IDENTIFYING)]
    raw = pdf.read_bytes()
    if b"<x:xmpmeta" in raw:
        bad.append("XMP packet present")
    return "clean" if not bad else "IDENTIFYING METADATA: " + "; ".join(bad)


def finish(d: Path, readme: str) -> None:
    """Drop LaTeX intermediates and document the package."""
    for pat in ("*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.out", "*.blg",
                "*.xdv", "*.pag", "*.toc", "*.synctex.gz"):
        for f in d.glob(pat):
            f.unlink()
    (d / "README.txt").write_text(readme)


# --------------------------------------------------------------- arXiv upload
# Exactly the files pdflatex actually reads (taken from main.fls / supplement.fls),
# nothing more. Fonts/ is NOT needed: USG.cls has its fontspec/setmainfont lines
# commented out, so the document builds under pdflatex and Stix comes from TeX
# Live. NJDapacite.sty, mla.sty, refs.bib and the .bst are never read either -
# the prebuilt .bbl covers the bibliography.
ARXIV_NEED = ["USG.cls", "lettersp.sty", "NJDnatbib.sty"]
ARXIV_FIGS = ["fig1_agentic_loop.pdf", "fig2_params_vs_hr.pdf", "fig3_reversal.pdf",
              "fig4_recon_panels.pdf", "fig_data_examples.pdf"]
ARXIV_IMGS = ["Wiley_logo.eps", "Wiley_logo-eps-converted-to.pdf",
              "allergy.eps", "allergy-eps-converted-to.pdf",
              "openaccess.eps", "openaccess-eps-converted-to.pdf"]

ARXIV_ZIP_README = """arXiv upload - minimal source package
=====================================

Engine: pdflatex. USG.cls has its fontspec/setmainfont lines commented out, so
nothing here needs XeLaTeX; main.tex carries \\pdfoutput=1 so arXiv selects
pdflatex and PDF output explicitly. Verified: pdflatex gives the same 10-page
result as our XeLaTeX build, with no errors and no undefined references.

Contents - only the files the build actually reads:
  main.tex, main.bbl        the manuscript (bbl prebuilt, so no BibTeX run)
  USG.cls, lettersp.sty,    Wiley class + the two support packages it loads.
  NJDnatbib.sty             lettersp.sty is lowercase on purpose: the Wiley
                            distribution ships LETTERSP.STY, which resolves on
                            case-insensitive macOS but NOT on arXiv's Linux.
                            That is the "lettersp.sty not found" error.
  figures/                  the five figures main.tex includes
  images/                   the three logos USG.cls includes
  anc/supplement.pdf        supplementary material as an ancillary file

Deliberately excluded: Fonts/ (unused), NJDapacite.sty, mla.sty, refs.bib,
WileyNJD-AMA.bst (never read), and supplement.tex. Only ONE .tex file with a
\\documentclass is shipped - two would leave arXiv guessing which is the main
document, which is the likely cause of the error repeating three times.

If you would rather have the supplement typeset into the posting than attached
as an ancillary file, say so and it can be appended to main.tex instead.
"""


def build_arxiv_zip() -> Path:
    """Zip exactly the necessary files, flat at the archive root (no wrapper dir)."""
    import zipfile
    src = BUILD / "arxiv"
    out = BUILD / "arxiv_upload.zip"
    # \pdfoutput=1 tells arXiv: use pdflatex and produce PDF.
    tex = (src / "main.tex").read_text()
    if "\\pdfoutput" not in tex:
        tex = "\\pdfoutput=1\n" + tex
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        z.write(src / "main.bbl", "main.bbl")
        for f in ARXIV_NEED:
            z.write(src / f, f)
        for f in ARXIV_FIGS:
            z.write(src / "figures" / f, f"figures/{f}")
        for f in ARXIV_IMGS:
            p = src / "images" / f
            if p.exists():
                z.write(p, f"images/{f}")
        z.write(src / "supplement.pdf", "anc/supplement.pdf")
        z.writestr("README.txt", ARXIV_ZIP_README)
    return out


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)

    # ---- 1. anonymized Medical Physics package ----
    anon = BUILD / "medphys_anon"
    stage(anon)
    for stem in ("main", "supplement"):
        s = (TEX / f"{stem}.tex").read_text()
        s = fix_graphicspath(strip_front_matter(s))
        s = delink(strip_backmatter(s) if stem == "main" else delink(s))
        (anon / f"{stem}.tex").write_text(s)
    (anon / "title_page.tex").write_text(TITLE_PAGE)
    pages = {stem: compile_tex(anon, stem) for stem in ("main", "supplement", "title_page")}
    meta = {s: check_pdf_metadata(anon / f"{s}.pdf")
            for s in ("main", "supplement", "title_page") if (anon / f"{s}.pdf").exists()}
    finish(anon, ANON_README)

    # ---- 2. arXiv package (full, non-anonymized) ----
    arx = BUILD / "arxiv"
    stage(arx)
    for stem in ("main", "supplement"):
        s = fix_graphicspath((TEX / f"{stem}.tex").read_text())
        (arx / f"{stem}.tex").write_text(s)
    arx_pages = {stem: compile_tex(arx, stem) for stem in ("main", "supplement")}
    finish(arx, ARXIV_README)

    print("=== medphys_anon (double-anonymized) ===")
    for k, v in pages.items():
        print(f"  {k}.pdf: {v} pages   metadata: {meta.get(k, 'n/a')}")
    print("=== arxiv (full) ===")
    for k, v in arx_pages.items():
        print(f"  {k}.pdf: {v} pages")
    z = build_arxiv_zip()
    print(f"=== arXiv upload zip ===\n  {z.relative_to(REPO)}  "
          f"({z.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
