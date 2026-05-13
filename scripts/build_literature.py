"""Convert offline references into markdown under ``literature/``.

Produces:
  - literature/artifact_gallery.md   (+ artifact_gallery_images/)
  - literature/conrad_api_tutorials.md   (+ conrad_api_tutorials_images/)
  - literature/<paper>.md            (one per pdf in papers/)

Each output document opens with a "Source" line linking back to the canonical
URL or DOI so we don't lose attribution.
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import html2text

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
PAPERS = ROOT / "papers"

UA = {"User-Agent": "Mozilla/5.0 (Agent4CT offline literature builder)"}


# ----- HTTP / image handling ----------------------------------------------- #

def fetch(url: str, retries: int = 3) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(1 + i)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def save_image(url: str, dest_dir: Path) -> str:
    """Download an image; return the local filename only (no directory prefix)."""
    parsed = urlparse(url)
    name = Path(parsed.path).name or "img"
    # Make names safe + unique-ish.
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    dst = dest_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return name
    try:
        r = fetch(url)
        dst.write_bytes(r.content)
    except Exception as e:
        print(f"  WARN  could not download {url}: {e}")
        return ""
    return name


# ----- HTML extraction (FAU TYPO3 layout) ---------------------------------- #

NAV_IMAGE_NEEDLES = (
    "fau-banner", "rsl-banner", "lme.jpg", "email.gif", "univis2typo3",
)


def extract_content(soup: BeautifulSoup) -> BeautifulSoup | None:
    """Find the content column and drop chrome (breadcrumbs, nav, footer)."""
    # The right column is the main content area.
    right = soup.find("td", id="rechte-spalte") or soup.find("td", id_="rechte-spalte")
    if right is None:
        return None
    # Drop breadcrumbs ("pfad-info") and any nav-residue.
    for sel in ("pfad-info", "left-menu", "kontakt-box", "logo-box"):
        for n in right.find_all(id=sel):
            n.decompose()
    # Drop banner / boilerplate images.
    for img in right.find_all("img"):
        src = img.get("src", "")
        if any(n in src for n in NAV_IMAGE_NEEDLES):
            img.decompose()
    return right


def html_to_markdown(content: BeautifulSoup, page_url: str,
                     image_dir: Path, image_subpath: str) -> str:
    """Download images + convert content to markdown."""
    # Localise image srcs.
    for img in content.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        full = urljoin(page_url, src)
        local = save_image(full, image_dir)
        if local:
            img["src"] = f"{image_subpath}/{local}"
    # Strip junk attributes.
    for tag in content.find_all(True):
        for attr in list(tag.attrs):
            if attr not in ("href", "src", "alt", "title"):
                del tag.attrs[attr]
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.ignore_images = False
    h.protect_links = True
    md = h.handle(str(content))
    # Collapse > 2 blank lines.
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


# ----- Crawl drivers ------------------------------------------------------- #

def build_artifact_gallery():
    base = "https://www5.cs.fau.de/conrad/tutorials/artifact-gallery/"
    subpages = [
        ("Detector Shift", "detector-shift/index.html"),
        ("Flower Artifact", "flower-artifact/index.html"),
        ("Filter Discretization", "filter-discretization/index.html"),
        ("Limited Angle", "limited-angle/index.html"),
    ]
    image_dir = LIT / "artifact_gallery_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    out.append("# CONRAD Artifact Gallery")
    out.append("")
    out.append(f"Mirrored from <{base}index.html> for offline reference.")
    out.append("Images live under [`artifact_gallery_images/`](artifact_gallery_images).")
    out.append("")
    out.append("The gallery shows characteristic CT-reconstruction artefacts —"
               " how they look in the final image and what causes them. "
               "Useful sanity-checks when debugging a fan-beam pipeline.")
    out.append("")

    # Overview page
    print("[gallery] index")
    r = fetch(base + "index.html")
    soup = BeautifulSoup(r.text, "lxml")
    main = extract_content(soup)
    if main:
        out.append("## Overview")
        out.append("")
        out.append(f"_Source: <{base}index.html>_")
        out.append("")
        out.append(html_to_markdown(main, base, image_dir, "artifact_gallery_images"))
        out.append("")

    for title, sub in subpages:
        url = base + sub
        print(f"[gallery] {title}")
        r = fetch(url)
        soup = BeautifulSoup(r.text, "lxml")
        main = extract_content(soup)
        if main is None:
            print(f"  WARN no content extracted from {url}")
            continue
        out.append(f"## {title}")
        out.append("")
        out.append(f"_Source: <{url}>_")
        out.append("")
        out.append(html_to_markdown(main, url, image_dir, "artifact_gallery_images"))
        out.append("")

    target = LIT / "artifact_gallery.md"
    target.write_text("\n".join(out))
    print(f"wrote {target}  ({target.stat().st_size:,} bytes)")


def build_api_tutorials():
    base = "https://www5.cs.fau.de/conrad/tutorials/api-tutorials/"
    sections = [
        ("Basic Tutorials", "basic-tutorials/", [
            ("Grid Data Container", "grid-data-container/index.html"),
            ("Read Image Data", "read-image-data/index.html"),
            ("Simple MHD Reader", "simple-mhd-reader/index.html"),
            ("Point Cloud Visualization", "point-cloud-visualization/index.html"),
            ("OpenCL Introduction", "opencl-introduction/index.html"),
            ("Rotate a 2D image", "rotate-a-2d-image/index.html"),
        ]),
        ("Basic Tutorials (Videos)", "basic-tutorials-videos/", []),
        ("Reconstruction", "reconstruction/", [
            ("Iterative Reconstruction", "iterative-reconstruction/index.html"),
            ("Ordered Subsets", "ordered-subsets/index.html"),
            ("Implementation of ART", "implementation-of-art/index.html"),
            ("Minimal Scan", "minimal-scan/index.html"),
            ("Ring Correction", "ring-correction/index.html"),
            ("Ray-by-Ray Filtering", "ray-by-ray-filtering/index.html"),
            ("Scale-Space Reconstruction", "scale-space-reconstruction/index.html"),
            ("Truncation - Polynomial Extrapolation",
             "truncation-polynomial-extrapolation/index.html"),
            ("Cardiac Vasculature Reconstruction",
             "cardiac-vasculature-reconstruction/index.html"),
        ]),
        ("Advanced", "advanced/", [
            ("Spectral Absorption", "spectral-absorption/index.html"),
            ("Custom Materials", "custom-materials/index.html"),
            ("Javadoc Generation", "javadoc-generation/index.html"),
            ("Matlab Integration", "matlab-integration/index.html"),
            ("Python Integration", "python-integration/index.html"),
            ("Memory Trouble", "memory-trouble/index.html"),
            ("OpenCL Considerations", "opencl-considerations/index.html"),
            ("Statistical Shape Models (Video)",
             "statistical-shape-models-video/index.html"),
        ]),
    ]
    image_dir = LIT / "conrad_api_tutorials_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    out.append("# CONRAD API Tutorials")
    out.append("")
    out.append(f"Mirrored from <{base}index.html> for offline reference.")
    out.append("Images live under [`conrad_api_tutorials_images/`]"
               "(conrad_api_tutorials_images).")
    out.append("")
    out.append("> **Python wrapper available.** Most of the Java API "
               "documented below can be driven from Python via **pyCONRAD**: "
               "<https://git5.cs.fau.de/PyConrad/pyCONRAD>. If you only need "
               "to *call* CONRAD (geometry helpers, projectors, phantoms, "
               "file IO) from Python, start there — these tutorials are "
               "still the canonical reference for the underlying Java API "
               "but you do not need to write Java to use it.")
    out.append("")
    out.append("CONRAD is a Java open-source CT software framework from the "
               "Pattern Recognition Lab (Erlangen) / Stanford Radiology, "
               "developed by C. Schaller, A. Maier, R. Fahrig et al. "
               "The tutorials below are useful references when porting "
               "geometry / projector conventions between CONRAD and PYRO-NN.")
    out.append("")

    # Overview index
    r = fetch(base + "index.html")
    soup = BeautifulSoup(r.text, "lxml")
    main = extract_content(soup)
    if main is not None:
        out.append("## Overview")
        out.append("")
        out.append(f"_Source: <{base}index.html>_")
        out.append("")
        out.append(html_to_markdown(main, base, image_dir,
                                    "conrad_api_tutorials_images"))
        out.append("")

    for section_title, section_dir, sub in sections:
        # Section index
        section_url = base + section_dir + "index.html"
        print(f"[api] {section_title} (index)")
        try:
            r = fetch(section_url)
            soup = BeautifulSoup(r.text, "lxml")
            main = extract_content(soup)
        except Exception as e:
            print(f"  WARN could not fetch section index {section_url}: {e}")
            main = None
        out.append(f"## {section_title}")
        out.append("")
        out.append(f"_Source: <{section_url}>_")
        out.append("")
        if main is not None:
            out.append(html_to_markdown(main, section_url, image_dir,
                                        "conrad_api_tutorials_images"))
            out.append("")
        for title, page in sub:
            url = base + section_dir + page
            print(f"[api]   {title}")
            try:
                r = fetch(url)
            except Exception as e:
                print(f"    WARN {url}: {e}")
                continue
            soup = BeautifulSoup(r.text, "lxml")
            main = extract_content(soup)
            if main is None:
                continue
            out.append(f"### {section_title} → {title}")
            out.append("")
            out.append(f"_Source: <{url}>_")
            out.append("")
            out.append(html_to_markdown(main, url, image_dir,
                                        "conrad_api_tutorials_images"))
            out.append("")

    target = LIT / "conrad_api_tutorials.md"
    target.write_text("\n".join(out))
    print(f"wrote {target}  ({target.stat().st_size:,} bytes)")


# ----- PDF → markdown ------------------------------------------------------ #

PAPER_SOURCES = {
    "2604.13282_Agent4MR.pdf":
        ("Agent4MR — Agentic MR sequence development",
         "https://arxiv.org/abs/2604.13282"),
    "2201.10345_Wagner_TrainableBilateralFilter_MedPhys2022.pdf":
        ("Wagner et al. — Ultra-low Parameter Denoising: Trainable Bilateral "
         "Filter Layers in CT (Med. Phys. 2022)",
         "https://arxiv.org/abs/2201.10345"),
    "2211.01111_Wagner_DualDomainDenoising_LDCT.pdf":
        ("Wagner et al. — On the Benefit of Dual-domain Denoising in a "
         "Self-Supervised LDCT Setting (ISBI 2023)",
         "https://arxiv.org/abs/2211.01111"),
    "mccollough_2017_mayo_ldct.pdf":
        ("McCollough et al. — Low-dose CT for the detection and classification "
         "of metastatic liver lesions: Results of the 2016 Low Dose CT Grand "
         "Challenge (Med. Phys. 2017)",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC5656004/"),
    "sidky_2022_dl_sparse_view_2109.09640.pdf":
        ("Sidky & Pan — Report on the AAPM DL-Sparse-View CT Grand Challenge "
         "(Med. Phys. 2022)",
         "https://arxiv.org/abs/2109.09640"),
    "sidky_2024_dl_spectral_2212.06718.pdf":
        ("Sidky & Pan — Report on the AAPM DL-Spectral CT Grand Challenge "
         "(Med. Phys. 2024)",
         "https://arxiv.org/abs/2212.06718"),
    "abadi_2025_truect.pdf":
        ("Abadi et al. — AAPM Truth-based CT (TrueCT) Reconstruction Grand "
         "Challenge (Med. Phys. 2025)",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC11973969/"),
    "haneda_2025_ctmar.pdf":
        ("Haneda et al. — AAPM CT Metal Artifact Reduction (CT-MAR) Grand "
         "Challenge (Med. Phys. 2025)",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC12757780/"),
}


def convert_pdfs():
    import pymupdf4llm
    for filename, (title, source_url) in PAPER_SOURCES.items():
        src = PAPERS / filename
        if not src.exists():
            print(f"  SKIP missing {src}")
            continue
        target = LIT / (src.stem + ".md")
        if target.exists() and target.stat().st_mtime > src.stat().st_mtime:
            print(f"  cached {target.name}")
            continue
        print(f"[pdf] {src.name}")
        try:
            md = pymupdf4llm.to_markdown(str(src))
        except Exception as e:
            print(f"  WARN  pymupdf4llm failed on {src.name}: {e}")
            continue
        header = (
            f"# {title}\n\n"
            f"_Source: <{source_url}>_\n\n"
            f"_PDF: `papers/{filename}`_\n\n"
            "---\n\n"
        )
        target.write_text(header + md)
        print(f"  wrote {target.name}  ({target.stat().st_size:,} bytes)")


# ----- main ---------------------------------------------------------------- #

if __name__ == "__main__":
    LIT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = ["gallery", "api", "papers"]
    if "gallery" in tasks:
        build_artifact_gallery()
    if "api" in tasks:
        build_api_tutorials()
    if "papers" in tasks:
        convert_pdfs()
