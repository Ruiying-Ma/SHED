#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from git import Repo
from sphinx.application import Sphinx
from docutils import nodes


# ===============================
# Config
# ===============================

OUTPUT_DIR = Path("toc_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "toc-crawler/1.0"}
TIMEOUT = 15


# ===============================
# Web utilities
# ===============================

def fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def is_sphinx_site(html):
    soup = BeautifulSoup(html, "html.parser")
    for meta in soup.find_all("meta", attrs={"name": "generator"}):
        if meta.get("content", "").lower().startswith("sphinx"):
            return True
    return False


def find_github_repo(html):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "github.com" in href and href.count("/") >= 4:
            return href.split("#")[0]
    return None


# ===============================
# Repo utilities
# ===============================

def clone_repo(repo_url, workdir):
    repo_path = workdir / "repo"
    Repo.clone_from(repo_url, repo_path, depth=1)
    return repo_path


def find_docs_root(repo_path):
    for root, dirs, files in os.walk(repo_path):
        if "conf.py" in files:
            return Path(root)
    return None


# ===============================
# Sphinx ToC extraction
# ===============================

def build_sphinx_and_extract_toc(srcdir):
    builddir = srcdir / "_build"
    doctreedir = srcdir / "_doctree"

    app = Sphinx(
        srcdir=srcdir,
        confdir=srcdir,
        outdir=builddir,
        doctreedir=doctreedir,
        buildername="html",
        freshenv=True,
        warningiserror=False,
    )
    app.build()

    def walk(node):
        items = []
        for child in node.children:
            if isinstance(child, nodes.list_item):
                ref = child.next_node(nodes.reference)
                title = ref.astext() if ref else None
                items.append({
                    "title": title,
                    "children": walk(child),
                })
        return items

    toc_by_doc = {}
    for docname, tocnode in app.env.tocs.items():
        toc_by_doc[docname] = walk(tocnode)

    return toc_by_doc


# ===============================
# Main pipeline
# ===============================

def process_site(url):
    print(f"[+] Processing {url}")
    html = fetch_html(url)

    if not is_sphinx_site(html):
        print("  [-] Not a Sphinx site")
        return

    repo_url = find_github_repo(html)
    if not repo_url:
        print("  [-] No GitHub repo found")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        try:
            repo_path = clone_repo(repo_url, tmp)
        except Exception as e:
            print(f"  [-] Clone failed: {e}")
            return

        docs_root = find_docs_root(repo_path)
        if not docs_root:
            print("  [-] No Sphinx docs root found")
            return

        try:
            toc = build_sphinx_and_extract_toc(docs_root)
        except Exception as e:
            print(f"  [-] Sphinx build failed: {e}")
            return

        doc_id = urlparse(url).netloc.replace(".", "_")
        out = {
            "doc_id": doc_id,
            "source": {
                "type": "sphinx",
                "site": url,
                "repo": repo_url,
                "docs_root": str(docs_root.relative_to(repo_path)),
            },
            "toc": toc,
        }

        with open(OUTPUT_DIR / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"  [+] Saved ToC → {doc_id}.json")


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    SEED_URLS = [
        "https://pytorch.org/docs/stable/",
        "https://docs.python.org/3/",
        "https://huggingface.co/docs",
    ]

    for url in SEED_URLS:
        try:
            process_site(url)
        except Exception as e:
            print(f"[!] Fatal error on {url}: {e}")
