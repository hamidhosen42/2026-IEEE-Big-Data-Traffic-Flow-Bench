#!/bin/sh
# Build the award report: figures from repository numbers, then the PDF (pdflatex, two passes).
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
"$ROOT/.venv/bin/python" make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error report.tex > build.log
grep -E "Output written|Warning" build.log || true
