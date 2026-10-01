"""Local entry point for the verbatim, numbered notebook stages.
No automatic dependency installation; no pretrained-weight or GPU work occurs until run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

def main():
    parser = argparse.ArgumentParser(description="Success-derived FSM reproduction")
    parser.add_argument("--output", type=Path, default=Path("runs/data-fsm"))
    parser.add_argument("--backend", choices=("bge", "tfidf"), default="bge")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = args.output.resolve()
    results = base / "results"
    results.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(base / "model_cache"))
    data = root / "data"
    source_manifest = json.loads((data / "source_manifest.json").read_text())
    for name, expected in source_manifest["snapshot_files"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    import pandas as pd
    from IPython.display import display, Markdown, SVG
    context = dict(__name__="__main__", BASE=base, DATA=data, RESULTS=results,
                   IN_COLAB=False, source_manifest=source_manifest, pd=pd,
                   display=display, Markdown=Markdown, SVG=SVG, os=os, sys=sys,
                   shutil=shutil, zipfile=zipfile)
    for step in sorted((root / "steps").glob("*.py")):
        if step.name.startswith("17_"):
            continue  # Widget is optional; the command line does not create a notebook UI.
        code = step.read_text()
        if step.name.startswith("07_"):
            # Only runtime configuration changes; the saved source cell is untouched.
            code = code.replace('BACKEND = "bge"', 'BACKEND = ' + repr(args.backend), 1)
        print("Running", step.name, flush=True)
        exec(compile(code, str(step), "exec"), context)

if __name__ == "__main__":
    main()
