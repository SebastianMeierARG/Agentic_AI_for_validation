#!/usr/bin/env python
"""
run_all_projects.py
-------------------
Runs src/run_audit.py (--no-validation) for each subfolder found in documents/.

Per-project outputs and FAISS client index are stored inside the project folder:
  documents/<PROJECT>/faiss_index_client/    <- vector store (built once, cached)
  documents/<PROJECT>/outputs/<timestamp>/   <- audit results

Loose files at documents/ root are skipped (assumed to be a separate dataset
already processed).

Usage:
  python run_all_projects.py
"""

import os
import sys
import copy
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_ROOT = PROJECT_ROOT / "documents"
GLOBAL_OUTPUTS = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ── PER-PROJECT OVERRIDES ─────────────────────────────────────────────────────
# Add any project-specific settings here. Only listed keys are overridden;
# everything else inherits from config.yaml.
# Supported override keys (nested as dot-separated): rag_settings.document_language
PROJECT_OVERRIDES = {
    "BPN":         {"rag_settings": {"document_language": "Spanish"}},
    "UCI":         {"rag_settings": {"document_language": "Spanish"}},
    "CapitalFlow": {"rag_settings": {"document_language": "English"}},
}
# ─────────────────────────────────────────────────────────────────────────────


def load_base_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_dirs():
    """Return subdirectories of documents/ sorted by name."""
    return sorted(
        [d for d in DOCUMENTS_ROOT.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )


def run_project(project_dir: Path, base_config: dict):
    project_name = project_dir.name
    print(f"\n{'='*60}")
    print(f"  PROJECT: {project_name}")
    print(f"{'='*60}")

    faiss_dir = project_dir / "faiss_index_client"
    outputs_dir = project_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build per-project config — override only the paths that differ per project
    cfg = copy.deepcopy(base_config)
    cfg.setdefault('paths', {})
    cfg['paths']['documents_folder'] = str(project_dir)
    cfg['paths']['faiss_index_client'] = str(faiss_dir)
    cfg.setdefault('audit_trail', {})['run_validation'] = False

    # Apply any project-specific overrides (e.g. document_language)
    for section, keys in PROJECT_OVERRIDES.get(project_name, {}).items():
        cfg.setdefault(section, {}).update(keys)
        for k, v in keys.items():
            print(f"  Override: {section}.{k} = {v}")

    # Write temporary config file next to the real one
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml',
        dir=PROJECT_ROOT,
        prefix=f'_tmp_{project_name}_',
        delete=False,
        encoding='utf-8',
    )
    try:
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        tmp_config_path = tmp.name

        # Snapshot global outputs/ before the run so we can detect the new folder
        GLOBAL_OUTPUTS.mkdir(exist_ok=True)
        before = set(GLOBAL_OUTPUTS.iterdir())

        env = {**os.environ, 'AUDIT_CONFIG_PATH': tmp_config_path}
        result = subprocess.run(
            [sys.executable, "src/run_audit.py", "--no-validation"],
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            print(f"  WARNING: run_audit.py exited with code {result.returncode} for {project_name}.")

        # Detect new output folder(s) and move them into the project folder
        after = set(GLOBAL_OUTPUTS.iterdir())
        new_folders = sorted(after - before, key=lambda p: p.name)

        if new_folders:
            for folder in new_folders:
                dest = outputs_dir / folder.name
                shutil.move(str(folder), str(dest))
                print(f"  Results moved to: {dest}")
        else:
            print(f"  WARNING: No new output folder detected for {project_name}.")

    finally:
        if os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)


def main():
    projects = get_project_dirs()
    if not projects:
        print("No project subfolders found in documents/. Exiting.")
        return

    print(f"Projects found: {[p.name for p in projects]}")
    base_config = load_base_config()

    for project_dir in projects:
        run_project(project_dir, base_config)

    print("\nAll projects completed.")
    print("Results are in documents/<PROJECT>/outputs/")


if __name__ == "__main__":
    main()
