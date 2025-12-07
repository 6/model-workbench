#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "models.yaml"
MODELS_ROOT = Path.home() / "models"

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()

def local_dir_for_repo(repo_id: str) -> Path:
    # Mirror HF structure under ~/models/<org>/<repo>
    return MODELS_ROOT / repo_id

def main():
    if not CONFIG.exists():
        raise SystemExit(f"Missing config: {CONFIG}")

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(CONFIG.read_text()) or {}
    items = data.get("models") or []
    if not isinstance(items, list):
        raise SystemExit("config/models.yaml: 'models' must be a list")

    hf_models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("source") != "hf":
            continue
        repo_id = item.get("repo_id")
        if not repo_id:
            continue

        # allow override, but default to mirroring repo_id
        out_dir = item.get("local_dir")
        if out_dir:
            out_path = expand(out_dir)
        else:
            out_path = local_dir_for_repo(repo_id)

        hf_models.append((repo_id, out_path))

    if not hf_models:
        print("No HF models found in config.")
        return

    for repo_id, out_path in hf_models:
        if "REPLACE_ME" in repo_id:
            print(f"Skipping placeholder repo_id: {repo_id}")
            continue

        out_path.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading:")
        print(f"  repo_id:   {repo_id}")
        print(f"  local_dir: {out_path}")

        run([
            "hf", "download", repo_id,
            "--local-dir", str(out_path),
        ])

    print("\nModel fetch complete.")

if __name__ == "__main__":
    main()
