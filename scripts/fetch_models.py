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

def normalize_to_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]

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
        if not repo_id or "REPLACE_ME" in repo_id:
            continue

        out_dir = item.get("local_dir")
        out_path = expand(out_dir) if out_dir else local_dir_for_repo(repo_id)

        hf_models.append((item, repo_id, out_path))

    if not hf_models:
        print("No HF models found in config.")
        return

    for item, repo_id, out_path in hf_models:
        out_path.mkdir(parents=True, exist_ok=True)

        files = normalize_to_list(item.get("files"))
        include = normalize_to_list(item.get("include"))
        exclude = normalize_to_list(item.get("exclude"))
        revision = item.get("revision")
        repo_type = item.get("repo_type")  # optional: "model", "dataset", "space"

        print(f"\nDownloading:")
        print(f"  repo_id:   {repo_id}")
        print(f"  local_dir: {out_path}")
        if files:
            print(f"  files:     {files}")
        if include:
            print(f"  include:   {include}")
        if exclude:
            print(f"  exclude:   {exclude}")
        if revision:
            print(f"  revision:  {revision}")
        if repo_type:
            print(f"  repo_type: {repo_type}")

        cmd = ["hf", "download", repo_id]

        # If files are specified, download only those exact files.
        # Paths can include subdirectories like "UD-Q4_K_XL/xyz.gguf".
        if files:
            cmd.extend(files)

        # Pattern-based filtering for partial repo downloads.
        if include:
            cmd.append("--include")
            cmd.extend(include)
        if exclude:
            cmd.append("--exclude")
            cmd.extend(exclude)

        if revision:
            cmd.extend(["--revision", str(revision)])
        if repo_type:
            cmd.extend(["--repo-type", str(repo_type)])

        cmd.extend(["--local-dir", str(out_path)])

        run(cmd)

    print("\nModel fetch complete.")

if __name__ == "__main__":
    main()
