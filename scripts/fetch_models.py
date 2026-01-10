#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml
from common import CONFIG_PATH, GGUF_MODELS_ROOT, MODELS_ROOT


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def is_gguf_model(item: dict) -> bool:
    """Detect if model config indicates GGUF format.

    A model is GGUF if:
    1. Repo ID contains "GGUF" (case-insensitive) - e.g., unsloth/Model-GGUF
    2. Include patterns contain .gguf files
    3. Explicit format: gguf in config
    """
    # Check explicit format
    if item.get("format", "").lower() == "gguf":
        return True

    # Check repo name
    repo_id = item.get("repo_id", "").lower()
    if "gguf" in repo_id:
        return True

    # Check include patterns for .gguf files
    include = item.get("include", [])
    if isinstance(include, str):
        include = [include]
    if any(".gguf" in p.lower() for p in include):
        return True

    return False


def local_dir_for_repo(repo_id: str, is_gguf: bool = False) -> Path:
    """Get storage path, routing GGUF to secondary storage."""
    root = GGUF_MODELS_ROOT if is_gguf else MODELS_ROOT
    return root / repo_id


def normalize_to_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def get_expected_paths(items: list) -> set[Path]:
    """Get all model paths that should exist based on config."""
    paths = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id")
        if not repo_id or "REPLACE_ME" in repo_id:
            continue

        # Determine storage location (same logic as main download)
        out_dir = item.get("local_dir")
        if out_dir:
            out_path = expand(out_dir)
        else:
            gguf = is_gguf_model(item)
            out_path = local_dir_for_repo(repo_id, is_gguf=gguf)

        # Handle revisions - each creates a subfolder
        revisions = item.get("revisions") or []
        if not revisions:
            single_rev = item.get("revision")
            if single_rev:
                revisions = [single_rev]

        if revisions:
            # Add revision subfolders
            for revision in revisions:
                paths.add(out_path / revision)
        # Always add the base model path
        paths.add(out_path)

    return paths


def find_existing_models() -> set[Path]:
    """Find all model folders in storage directories (2 levels deep: org/model)."""
    existing = set()
    for root in {MODELS_ROOT, GGUF_MODELS_ROOT}:
        if not root.exists():
            continue
        for org_dir in root.iterdir():
            if not org_dir.is_dir():
                continue
            for model_dir in org_dir.iterdir():
                if model_dir.is_dir():
                    existing.add(model_dir)
    return existing


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    else:
        return f"{size_bytes / 1e3:.1f} KB"


def prune_models(items: list):
    """Find and interactively delete orphaned model folders."""
    expected = get_expected_paths(items)
    existing = find_existing_models()

    # Orphaned = exists but not expected (and not parent of expected)
    orphaned = []
    for path in existing:
        if path in expected:
            continue
        # Check if this path is a parent of any expected path
        if any(exp.is_relative_to(path) for exp in expected):
            continue
        orphaned.append(path)

    if not orphaned:
        print("No orphaned model folders found.")
        return

    print(f"Found {len(orphaned)} orphaned model folder(s):\n")

    for path in sorted(orphaned):
        size = get_dir_size(path)
        size_str = format_size(size)

        try:
            response = input(f"Delete {path} ({size_str})? [y/N]: ").lower().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

        if response == "y":
            try:
                shutil.rmtree(path)
                print(f"  Deleted: {path}")
                # Clean up empty parent (org) directory
                if path.parent not in {MODELS_ROOT, GGUF_MODELS_ROOT}:
                    if path.parent.exists() and not any(path.parent.iterdir()):
                        path.parent.rmdir()
                        print(f"  Removed empty: {path.parent}")
            except (OSError, PermissionError) as e:
                print(f"  Error deleting {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch models from HuggingFace Hub")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Interactively remove model folders not referenced in config",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config: {CONFIG_PATH}")

    # Ensure storage directories exist
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    GGUF_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    items = data.get("models") or []
    if not isinstance(items, list):
        raise SystemExit("config/models.yaml: 'models' must be a list")

    # Handle --prune flag
    if args.prune:
        prune_models(items)
        return

    hf_models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("repo_id")
        if not repo_id or "REPLACE_ME" in repo_id:
            continue

        # Determine storage location based on format
        out_dir = item.get("local_dir")
        if out_dir:
            out_path = expand(out_dir)
        else:
            gguf = is_gguf_model(item)
            out_path = local_dir_for_repo(repo_id, is_gguf=gguf)

        hf_models.append((item, repo_id, out_path))

    if not hf_models:
        print("No HF models found in config.")
        return

    for item, repo_id, out_path in hf_models:
        files = normalize_to_list(item.get("files"))
        include = normalize_to_list(item.get("include"))
        exclude = normalize_to_list(item.get("exclude"))
        repo_type = item.get("repo_type")  # optional: "model", "dataset", "space"

        # Support both 'revision' (string) and 'revisions' (array)
        # Each revision is downloaded to its own subfolder: {out_path}/{revision}/
        revisions = item.get("revisions") or []
        if not revisions:
            # Backward compat: single 'revision' string
            single_rev = item.get("revision")
            if single_rev:
                revisions = [single_rev]
            else:
                revisions = [None]  # No revision specified, download default branch

        for revision in revisions:
            # Determine target directory - use subfolder if revision specified
            if revision:
                target_dir = out_path / revision
            else:
                target_dir = out_path
            target_dir.mkdir(parents=True, exist_ok=True)

            print("\nDownloading:")
            print(f"  repo_id:   {repo_id}")
            print(f"  local_dir: {target_dir}")
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
            # IMPORTANT: Use single flag with multiple patterns, NOT a for-loop with
            # separate flags per pattern. The hf CLI overwrites (not appends) when
            # the same flag is repeated, so `--include a --include b` only downloads b.
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

            cmd.extend(["--local-dir", str(target_dir)])

            run(cmd)

    print("\nModel fetch complete.")


if __name__ == "__main__":
    main()
