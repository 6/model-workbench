Nightly Environment
===================

This directory contains a separate Python environment with bleeding-edge
transformers/tokenizers from git master. Required for models like GLM-4.6V
that need unreleased fixes.

Setup
-----
  cd nightly && uv sync

Or run ./scripts/bootstrap.sh which sets up both environments.

Usage
-----
Models with `nightly: true` in config/models.yaml automatically use this
environment when running benchmarks. No manual switching needed.

Manual override flags:
  --force-nightly    Use nightly env regardless of model config
  --force-stable     Use stable env regardless of model config

Structure
---------
  .venv/             Python environment (gitignored)
  pyproject.toml     Dependencies with git-based overrides
