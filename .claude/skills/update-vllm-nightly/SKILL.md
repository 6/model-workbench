---
name: update-vllm-nightly
description: List available vLLM nightly builds and update config/models.yaml to pin a specific version. Use when user wants to browse nightlies, pin a specific vLLM version, or update vLLM nightly config.
user-invocable: true
---

# Update vLLM Nightly Version

This skill lists available vLLM nightly builds and helps pin a specific version in `config/models.yaml`.

## Step 1: Fetch Available Nightlies

Query Docker Hub API for recent nightly tags:

```
GET https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=15&name=nightly-
```

Filter results to only include tags matching pattern `nightly-{40-character-hex}` (exclude `nightly`, `nightly-x86_64`, `nightly-aarch64`).

Extract the commit hash from each tag name (everything after `nightly-`).

## Step 2: Get Commit Details

For each commit hash, query GitHub API:

```
GET https://api.github.com/repos/vllm-project/vllm/commits/{full_commit_hash}
```

Extract:
- `commit.message` (first line only)
- `commit.author.date` (format as YYYY-MM-DD)

## Step 3: Check cu130 Availability

For each commit, check if cu130 wheels are available by fetching:

```
GET https://wheels.vllm.ai/{full_commit_hash}/cu130/
```

If returns 200, mark as available. If 404, mark as unavailable.

## Step 4: Display Results

Show a formatted table:

```
Recent vLLM Nightly Builds

#   COMMIT    DATE        CU130  MESSAGE
1   da6709c9  2025-01-10  Y      fix: memory leak in scheduler
2   72d9c316  2025-01-09  Y      feat: add flashinfer backend
3   abc12345  2025-01-08  N      refactor: cleanup unused imports
...

Current config: defaults.backends.vllm.version = v0.13.0
```

## Step 5: Interactive Selection

Use AskUserQuestion to ask the user:

**Question 1:** "Which version would you like to pin?"
- Options: Show top 4 commits as options (e.g., "1. da6709c9 - fix: memory leak")
- Include "Browse only (no update)" option

If user selects "Browse only", stop here.

**Question 2:** "Where should this version be applied?"
- "Global default" - Updates `defaults.backends.vllm`
- "Specific model" - Ask which model, then update that model's config

**Question 3:** (Only if commit has cu130 available) "Which image type?"
- "Prebuilt Docker (recommended)" - Uses `vllm/vllm-openai` images
- "cu130 (Blackwell GPUs)" - Uses wheels from `wheels.vllm.ai/{commit}/cu130`

## Step 6: Update Config

Read `config/models.yaml` and update the appropriate section.

### For Global Default (prebuilt):
```yaml
defaults:
  backends:
    vllm:
      version: nightly-{full_commit_hash}  # Update this line
      image_type: prebuilt                  # Ensure this is prebuilt
```

### For Global Default (cu130):
```yaml
defaults:
  backends:
    vllm:
      version: {full_commit_hash}           # Just the hash, no nightly- prefix
      image_type: cu130                      # Change to cu130
```

### For Model-Specific:
Add or update the model's backends section:
```yaml
models:
  - repo_id: org/model-name
    backends:
      vllm:
        version: nightly-{full_commit_hash}  # or just hash for cu130
        image_type: prebuilt                  # or cu130
```

## Step 7: Confirm

After updating, show the user what was changed:

```
Updated config/models.yaml:
  defaults.backends.vllm.version: nightly-da6709c9fe6965b7348692576ffadeee8439388e

To apply this change, rebuild images with:
  uv run python scripts/run_bench.py --model ~/models/... --rebuild
```
