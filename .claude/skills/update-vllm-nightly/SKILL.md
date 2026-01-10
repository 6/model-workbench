---
name: update-vllm-nightly
description: List available vLLM nightly builds and update config/models.yaml to pin a specific version. Use when user wants to browse nightlies, pin a specific vLLM version, or update vLLM nightly config.
user-invocable: true
---

# Update vLLM Nightly Version

This skill lists available vLLM nightly builds and helps pin a specific version in `config/models.yaml`.

## Step 1: Fetch Top 5 Docker Tags

```bash
curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=10&name=nightly-' | \
  jq -r '.results[] | select(.name | test("^nightly-[a-f0-9]{40}$")) | [(.name | ltrimstr("nightly-")), (.last_updated | split("T")[0])] | @tsv' | head -5
```
This outputs: `{commit_hash}\t{push_date}` for the 5 most recent nightly builds.

## Step 2: Fetch Commit Details via gh CLI

For each of the 5 commit hashes, fetch the commit message using `gh` (authenticated, handles pagination):

```bash
gh api repos/vllm-project/vllm/commits/{COMMIT_HASH} --jq '[.sha[0:8], (.commit.author.date | split("T")[0]), (.commit.message | split("\n")[0] | .[0:60])] | @tsv'
```

Run these 5 requests in parallel. Each returns: `{short_sha}\t{date}\t{message}`

## Step 3: Check cu130 Availability

Only check the top 5 commits (the ones most likely to be selected). Make these requests in parallel:

```
GET https://wheels.vllm.ai/{commit_hash}/cu130/
```

If returns 200, mark as "Y". If 404 or error, mark as "N".
For commits not checked, mark as "?" (can check on demand if user selects).

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
