---
name: update-sglang-nightly
description: List available SGLang nightly builds and update config/models.yaml to pin a specific version. Use when user wants to browse nightlies, pin a specific SGLang version, or update SGLang nightly config.
user-invocable: true
---

# Update SGLang Nightly Version

This skill lists available SGLang CUDA 13 nightly builds and helps pin a specific version in `config/models.yaml`.

## Step 1: Fetch Recent CUDA 13 Nightly Tags

```bash
curl -s 'https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags?page_size=20&name=nightly-dev-cu13' | \
  jq -r '.results[] | select(.name | startswith("nightly-dev-cu13-")) | [.name, (.last_updated | split("T")[0])] | @tsv' | head -10
```

This outputs: `{tag}\t{push_date}` for the 10 most recent CUDA 13 nightly builds.

## Step 2: Display Results

Show a formatted table:

```
Recent SGLang CUDA 13 Nightly Builds

#   TAG                                      DATE
1   nightly-dev-cu13-20260110-8a1be0dc       2026-01-10
2   nightly-dev-cu13-20260109-abc12345       2026-01-09
3   nightly-dev-cu13-20260108-def67890       2026-01-08
...

Current config: defaults.backends.sglang.backend_version = nightly-dev-cu13-20260110-8a1be0dc
```

## Step 3: Interactive Selection

Use AskUserQuestion to ask the user:

**Question 1:** "Which SGLang version would you like to pin?"
- Options: Show top 4 tags as options (e.g., "1. nightly-dev-cu13-20260110-8a1be0dc")
- Include "Browse only (no update)" option

If user selects "Browse only", stop here.

**Question 2:** "Where should this version be applied?"
- "Global default" - Updates `defaults.backends.sglang`
- "Specific model" - Ask which model, then update that model's config

## Step 4: Update Config

Read `config/models.yaml` and update the appropriate section.

### For Global Default:
```yaml
defaults:
  backends:
    sglang:
      backend_version: {selected_tag}  # Update this line
```

### For Model-Specific:
Add or update the model's profiles section:
```yaml
models:
  - repo_id: org/model-name
    profiles:
      default:
        backend: sglang
        backend_version: {selected_tag}
```

## Step 5: Confirm

After updating, show the user what was changed:

```
Updated config/models.yaml:
  defaults.backends.sglang.backend_version: nightly-dev-cu13-20260110-8a1be0dc

To apply this change, run:
  uv run python scripts/run_bench.py --model ~/models/... --backend sglang
```

## Notes

- CUDA 13 nightlies use the tag pattern: `nightly-dev-cu13-{YYYYMMDD}-{short_hash}`
- For stable releases, use tags like `v0.5.7-cu130-runtime` or `latest-cu130-runtime`
- The short hash is derived from the SGLang commit (not the full SHA like vLLM)
