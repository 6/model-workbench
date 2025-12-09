# Eval Implementation Plan

Add accuracy benchmarks (IFEval, HumanEval) to compare quantized vs full model performance.

## Why These Benchmarks

| Benchmark | What It Measures | API Calls | Why It Matters for Quantization |
|-----------|------------------|-----------|--------------------------------|
| **IFEval** | Instruction following accuracy | 541 | Ensures quantized models still follow prompts correctly |
| **HumanEval** | Code generation (pass@1) | 164 | Tests if code ability degrades with quantization |

**Total: 705 API calls** - fast to run while covering instruction following and code generation.

**Note on HumanEval:** Standard papers generate n=200 samples per problem (32,800 calls!) to calculate statistical pass@k. We use **n=1 with temperature=0** (deterministic) for pass@1, which is sufficient for comparing quantization degradation and matches most leaderboard methodologies.

## Architecture

```
scripts/
├── run_eval.py              # Main entry point (like run_bench.py)
├── eval_model_wrapper.py    # DeepEvalBaseLLM wrapper for local servers
└── bench_utils.py           # Existing utilities (reuse)

evals/                       # Eval results (new, top-level)
└── 2025-01-15_org_model_ifeval-humaneval.json
```

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps
    "deepeval>=2.0.0",
]
```

## Implementation

### Step 1: Model Wrapper (`scripts/eval_model_wrapper.py`)

DeepEval requires a `DeepEvalBaseLLM` subclass. This wraps our local server:

```python
"""DeepEval model wrapper for local vLLM/llama.cpp servers."""

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class LocalServerLLM(DeepEvalBaseLLM):
    """Wrapper for OpenAI-compatible local servers (vLLM, llama.cpp)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "local-model",
        temperature: float = 0.0,  # Deterministic for evals
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def load_model(self):
        """No-op since server is already running."""
        pass

    def generate(self, prompt: str) -> str:
        """Generate a single response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,  # HumanEval needs longer outputs for code
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """Async generation (falls back to sync)."""
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        """Batch generation for efficiency."""
        return [self.generate(p) for p in prompts]

    def get_model_name(self) -> str:
        return self.model_name
```

### Step 2: Eval Runner (`scripts/run_eval.py`)

```python
#!/usr/bin/env python3
"""
Model Workbench Eval Runner

Runs IFEval and HumanEval benchmarks against local model servers.

Examples:
  # Run IFEval on a vLLM server (auto-starts server)
  uv run python scripts/run_eval.py --model ~/models/org/model --benchmark ifeval

  # Run HumanEval on already-running server
  uv run python scripts/run_eval.py --model ~/models/org/model --benchmark humaneval --no-autostart

  # Run both benchmarks
  uv run python scripts/run_eval.py --model ~/models/org/model --benchmark ifeval humaneval
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from deepeval.benchmarks import IFEval, HumanEval

from bench_utils import (
    ROOT,
    sanitize,
    extract_repo_id,
    get_gpu_info,
    resolve_model_path,
    detect_model_format,
    log,
)
from eval_model_wrapper import LocalServerLLM


EVAL_RESULTS_ROOT = ROOT / "evals"


def run_ifeval(model: LocalServerLLM) -> dict:
    """Run IFEval benchmark (541 instruction-following prompts)."""
    log("Running IFEval (541 prompts)...")
    benchmark = IFEval()
    benchmark.evaluate(model=model)

    return {
        "benchmark": "ifeval",
        "questions": 541,
        "overall_score": benchmark.overall_score,
    }


def run_humaneval(model: LocalServerLLM, n_samples: int = 1) -> dict:
    """Run HumanEval benchmark (164 coding problems).

    Args:
        n_samples: Number of code samples per problem. Default 1 (pass@1 with temp=0).
                   Standard papers use n=200 for statistical pass@k, but that's 32,800 calls.
                   For quick quantization comparison, n=1 with temp=0 is sufficient.
    """
    log(f"Running HumanEval (164 problems, n={n_samples})...")
    benchmark = HumanEval(n=n_samples)
    benchmark.evaluate(model=model)

    return {
        "benchmark": "humaneval",
        "questions": 164,
        "n_samples": n_samples,
        "overall_score": benchmark.overall_score,  # pass@1 score
    }


def save_results(results: dict, repo_id: str, benchmark_name: str):
    """Save results to evals/."""
    EVAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_name = sanitize(repo_id)
    filename = f"{date_str}_{safe_name}_{benchmark_name}.json"
    path = EVAL_RESULTS_ROOT / filename

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"Results saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Run evals on local models")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=["ifeval", "humaneval"],
        default=["ifeval", "humaneval"],
        help="Benchmark(s) to run",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Don't auto-start server (assume already running)",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    repo_id = extract_repo_id(model_path)
    model_format = detect_model_format(model_path)

    log(f"Model: {repo_id} ({model_format})")

    # TODO: Add server auto-start logic (copy from run_bench_vllm_server.py)
    # For now, assume server is running

    base_url = f"http://localhost:{args.port}/v1"
    model = LocalServerLLM(base_url=base_url, model_name=repo_id)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "model_path": str(model_path),
        "model_format": model_format,
        "gpu_info": get_gpu_info(),
        "benchmarks": {},
    }

    for benchmark in args.benchmark:
        if benchmark == "ifeval":
            result = run_ifeval(model)
        elif benchmark == "humaneval":
            result = run_humaneval(model)

        all_results["benchmarks"][benchmark] = result
        log(f"{benchmark.upper()} score: {result['overall_score']:.3f}")

    # Save combined results
    benchmarks_str = "-".join(args.benchmark)
    save_results(all_results, repo_id, benchmarks_str)


if __name__ == "__main__":
    main()
```

### Step 3: Usage

```bash
# Run both benchmarks (default, 705 questions total)
uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8

# IFEval only (541 prompts)
uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8 --benchmark ifeval

# HumanEval only (164 coding problems)
uv run python scripts/run_eval.py --model ~/models/zai-org/GLM-4.6V-FP8 --benchmark humaneval
```

### Step 4: Compare Results

Results are JSON in `evals/`. Quick comparison:

```bash
# Compare two models on IFEval
jq '.benchmarks.ifeval.overall_score' evals/*GLM-4.6V-FP8*.json
jq '.benchmarks.ifeval.overall_score' evals/*GLM-4.6V-GGUF*.json

# Compare HumanEval scores
jq '.benchmarks.humaneval.overall_score' evals/*GLM-4.6V-FP8*.json
jq '.benchmarks.humaneval.overall_score' evals/*GLM-4.6V-GGUF*.json
```

## Result Schema

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "repo_id": "zai-org/GLM-4.6V-FP8",
  "model_path": "/home/user/models/zai-org/GLM-4.6V-FP8",
  "model_format": "safetensors",
  "gpu_info": { ... },
  "benchmarks": {
    "ifeval": {
      "benchmark": "ifeval",
      "questions": 541,
      "overall_score": 0.85
    },
    "humaneval": {
      "benchmark": "humaneval",
      "questions": 164,
      "n_samples": 1,
      "overall_score": 0.62
    }
  }
}
```

## Future Enhancements

1. **Auto-start server** - Copy server lifecycle from `run_bench_vllm_server.py`
2. **llama.cpp support** - Adapt wrapper for `/completion` endpoint
3. **More benchmarks** - MMLU (general knowledge), GSM8K (math), TruthfulQA (factual)
4. **Comparison script** - `scripts/compare_evals.py` to diff two result files
5. **CI integration** - Run evals on model updates

## References

- [DeepEval Docs](https://docs.deepeval.com/)
- [IFEval Paper](https://arxiv.org/abs/2311.07911)
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
