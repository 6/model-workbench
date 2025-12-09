# Troubleshooting

## Qwen3-Omni vLLM Error: PyMerges TypeError

### Symptom

When running Qwen3-Omni-30B-A3B-Instruct with vLLM server mode:

```
TypeError: argument 'merges' failed to extract enum field PyMerges::Merges.0,
caused by TypeError 'list' object cannot be converted to PyString
```

Stacktrace shows error in:
```
AsyncLLM.from_vllm_config > get_tokenizer > AutoTokenizer.from_pretrained
  -> transformers/models/qwen2/tokenization_qwen2.py
```

### Context

- **Models affected**: Qwen3-Omni-30B-A3B-Instruct (possibly other Qwen3-Omni variants)
- **Models that work fine**: Qwen3-30B-A3B-Instruct-2507-FP8, openai/gpt-oss-20b, GLM-4.6V-FP8
- **Engine**: vLLM server mode (`vllm serve`)

### Investigation

1. **Not GLM-specific flags** - Other models work with the same vLLM flags
2. **Not tensor parallelism** - Single GPU also fails
3. **Tokenizer-level issue** - Error occurs during tokenizer loading, before model loads

### Attempted Fixes

| Fix | Result |
|-----|--------|
| Remove GLM-specific vLLM flags | No change |
| Upgrade tokenizers to 0.22.1 (PyPI) | No change |
| Upgrade tokenizers to 0.22.2 (GitHub) | No change |

### Related Issues

- [vLLM #25834](https://github.com/vllm-project/vllm/issues/25834) - Qwen3 Omni requires transformers >= 4.57
- [vLLM #27932](https://github.com/vllm-project/vllm/issues/27932) - Qwen3OmniMoeProcessor AttributeError
- [QwenLM/Qwen3 #720](https://github.com/QwenLM/Qwen3/issues/720) - PyMerges error when modifying tokenizer

### Environment

```toml
# pyproject.toml overrides
[tool.uv]
override-dependencies = [
    "transformers @ git+https://github.com/huggingface/transformers.git",
    "tokenizers @ git+https://github.com/huggingface/tokenizers.git@v0.22.2#subdirectory=bindings/python",
]
```

### Status

**Unresolved** - Likely a compatibility issue between:
- transformers (git master)
- tokenizers 0.22.2
- vLLM's Qwen3-Omni support

May need to wait for upstream fixes in vLLM or transformers.

### Workaround

Use the non-Omni variant: `Qwen3-30B-A3B-Instruct-2507-FP8` works correctly.
