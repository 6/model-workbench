#!/bin/sh
set -eu

# Usage:
#   llama-serve.sh <relative-path-under-~/models> [port] [ngl]
#
# Examples:
#   ./llama-serve.sh unsloth/GLM-4.5-Air-GGUF/UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf
#   ./llama-serve.sh unsloth/GLM-4.5-Air-GGUF 8080 999
#
# If you pass a directory, this script will try to find:
#   1) a shard matching *-00001-of-*.gguf
#   2) otherwise a single non-sharded *.gguf
#   3) otherwise error

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/models}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path-under-~/models> [port] [ngl]" >&2
  exit 2
fi

REL="$1"
PORT="${2:-8080}"
NGL="${3:-999}"

TARGET="$MODELS_ROOT/$REL"

if [ ! -e "$TARGET" ]; then
  echo "Error: not found: $TARGET" >&2
  exit 1
fi

pick_model_file() {
  dir="$1"

  # Prefer first shard
  shard="$(find "$dir" -type f -name "*-00001-of-*.gguf" 2>/dev/null | sort | head -n 1 || true)"
  if [ -n "$shard" ]; then
    echo "$shard"
    return 0
  fi

  # Otherwise, if exactly one non-sharded gguf exists, use it
  # We'll approximate by filtering out the shard pattern.
  ggufs="$(find "$dir" -type f -name "*.gguf" 2>/dev/null | grep -Ev -- '-[0-9]{5}-of-[0-9]{5}\.gguf$' || true)"
  count="$(printf "%s\n" "$ggufs" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [ "$count" -eq 1 ]; then
    printf "%s\n" "$ggufs" | sed '/^$/d'
    return 0
  fi

  if [ "$count" -gt 1 ]; then
    echo "Error: multiple non-sharded GGUF files found under: $dir" >&2
    echo "Please pass a specific .gguf path." >&2
    exit 1
  fi

  # If we got here, maybe only shards exist but not 00001 pattern (unlikely)
  any="$(find "$dir" -type f -name "*.gguf" 2>/dev/null | sort | head -n 1 || true)"
  if [ -n "$any" ]; then
    echo "Error: GGUF files found, but couldn't pick a clear entrypoint under: $dir" >&2
    echo "Try passing the exact *-00001-of-*.gguf file path." >&2
    exit 1
  fi

  echo "Error: no .gguf files found under: $dir" >&2
  exit 1
}

if [ -d "$TARGET" ]; then
  MODEL_FILE="$(pick_model_file "$TARGET")"
else
  MODEL_FILE="$TARGET"
fi

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Error: llama-server not executable at: $LLAMA_SERVER" >&2
  echo "Set LLAMA_SERVER=/path/to/llama-server or build llama.cpp first." >&2
  exit 1
fi

echo "llama-server: $LLAMA_SERVER"
echo "model:        $MODEL_FILE"
echo "port:         $PORT"
echo "ngl:          $NGL"

exec "$LLAMA_SERVER" \
  -m "$MODEL_FILE" \
  --port "$PORT" \
  -ngl "$NGL"
