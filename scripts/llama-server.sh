#!/bin/sh
set -eu

# llama-serve.sh
#
# Default behavior (no flags):
#   - Uses llama.cpp defaults for multi-GPU behavior
#   - Uses -ngl 999
#
# New flag:
#   --split-evenly
#     Adds explicit multi-GPU split settings:
#       --split-mode layer
#       --tensor-split 1,1 (or 1,1,1... based on CUDA_VISIBLE_DEVICES)
#
# Usage (backward compatible):
#   ./llama-serve.sh <relative-path-under-~/models> [port] [ngl]
#
# Preferred modern usage:
#   ./llama-serve.sh [--port N] [--ngl N] [--split-evenly] <model-or-dir>
#
# Examples:
#   ./llama-serve.sh unsloth/GLM-4.6-GGUF/UD-Q4_K_XL
#   ./llama-serve.sh --split-evenly unsloth/GLM-4.6-GGUF/UD-Q4_K_XL
#   ./llama-serve.sh --split-evenly --port 8081 --ngl 110 unsloth/GLM-4.6-GGUF/UD-Q4_K_XL
#   ./llama-serve.sh unsloth/GLM-4.6-GGUF/UD-Q4_K_XL 8080 999

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/models}"

# Defaults
PORT=""
NGL=""
SPLIT_EVENLY=0

print_usage() {
  cat >&2 <<EOF
Usage:
  $0 [--port N] [--ngl N] [--split-evenly] <path-under-~/models>
  $0 <path-under-~/models> [port] [ngl]

Flags:
  --port N           Port for llama-server (default 8080)
  --ngl N            Number of GPU layers to offload (default 999)
  --split-evenly     Add explicit multi-GPU split:
                     --split-mode layer
                     --tensor-split 1,1 (auto-sized if CUDA_VISIBLE_DEVICES set)
  -h, --help         Show this help

Notes:
  - If <path> is a directory, script picks:
      1) *-00001-of-*.gguf
      2) otherwise a single non-sharded *.gguf
      3) otherwise errors
EOF
}

# --- Parse args ---
# We support:
#   1) New style flags anywhere before the model argument
#   2) Old style positional: <model> [port] [ngl]
#
# We'll do a simple long-option parser.
ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)
      [ $# -ge 2 ] || { echo "Error: --port requires a value" >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --ngl)
      [ $# -ge 2 ] || { echo "Error: --ngl requires a value" >&2; exit 2; }
      NGL="$2"
      shift 2
      ;;
    --split-evenly)
      SPLIT_EVENLY=1
      shift 1
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --) # end of flags
      shift
      break
      ;;
    -*) # unknown flag
      echo "Error: unknown option: $1" >&2
      print_usage
      exit 2
      ;;
    *) # first non-flag is model/path
      break
      ;;
  esac
done

# Remaining args now start with model/path (new style)
# or could be old style where user didn't pass flags.
if [ $# -lt 1 ]; then
  print_usage
  exit 2
fi

REL="$1"
shift 1

# Old positional fallback:
# If PORT/NGL not provided via flags, and extra args exist, use them.
if [ -z "${PORT}" ] && [ $# -ge 1 ]; then
  PORT="$1"
  shift 1
fi
if [ -z "${NGL}" ] && [ $# -ge 1 ]; then
  NGL="$1"
  shift 1
fi

# Final defaults if still unset
PORT="${PORT:-8080}"
NGL="${NGL:-999}"

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

# Build an even tensor-split string.
# If CUDA_VISIBLE_DEVICES is set, use its count.
# Otherwise assume 2 GPUs.
build_even_tensor_split() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    old_ifs="$IFS"
    IFS=","
    set -- $CUDA_VISIBLE_DEVICES
    IFS="$old_ifs"
    n="$#"
  else
    n=2
  fi

  # Generate "1,1,1..." length n
  i=1
  out="1"
  while [ "$i" -lt "$n" ]; do
    out="$out,1"
    i=$((i + 1))
  done
  printf "%s" "$out"
}

EXTRA_SPLIT_ARGS=""
if [ "$SPLIT_EVENLY" -eq 1 ]; then
  TENSOR_SPLIT="$(build_even_tensor_split)"
  EXTRA_SPLIT_ARGS="--split-mode layer --tensor-split $TENSOR_SPLIT"
fi

echo "llama-server:   $LLAMA_SERVER"
echo "model:          $MODEL_FILE"
echo "port:           $PORT"
echo "ngl:            $NGL"
if [ "$SPLIT_EVENLY" -eq 1 ]; then
  echo "split-evenly:   enabled"
  echo "split-mode:     layer"
  echo "tensor-split:   ${TENSOR_SPLIT:-1,1}"
else
  echo "split-evenly:   disabled (llama.cpp defaults)"
fi

# shellcheck disable=SC2086
exec "$LLAMA_SERVER" \
  -m "$MODEL_FILE" \
  --port "$PORT" \
  -ngl "$NGL" \
  $EXTRA_SPLIT_ARGS
