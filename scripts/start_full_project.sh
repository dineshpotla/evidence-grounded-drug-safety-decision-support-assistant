#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

BENCHMARK="${1:-$ROOT/evaluation/benchmarks/synthetic_3000.jsonl}"
CORPUS="${2:-${CORPUS_JSONL_PATH:-$ROOT/artifacts/corpus/drug_safety_corpus.jsonl}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$ROOT/evaluation/results/eval_full_${TIMESTAMP}.json"
LOGFILE="$ROOT/logs/full_eval_${TIMESTAMP}.log"

mkdir -p "$ROOT/logs" "$ROOT/run"

printf "[1/6] Installing project dependencies...\n"
python3 -m pip install -e '.[dev,api]' >/dev/null

printf "[2/6] Running lint + tests...\n"
python3 -m ruff check .
python3 -m pytest -q tests

if [[ ! -f "$BENCHMARK" ]]; then
  printf "[3/6] Benchmark missing. Generating 3000-query benchmark...\n"
  python3 evaluation/scripts/generate_synthetic_benchmark.py \
    --num-queries 3000 \
    --output "$BENCHMARK"
else
  printf "[3/6] Using existing benchmark: %s\n" "$BENCHMARK"
fi

if [[ "${USE_PERSISTENT_INDEX:-0}" == "1" ]]; then
  INDEX_DIR="${PERSISTENT_INDEX_DIR:-$ROOT/artifacts/persistent_index}"
  MANIFEST="$INDEX_DIR/manifest.json"

  if [[ ! -f "$MANIFEST" ]]; then
    if [[ ! -f "$CORPUS" ]]; then
      printf "[4/6] Persistent index requested but corpus missing: %s\n" "$CORPUS"
      printf "       Build corpus first with scripts/build_corpus_from_drug_list.py\n"
      exit 1
    fi

    printf "[4/6] Building persistent index at %s ...\n" "$INDEX_DIR"
    python3 scripts/build_persistent_index.py \
      --corpus "$CORPUS" \
      --index-dir "$INDEX_DIR" \
      --batch-size 128
  else
    printf "[4/6] Using existing persistent index: %s\n" "$INDEX_DIR"
  fi
else
  printf "[4/6] Persistent index disabled (USE_PERSISTENT_INDEX=0).\n"
fi

printf "[5/6] Running full evaluation...\n"
if [[ "${ENABLE_LLM_JUDGE:-0}" == "1" ]]; then
  JUDGE_FLAG="--llm-judge"
else
  JUDGE_FLAG="--no-llm-judge"
fi
PYTHONUNBUFFERED=1 python3 -u evaluation/scripts/run_eval.py \
  --benchmark "$BENCHMARK" \
  --workers 8 \
  --judge-workers 4 \
  "$JUDGE_FLAG" \
  --output "$OUTPUT" | tee "$LOGFILE"

printf "[6/6] Completed.\n"
printf "Log: %s\n" "$LOGFILE"
printf "Output: %s\n" "$OUTPUT"
