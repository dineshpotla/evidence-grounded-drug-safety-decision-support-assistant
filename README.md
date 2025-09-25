# Evidence-Grounded Drug Safety Decision Support Assistant

Patient-specific drug safety assistant using OpenFDA labels, PubMed evidence, and FAERS post-marketing signals.

## Architecture implemented

- NVIDIA extraction layer (intent + entity extraction with deterministic fallback)
- Agentic RAG layer with:
  - MedCPT embeddings (`ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder`)
  - FAISS vector recall (fallback to NumPy vector search)
  - NVIDIA/Anthropic reranking + claim generation (fallback to deterministic mode)
- Safety controls:
  - Minimal-context follow-up questioning (max 3 missing slots)
  - Hallucination guard (citation checks, support consistency, PRR numeric checks)
  - Explainable risk scoring (label + literature + FAERS + patient modifiers)
- Prompt governance:
  - Versioned prompt templates (`prompts/v1`, `prompts/v2`)
  - Prompt registry (`prompt_registry.yaml`)
- Evaluation framework:
  - Benchmark schema + sample benchmark
  - Synthetic benchmark generator for 3k+ query runs
  - Multi-metric evaluator

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

### Enable APIs and ML backends (optional but recommended)

```bash
pip install -e '.[dev,api,ml]'
```

Set env vars in `.env`:

- `ANTHROPIC_API_KEY` for Claude reranking/generation
- `NVIDIA_API_KEY` for NVIDIA-hosted model backend (default: `nvidia/llama-3.3-nemotron-super-49b-v1.5`)
- `NVIDIA_EXTRACT_MODEL` for intent/entity extraction (recommended: `mistralai/mistral-nemotron`)
- `NVIDIA_JUDGE_MODEL` for LLM-as-judge scoring (recommended: `mistralai/mistral-nemotron`)
- `NCBI_API_KEY` (optional) for higher PubMed throughput
- `ENABLE_MEDCPT_MODELS=1` to load real MedCPT encoders
- `ENABLE_LLM_JUDGE=1` to score generated answers with LLM-as-judge in evaluation
- `USE_PERSISTENT_INDEX=1` to enable persistent hybrid retrieval
- `PERSISTENT_INDEX_DIR` path to prebuilt index artifacts

Without these keys, the assistant still runs with deterministic fallbacks.

## Run

Run tests:

```bash
python3 -m pytest -q
```

Run app:

```bash
streamlit run src/drug_safety_assistant/app.py
```

UI highlights:

- Structured clinical output tabs (summary, evidence, traceability, export)
- Risk banner + gauge with support-ratio confidence indicator
- Downloadable JSON/Markdown response artifacts
- Session history in sidebar for rapid comparison across assessments

Run evaluation (sample benchmark):

```bash
python3 evaluation/scripts/run_eval.py --benchmark evaluation/benchmarks/sample_benchmark.jsonl --workers 6
```

Run evaluation with LLM-as-judge enabled:

```bash
python3 evaluation/scripts/run_eval.py --benchmark evaluation/benchmarks/sample_benchmark.jsonl --workers 6 --llm-judge
```

Benchmark NVIDIA extraction models:

```bash
python3 evaluation/scripts/benchmark_nvidia_extract_models.py
```

Generate a 3,000-query benchmark and evaluate:

```bash
python3 evaluation/scripts/generate_synthetic_benchmark.py --num-queries 3000 --output evaluation/benchmarks/synthetic_3000.jsonl
python3 evaluation/scripts/run_eval.py --benchmark evaluation/benchmarks/synthetic_3000.jsonl --workers 8 --output evaluation/results/eval_3000.json
```

Run full project pipeline in one command:

```bash
./scripts/start_full_project.sh
```

Build production corpus + persistent index:

```bash
python3 scripts/build_corpus_from_drug_list.py --drugs data/drug_list.txt --output artifacts/corpus/drug_safety_corpus.jsonl
python3 scripts/build_persistent_index.py --corpus artifacts/corpus/drug_safety_corpus.jsonl --index-dir artifacts/persistent_index --batch-size 128
```

Enable persistent index in `.env`:

```bash
USE_PERSISTENT_INDEX=1
PERSISTENT_INDEX_DIR=./artifacts/persistent_index
LEXICAL_CANDIDATE_K=300
HYBRID_DENSE_WEIGHT=0.7
```

Detailed scaling notes: `docs/production_scaling.md`.

## Repository layout

```text
drug-safety-assistant/
  prompts/
    v1/
    v2/
  src/drug_safety_assistant/
    llm/
    retrieval/
    pipeline/
  evaluation/
    benchmarks/
    scripts/
    results/
  skills/
    custom-retrieval/
  docs/
  model_card.md
  prompt_registry.yaml
```

## Safety notes

- This project provides decision support and not medical diagnosis or prescribing guidance.
- Dosing instructions are intentionally omitted.
- Missing or low-quality evidence is explicitly surfaced in the uncertainty section.
