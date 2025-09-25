# Prompt Governance and Evaluation

## Prompt lifecycle

1. Add/modify template under `prompts/<version>/`
2. Update `prompt_registry.yaml`
3. Run benchmark evaluation
4. Include before/after examples and metric deltas in PR

## Required checks per prompt/model update

- Citation precision does not regress
- Supported-claim rate does not regress
- Hallucination rate does not increase materially
- Risk agreement remains within tolerance

## Evaluation metrics tracked

- Retrieval recall proxy
- Citation precision
- Supported-claim rate
- Hallucination rate
- Risk-level agreement
- Helpfulness and conciseness

## Benchmarking

- Baseline sample set: `evaluation/benchmarks/sample_benchmark.jsonl`
- Large synthetic set: generated via `generate_synthetic_benchmark.py`
- Output artifacts: `evaluation/results/*.json`

## A/B variants

- Prompt variant (`v1`, `v2`, future `v3`)
- Reranking strategy (heuristic vs Claude)
- Retrieval stack (raw retrieval vs MedCPT + FAISS + rerank)
