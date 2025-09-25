# Production-Scale Retrieval

This project now supports release-style retrieval with a persistent hybrid index:

- Lexical candidate generation: SQLite FTS5
- Dense scoring: MedCPT (or fallback) embeddings
- Hybrid rank: dense + lexical weighted score
- Optional agentic rerank: NVIDIA/Anthropic model

## Why this is scalable

- Query-time retrieval runs against prebuilt index artifacts.
- You do not embed millions of documents on each request.
- Index refresh can run as offline batch jobs.

## Index artifacts

By default under `PERSISTENT_INDEX_DIR`:

- `manifest.json`
- `chunks.jsonl`
- `vectors.npy`
- `chunks.db`

## Corpus schema

Corpus input is JSONL with fields:

```json
{
  "source": "openfda|pubmed|faers|...",
  "citation_id": "SOURCE:ID",
  "title": "Document title",
  "text": "Raw document text/snippet",
  "metadata": {},
  "strength_score": 0
}
```

## Build corpus from drug list

```bash
python3 scripts/build_corpus_from_drug_list.py \
  --drugs data/drug_list.txt \
  --output artifacts/corpus/drug_safety_corpus.jsonl \
  --pubmed-per-drug 25 \
  --label-per-drug 12
```

## Build persistent index

```bash
python3 scripts/build_persistent_index.py \
  --corpus artifacts/corpus/drug_safety_corpus.jsonl \
  --index-dir artifacts/persistent_index \
  --batch-size 128
```

## Enable in runtime

Set in `.env`:

```bash
USE_PERSISTENT_INDEX=1
PERSISTENT_INDEX_DIR=./artifacts/persistent_index
LEXICAL_CANDIDATE_K=300
HYBRID_DENSE_WEIGHT=0.7
```

## Refresh strategy

- Nightly full refresh: rebuild corpus and index.
- Incremental refresh: append new documents and run partial rebuild by shard.
- Keep previous index dir and swap atomically after successful build.
