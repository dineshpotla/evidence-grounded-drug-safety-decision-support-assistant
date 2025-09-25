# System Architecture

```text
User Input
  -> NVIDIA Intent + Entity Extractor
  -> Context Sufficiency Checker
  -> Follow-up Question Engine (if needed)
  -> Retrieval Route:
     -> Persistent Hybrid Index (SQLite FTS + vectors) OR
     -> Live Source Retrieval (OpenFDA + PubMed + FAERS)
  -> Evidence Chunking
  -> MedCPT Embeddings
  -> Vector Recall
  -> NVIDIA/Anthropic Agentic Reranking
  -> Claim Generation
  -> Hallucination Guard
  -> Risk Scoring
  -> Final Structured Response
```

## Components

- `llm/nvidia_extractor.py`: intent/entity extraction with fallback
- `llm/claude.py`: NVIDIA/Anthropic reranking and claim generation wrapper with fallback
- `retrieval/aggregator.py`: source-specific retrieval entry point
- `retrieval/persistent_index.py`: persistent index builder and hybrid retriever
- `retrieval/corpus.py`: evidence chunk generation
- `retrieval/embeddings.py`: MedCPT embedding wrapper
- `retrieval/vector_index.py`: FAISS-backed vector index (or NumPy fallback)
- `retrieval/agentic.py`: multi-stage retrieval orchestration
- `question_engine.py`: minimal-context questioning (max 3)
- `hallucination.py`: citation/support/numeric verification
- `risk_scoring.py`: explainable weighted risk model
- `pipeline/orchestrator.py`: end-to-end orchestration
