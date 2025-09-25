# Model Card: Drug Safety Assistant (Baseline)

## Intended use

Clinical information support for drug safety risk awareness using retrieved evidence.

## Not intended use

- Prescribing decisions without clinician judgment
- Emergency triage
- Dose calculation

## Evidence sources

- OpenFDA labels
- PubMed
- FAERS

## Modeling stack

- Intent/entity extraction: NVIDIA NIM extractor wrapper (with deterministic fallback)
- Retrieval embeddings: MedCPT (query + article encoders)
- Vector retrieval: FAISS (fallback NumPy search)
- Reranking/generation: NVIDIA/Anthropic agent wrapper (fallback deterministic claims)

## Known limitations

- Retrieval quality depends on query synonym coverage
- FAERS is observational and subject to reporting bias
- Heuristic claim-evidence consistency check may under-detect nuanced unsupported claims

## Evaluation signals

- Citation precision
- Unsupported claim rate
- Hallucination rate
- Risk-level agreement with benchmark label
