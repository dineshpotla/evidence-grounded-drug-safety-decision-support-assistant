# custom-retrieval

Use this skill to run evidence retrieval for drug safety questions using OpenFDA, PubMed, and FAERS with MedCPT + FAISS + reranking.

## Inputs

- Clinical question
- Optional patient context: age group, pregnancy status/trimester, kidney status, liver status, med list

## Workflow

1. Extract intent and entities.
2. Validate required slots by intent.
3. Retrieve source documents from:
   - OpenFDA label API
   - PubMed E-utilities
   - OpenFDA FAERS endpoint
4. Chunk evidence by semantic span with citation-preserving IDs.
5. Embed chunks with MedCPT:
   - Query encoder: `ncbi/MedCPT-Query-Encoder`
   - Document encoder: `ncbi/MedCPT-Article-Encoder`
6. Recall candidates from FAISS index.
7. Rerank candidates agentically.
8. Return top grounded evidence with citation IDs.

## Constraints

- Do not request or store direct PII.
- Drop unsupported claims.
- Every major clinical claim must map to at least one citation ID.
- Do not provide dosing instructions.

## Output contract

- Ranked chunks with source + citation IDs
- Grounded claims
- Uncertainty statement if evidence is sparse or conflicting
