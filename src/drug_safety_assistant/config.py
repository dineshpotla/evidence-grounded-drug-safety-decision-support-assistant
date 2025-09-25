from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency path
    load_dotenv = None

if load_dotenv is not None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openfda_base_url: str = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov")
    pubmed_eutils_base: str = os.getenv(
        "PUBMED_EUTILS_BASE",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    )
    ncbi_api_key: str = os.getenv("NCBI_API_KEY", "")
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_model: str = os.getenv(
        "NVIDIA_MODEL",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    )
    nvidia_extract_model: str = os.getenv(
        "NVIDIA_EXTRACT_MODEL",
        "mistralai/mistral-nemotron",
    )
    nvidia_judge_model: str = os.getenv(
        "NVIDIA_JUDGE_MODEL",
        "mistralai/mistral-nemotron",
    )
    medcpt_query_model: str = os.getenv(
        "MEDCPT_QUERY_MODEL",
        "ncbi/MedCPT-Query-Encoder",
    )
    medcpt_doc_model: str = os.getenv(
        "MEDCPT_DOC_MODEL",
        "ncbi/MedCPT-Article-Encoder",
    )
    enable_medcpt_models: bool = os.getenv("ENABLE_MEDCPT_MODELS", "0") == "1"
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "14"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "7"))
    use_faiss: bool = os.getenv("USE_FAISS", "1") == "1"
    use_persistent_index: bool = os.getenv("USE_PERSISTENT_INDEX", "0") == "1"
    persistent_index_dir: str = os.getenv(
        "PERSISTENT_INDEX_DIR",
        "./artifacts/persistent_index",
    )
    lexical_candidate_k: int = int(os.getenv("LEXICAL_CANDIDATE_K", "300"))
    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
    drug_dictionary_path: str = os.getenv("DRUG_DICTIONARY_PATH", "./data/drug_list.txt")
    enable_llm_judge: bool = os.getenv("ENABLE_LLM_JUDGE", "0") == "1"
    llm_judge_timeout_seconds: int = int(
        os.getenv("LLM_JUDGE_TIMEOUT_SECONDS", os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    )


settings = Settings()
