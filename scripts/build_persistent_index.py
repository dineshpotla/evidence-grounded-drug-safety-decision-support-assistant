from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_safety_assistant.retrieval.persistent_index import PersistentCorpusIndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persistent hybrid retrieval index")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help=(
            "Path to corpus JSONL with fields "
            "source,citation_id,title,text,metadata,strength_score"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Output directory for persistent index artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = PersistentCorpusIndexBuilder(index_dir=args.index_dir)
    manifest = builder.build_from_jsonl(corpus_path=args.corpus, batch_size=args.batch_size)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
