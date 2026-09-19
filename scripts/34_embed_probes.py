"""Encode the v2 probe questions with a given checkpoint, write a cached .npy.

    python scripts/34_embed_probes.py --checkpoint BAAI/bge-m3 \
        --out models/index_bge_m3_zeroshot_v1/probe_query_v2.npy
    python scripts/34_embed_probes.py --checkpoint models/checkpoint_bge_m3_attempt4 \
        --out models/index_bge_m3_attempt4_v1/probe_query_v2.npy

Mirrors exactly how every existing probe_query.npy in this repo was produced
(notebooks/colab_bge_m3_eval.ipynb, colab_bge_m3_finetune.ipynb cell 12): raw
question_bn text, no query prefix (bge-m3 takes none), no normalize.light()
call. Matching that -- rather than introducing light() here for the first time
-- keeps v2 numbers comparable to v1 under the same methodology; the
BiEncoderRetriever.search() path that does apply light() is the demo-serving
path, not what built any cached probe_query.npy on disk.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probes", type=pathlib.Path,
                    default=pathlib.Path("data/processed/probe_questions_v2.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    probes = read_jsonl(args.probes)
    print(f"{len(probes)} probe questions from {args.probes}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.checkpoint, device="cpu")
    Q = model.encode(
        [p["question_bn"] for p in probes],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, Q)
    print(f"wrote {args.out}  shape={Q.shape}")


if __name__ == "__main__":
    main()
