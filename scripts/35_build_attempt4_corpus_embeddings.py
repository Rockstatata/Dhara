"""Full-corpus embed for attempt 4 (act-title template + recovered fine-tuned weights).

    python scripts/35_build_attempt4_corpus_embeddings.py

Long-running (CPU, no GPU on this machine): benchmarked at ~0.39s/doc for the
same checkpoint family, so 39,484 chunks is on the order of hours. Intended to
run in the background while other phases proceed.

Document template mirrors notebooks/colab_bge_m3_finetune.ipynb cell 3 exactly
(act_title_bn + act_title_en + provision_title_bn + text_bn, space-joined, no
extra normalization -- text_bn is already light-normalized at corpus build
time) so the corpus distribution matches what the recovered weights were
actually trained against.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

torch.set_num_threads(8)  # avoid oversubscription if anything else is running

sys.stdout.reconfigure(encoding="utf-8")

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
CHECKPOINT = pathlib.Path("models/checkpoint_bge_m3_attempt4")
OUT_DIR = pathlib.Path("models/index_bge_m3_attempt4_v1")
BATCH_SIZE = 16  # this machine has ~16GB RAM with ~9GB already in use at idle;
                 # batch 64 caused severe swap thrashing (measured 20x slowdown,
                 # 660s/batch instead of the ~25s a clean run gets at batch 16)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def document(c: dict) -> str:
    bits = []
    for key in ("act_title_bn", "act_title_en"):
        v = (c.get(key) or "").strip()
        if v and v not in bits:
            bits.append(v)
    t = (c.get("provision_title_bn") or "").strip()
    if t:
        bits.append(t)
    bits.append(c["text_bn"])
    return " ".join(b for b in bits if b).strip()


def main() -> None:
    assert CHECKPOINT.exists(), f"{CHECKPOINT} not found -- run scripts/33_merge_attempt4_checkpoint.py first"

    chunks = read_jsonl(CORPUS)
    print(f"{len(chunks)} chunks")
    cid = [c["chunk_id"] for c in chunks]
    texts = [document(c) for c in chunks]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(CHECKPOINT), device="cpu")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)
    dt = time.time() - t0
    print(f"encoded {embeddings.shape} in {dt/60:.1f} min")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    (OUT_DIR / "chunk_ids.json").write_text(json.dumps(cid), encoding="utf-8")

    manifest = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_source": "recovered from data/finetune-results/dhara_bge_m3_lora.zip "
                              "via scripts/33_merge_attempt4_checkpoint.py -- the zip's own "
                              "export was malformed (raw unmerged PEFT state_dict under a "
                              "plain-model config.json); this is the LoRA-merged repair",
        "max_seq_length": 512,
        "use_title": True,
        "use_act_title": True,
        "query_prefix": "",
        "passage_prefix": "",
        "dim": int(embeddings.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "n_chunks": int(embeddings.shape[0]),
        "corpus_file": str(CORPUS),
        "fine_tuned": True,
        "train_file": "data/processed/train_anchor_v1_negatives.jsonl",
        "train_variant": "anchor",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["query", "value"]},
        "document_template": "act_title_bn + act_title_en + provision_title_bn + text_bn",
        "built_by": "scripts/35_build_attempt4_corpus_embeddings.py, weights from "
                    "notebooks/colab_bge_m3_finetune.ipynb (stacked act-title + fine-tune)",
        "notes": "Attempt 4: act-title document template stacked with fine-tuning, the "
                 "combination DECISIONS.md flagged as never having been measured "
                 "(attempt 3 kept the bare-section template to isolate the weights). "
                 "Corpus embedded on CPU locally after recovering the checkpoint from a "
                 "malformed export -- see checkpoint_source above.",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR}/embeddings.npy, chunk_ids.json, manifest.json")


if __name__ == "__main__":
    main()
