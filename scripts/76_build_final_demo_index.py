"""Build the retrieval index for the final delivered demo, from the v6 fine-tuned checkpoint.

    python scripts/76_build_final_demo_index.py

Run this once, on a GPU box (same requirement as the fine-tuning notebook — 39,484
chunks through BGE-m3 is not a CPU-friendly job). Copies the fine-tuned checkpoint
into a stable in-repo path (`models/bge_m3_finetuned_v6/`) and builds
`models/index_bge_m3_finetuned_v6/` (`embeddings.npy`, `chunk_ids.json`,
`manifest.json`) in the exact schema `src/dhara/service.py`'s `Dhara.load()` reads.

The document template below is copied verbatim from
`notebooks/ssh-retrieval-finetune.ipynb`'s `document_text()` -- the model was
fine-tuned against that exact format (Act/Section-prefixed), which is a
different, richer template than `BiEncoderRetriever._document()`'s default
(title + text_bn, no Act/Section prefix, used for the zero-shot control index).
Building this index with the wrong template would quietly underperform what the
notebook's own eval measured, since the model's learned representation is tuned
to what it saw during training. Keep this template identical to the notebook's
if the notebook's `document_text()` ever changes.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import torch

WORKDIR = pathlib.Path(__file__).resolve().parents[1]
CORPUS = WORKDIR / "data" / "processed" / "corpus_v1.jsonl"

# Where the merged fine-tuned checkpoint was delivered from the SSH training run.
# Adjust if your copy landed somewhere else.
SOURCE_CHECKPOINT = WORKDIR / "dhara_outputs_results" / "outputs" / "bge_m3_retrieval_human_aware_v3_merged"
CHECKPOINT_DEST = WORKDIR / "models" / "bge_m3_finetuned_v6"
INDEX_OUT = WORKDIR / "models" / "index_bge_m3_finetuned_v6"

MAX_SEQ_LENGTH = 384  # matches EVAL_MAX_LENGTH in the training notebook
BATCH_SIZE = 64


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def document_text(row: dict) -> str:
    # Verbatim copy of the notebook's document_text(); keep identical to it.
    title = row.get("act_title_bn") or row.get("act_title_en") or ""
    section = row.get("provision_no_ascii") or row.get("provision_no_bn") or ""
    return f"Act: {title} | Section: {section} | {row.get('text_raw') or row.get('text_bn') or ''}"


def main() -> None:
    assert SOURCE_CHECKPOINT.exists(), (
        f"Fine-tuned checkpoint not found at {SOURCE_CHECKPOINT}. "
        "Copy dhara_outputs_results/outputs/bge_m3_retrieval_human_aware_v3_merged "
        "into the repo first, or edit SOURCE_CHECKPOINT above."
    )
    if not CHECKPOINT_DEST.exists():
        shutil.copytree(SOURCE_CHECKPOINT, CHECKPOINT_DEST)
        print("copied checkpoint to", CHECKPOINT_DEST)

    corpus = read_jsonl(CORPUS)
    print(f"encoding {len(corpus)} chunks with {CHECKPOINT_DEST}...")

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(CHECKPOINT_DEST), device=device,
                                 model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {})
    model.max_seq_length = MAX_SEQ_LENGTH

    docs = [document_text(r) for r in corpus]
    embeddings = model.encode(docs, batch_size=BATCH_SIZE, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    chunk_ids = [r["chunk_id"] for r in corpus]

    INDEX_OUT.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_OUT / "embeddings.npy", embeddings)
    (INDEX_OUT / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    manifest = {
        "checkpoint": str(CHECKPOINT_DEST.relative_to(WORKDIR)).replace("\\", "/"),
        "max_seq_length": MAX_SEQ_LENGTH,
        "use_title": True,
        "query_prefix": "",
        "passage_prefix": "",
        "dim": int(embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "n_chunks": len(corpus),
        "corpus_file": "data/processed/corpus_v1.jsonl",
        "fine_tuned": True,
        "built_by": "scripts/76_build_final_demo_index.py",
        "document_template": 'f"Act: {act_title_bn or act_title_en} | Section: {provision_no_ascii or provision_no_bn} | {text_raw or text_bn}"',
        "notes": (
            "v6 pool (2,685 approved + 942 authored + 408 real human rows), LoRA fine-tune "
            "merged. Test R@10 0.500 vs zero-shot 0.472 (+2.8pt); English-target test R@10 "
            "0.417 vs zero-shot 0.333 (+8.3pt). See docs/PROJECT_REPORT.md section 5."
        ),
    }
    (INDEX_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", INDEX_OUT)
    print("Now point src/dhara/service.py's DEFAULT_INDEX at this directory (already done if you pulled the latest code).")


if __name__ == "__main__":
    main()
