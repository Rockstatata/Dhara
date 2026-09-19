"""Create a checksummed private-transfer manifest for SSH retrieval training.

    python scripts/54_build_ssh_training_manifest_v1.py

Only files named in this manifest belong in the current SSH experiment. Legacy
v2 split files and raw annotation sheets are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
import pathlib


FILES = {
    "corpus": pathlib.Path("data/processed/corpus_v1.jsonl"),
    "supervised_train": pathlib.Path("data/processed/train_retrieval_v4.jsonl"),
    "development": pathlib.Path("data/processed/dev_retrieval_v3.jsonl"),
    "frozen_test": pathlib.Path("data/processed/test_retrieval_v3.jsonl"),
    "optional_coverage_pretrain": pathlib.Path("data/processed/pretrain_retrieval_coverage_v1.jsonl"),
    "diverse_augmentation_prompt_ledger_not_training": pathlib.Path("data/processed/diverse_augmentation_prompt_ledger_v1.jsonl"),
    "ssh_notebook": pathlib.Path("notebooks/ssh-retrieval-finetune"),
}
OUT = pathlib.Path("results/runs/ssh_retrieval_training_manifest_v2.json")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"immutable manifest already exists: {OUT}")
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise SystemExit("missing required SSH files: " + ", ".join(missing))
    manifest = {
        "version": "v2",
        "purpose": "private SSH retrieval experiment transfer",
        "files": {
            role: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for role, path in FILES.items()
        },
        "training_contract": {
            "default_supervision": "v4 now; v5 automatically when validated LLM augmentation is promoted",
            "coverage": "optional ablation only; do not use as final fine-tuning supervision",
            "prompt_ledger": "generate with script 56, validate with 57, then promote with 58 before it becomes v5 training data",
            "selection_metric": "provision-level development R@10",
            "report": "paired provision-level test R@1/5/10/100, P@k lower bounds, MRR@10, nDCG@10",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
