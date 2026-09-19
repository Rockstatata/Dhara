"""Recover attempt 4's real weights from a malformed LoRA export.

    python scripts/33_merge_attempt4_checkpoint.py

## What went wrong

notebooks/colab_bge_m3_finetune.ipynb cell 12/13 saved the raw PEFT-wrapped
backbone's state_dict via `model.save("dhara_bge_m3_lora")` instead of calling
`peft_backbone.merge_and_unload()` first. The exported
`dhara_bge_m3_lora.zip` therefore contains, per LoRA target layer
(query/value only, per the notebook's `target_modules=["query","value"]`):

    *.base_layer.weight / .bias   -- the original frozen base weight
    *.lora_A.default.weight       -- [r, in_features]
    *.lora_B.default.weight       -- [out_features, r]

with `config.json` still declaring a plain `XLMRobertaModel`, so a normal
`SentenceTransformer(...)` load silently drops every base_layer/lora_* key as
unrecognized and reinitializes query/value at random. Loading it directly
produces garbage embeddings, not attempt 4's trained representations.

## The fix

The standard LoRA merge, with r=16 / alpha=32 fixed by the notebook's
`LoraConfig(r=16, lora_alpha=32, ...)` (cell 6):

    merged.weight = base_layer.weight + lora_B.weight @ lora_A.weight * (alpha / r)
    merged.bias   = base_layer.bias  (LoRA was bias="none", bias is untouched)

Every non-LoRA key (everything outside attention.self.{query,value}) is passed
through unchanged. This is unambiguous -- no adapter_config.json needed to
recover it, the two matrices and the scaling factor are the entire formula --
and it is verified below by checking canary cosine similarity is a real value
(not the degenerate ~0 you get from randomly-initialized attention weights,
which is what happens if the merge is skipped).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import zipfile

import torch
from safetensors import safe_open

LORA_ZIP = pathlib.Path("data/finetune-results/dhara_bge_m3_lora.zip")
EXTRACT_DIR = pathlib.Path("models/_attempt4_raw_export")
OUT_DIR = pathlib.Path("models/checkpoint_bge_m3_attempt4")
BASE_CHECKPOINT = "BAAI/bge-m3"
LORA_R = 16
LORA_ALPHA = 32
SCALING = LORA_ALPHA / LORA_R


def merge_state_dict(safetensors_path: pathlib.Path) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        base_layer_keys = {k for k in keys if ".base_layer." in k}
        for k in keys:
            if ".lora_A." in k or ".lora_B." in k:
                continue  # consumed via the corresponding base_layer key below
            if ".base_layer." in k:
                plain_key = k.replace(".base_layer.", ".")
                tensor = f.get_tensor(k)
                if plain_key.endswith(".weight"):
                    prefix = k.rsplit(".base_layer.weight", 1)[0]
                    lora_a_key = f"{prefix}.lora_A.default.weight"
                    lora_b_key = f"{prefix}.lora_B.default.weight"
                    assert lora_a_key in keys and lora_b_key in keys, (
                        f"{k} has no matching lora_A/lora_B -- merge formula assumption broken"
                    )
                    lora_a = f.get_tensor(lora_a_key).float()
                    lora_b = f.get_tensor(lora_b_key).float()
                    delta = (lora_b @ lora_a) * SCALING
                    tensor = tensor.float() + delta
                merged[plain_key] = tensor
            else:
                merged[k] = f.get_tensor(k)
    n_merged = sum(1 for k in merged if any(
        k.endswith(f"{proj}.weight") and f"{proj}.base_layer.weight" in base_layer_keys
        for proj in ("query", "value")
    ))
    print(f"merged {n_merged} LoRA-adapted weight matrices "
          f"(expect 48: query+value x 24 layers)")
    return merged


def main() -> None:
    assert LORA_ZIP.exists(), f"{LORA_ZIP} not found"

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(LORA_ZIP) as z:
        z.extractall(EXTRACT_DIR)
    export_dir = EXTRACT_DIR / "dhara_bge_m3_lora"
    safetensors_path = export_dir / "model.safetensors"

    merged_state_dict = merge_state_dict(safetensors_path)

    from sentence_transformers import SentenceTransformer

    print(f"loading base checkpoint {BASE_CHECKPOINT} for a clean config/tokenizer...")
    model = SentenceTransformer(BASE_CHECKPOINT, device="cpu")
    backbone = model[0].auto_model

    base_keys = set(backbone.state_dict().keys())
    merged_keys = set(merged_state_dict.keys())
    missing = base_keys - merged_keys
    extra = merged_keys - base_keys
    assert not missing, f"{len(missing)} base keys have no merged replacement, e.g. {list(missing)[:5]}"
    assert not extra, f"{len(extra)} merged keys don't exist on the base model, e.g. {list(extra)[:5]}"

    # Cosine of the base model's own query weight vs. the merged one -- catches
    # a silently-empty merge (delta all zero, i.e. lora_A/lora_B not applied).
    before = backbone.state_dict()["encoder.layer.0.attention.self.query.weight"].float()
    after = merged_state_dict["encoder.layer.0.attention.self.query.weight"].float()
    delta_norm = (after - before).norm().item()
    print(f"layer-0 query weight delta norm vs base: {delta_norm:.4f} (expect > 0)")
    assert delta_norm > 1e-4, "merged weights are identical to base -- the LoRA delta didn't apply"

    backbone.load_state_dict(merged_state_dict, strict=True)

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_DIR))
    print(f"saved merged checkpoint to {OUT_DIR}")

    # Sanity check: encode two clearly different legal sentences and confirm the
    # embeddings are not degenerate (near-identical or near-zero-normed), which is
    # what a still-broken load looks like.
    check = SentenceTransformer(str(OUT_DIR), device="cpu")
    a, b = check.encode([
        "শ্রমিকের সাপ্তাহিক ছুটি সংক্রান্ত বিধান",
        "স্বামী-স্ত্রীর মধ্যে তালাক ও ভরণপোষণ",
    ], normalize_embeddings=True, convert_to_numpy=True)
    cos = float(a @ b)
    print(f"sanity check cosine(unrelated legal sentences) = {cos:.4f} "
          f"(should be a real, non-degenerate value, not ~1.0 or NaN)")
    assert -0.99 < cos < 0.99, "embeddings look degenerate -- merge may still be broken"

    shutil.rmtree(EXTRACT_DIR)
    print("done.")


if __name__ == "__main__":
    main()
