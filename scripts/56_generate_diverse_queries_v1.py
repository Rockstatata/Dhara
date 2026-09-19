"""Generate a private, batched LLM augmentation shard on the SSH GPU.

Example:
    python scripts/56_generate_diverse_queries_v1.py \
      --input data/processed/diverse_augmentation_prompt_ledger_v1.jsonl \
      --output data/processed/augmentation_raw_v1_part000.jsonl --start 0 --limit 500

Run multiple non-overlapping shards. This writes raw candidates only; script 57
must validate them before they can be part of the training set.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_answer(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("question"), str):
            return value
    # Qwen sometimes finishes the question but truncates the optional facts
    # list at the generation limit.  Recover only the complete JSON string
    # value; script 57 performs the substantive quality gates.
    marker = re.search(r'"question"\s*:\s*"', text)
    if marker:
        start = marker.end() - 1
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if char == '"' and not escaped:
                try:
                    question = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(question, str) and question.strip():
                    return {"question": question, "facts_preserved": []}
            escaped = (char == '\\' and not escaped)
            if char != '\\':
                escaped = False
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite raw generation shard: {args.output}")
    if args.start < 0 or args.limit <= 0:
        parser.error("--start must be non-negative and --limit positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("this generator requires the SSH CUDA GPU")
    ledger = read_jsonl(args.input)[args.start:args.start + args.limit]
    if not ledger:
        raise SystemExit("requested shard is empty")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=compute_dtype,
        quantization_config=quantization,
    )
    system = (
        "You create careful Bengali citizen questions for a legal retrieval training set. "
        "Follow the user's banned-word and JSON-only requirements exactly. "
        "Use the full requested length; do not stop after a short question. "
        "Return facts_preserved as an empty list and finish the JSON object."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted_json = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for offset in range(0, len(ledger), args.batch_size):
            batch = ledger[offset:offset + args.batch_size]
            chats = []
            for row in batch:
                target = int(row["target_words"])
                lower, upper = max(8, round(target * 0.75)), round(target * 1.25)
                length_instruction = (
                    f"\nMANDATORY LENGTH: write {lower}-{upper} whitespace-separated Bangla words "
                    f"(target {target}); use the full scenario before ending. "
                    "Return facts_preserved as [] exactly."
                )
                chats.append([
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"] + length_instruction},
                ])
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
            tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2300).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.75, top_p=0.9, repetition_penalty=1.08,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for row, prompt_tokens, answer_tokens in zip(batch, tokens["input_ids"], generated):
                raw = tokenizer.decode(answer_tokens[len(prompt_tokens):], skip_special_tokens=True).strip()
                parsed = json_answer(raw)
                if parsed is not None:
                    accepted_json += 1
                record = {
                    **row, "raw_output": raw, "parsed": parsed,
                    "generation_model": args.model,
                    "generation_index": args.start + offset,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print({"done": min(offset + len(batch), len(ledger)), "total": len(ledger), "json": accepted_json}, flush=True)
    print({"output": str(args.output), "rows": len(ledger), "parseable_json": accepted_json})


if __name__ == "__main__":
    main()
