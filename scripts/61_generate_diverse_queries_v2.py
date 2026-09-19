"""Generate a v2 augmentation shard from the v2 prompt ledger.

Local GPU (Qwen or any HF causal LM):

    python scripts/61_generate_diverse_queries_v2.py \
      --input data/processed/diverse_augmentation_prompt_ledger_v2.jsonl \
      --output data/processed/augmentation_raw_v2_part000.jsonl --start 0 --limit 500

Any OpenAI-compatible endpoint (vLLM, TGI, a hosted API):

    DHARA_LLM_API_KEY=... python scripts/61_generate_diverse_queries_v2.py \
      --backend openai --base-url https://host/v1 --model <name> \
      --input data/processed/diverse_augmentation_prompt_ledger_v2.jsonl \
      --output data/processed/augmentation_raw_v2_part000.jsonl --limit 500

This writes raw candidates only; script 62 must validate them before they can
become training data.

Three changes from script 56, each aimed at a measured v1 failure:

*   **Constrained decoding.** The v1 shard copied statute wording in 345 of 457
    parseable rows.  On the `hf` backend the ledger's banned words are compiled
    into `bad_words_ids`, so the decoder cannot emit them at all.  The ban list
    uses the same `aggressive()` surface forms the overlap metric counts, so the
    constraint and the metric agree token for token.  This prevents copying
    rather than concealing it, but it is still an intervention: record it in
    DECISIONS.md and keep the human audit sample in script 63 mandatory, because
    a question denied its subject vocabulary can come out vague instead of
    merely non-copying.  The post-hoc gates in script 62 are unchanged and stay
    the arbiter, and script 63 writes the audit sample.
*   **Enforced length.** v1 asked for a word count in prose and got ~25 words
    regardless of whether it asked for 18 or 120.  `min_new_tokens` now makes
    stopping early impossible.
*   **Plain-text output.** v1 wrapped answers in JSON and 147 of 500 outputs
    never closed the brace.  The question is now the whole completion.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dhara.normalize import BN_DIGITS  # noqa: E402

ASCII_TO_BN_DIGIT = {str(index): digit for index, digit in enumerate(BN_DIGITS)}

SYSTEM = (
    "You write Bangla questions in the voice of ordinary citizens with no legal training, "
    "for a legal-search training set. You obey the banned-word list absolutely and you write "
    "the full requested length. You reply with the Bangla question only."
)
# Qwen tokenizes Bangla poorly, so a Bangla word costs far more than one token.
# Measure on your own checkpoint and override with --tokens-per-word if it differs.
DEFAULT_TOKENS_PER_WORD = 3.4


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_prompt_ids(path: pathlib.Path) -> set[str]:
    """Prompt ids already written, so an interrupted shard can be resumed."""
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(json.loads(line)["prompt_id"])
    return seen


def clean_question(text: str) -> str | None:
    """Strip the wrappers a chat model adds around a bare answer."""
    text = text.strip()
    # Some models still emit JSON despite the instruction; take the field.
    match = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match:
        try:
            text = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            text = match.group(1)
    text = re.sub(r"^(?:here is|question|প্রশ্ন|উত্তর)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip()
    # Drop a trailing explanation the model appended after a blank line.
    text = text.split("\n\n")[0].strip()
    text = " ".join(text.split())
    return text or None


def length_budget(target_words: int, tokens_per_word: float) -> tuple[int, int]:
    minimum = max(16, int(target_words * 0.75 * tokens_per_word))
    maximum = max(minimum + 48, int(target_words * 1.7 * tokens_per_word) + 64)
    return minimum, maximum


def surface_variants(word: str) -> list[str]:
    """Every spelling of a banned word that the overlap metric would collapse.

    The ledger's banned words come from `aggressive()`, which maps Bangla digits
    to ASCII.  So a ban on "2008" must also ban "২০০৮": the model can emit the
    Bangla form, and the metric would normalize it straight back into a match.
    """
    bangla = "".join(ASCII_TO_BN_DIGIT.get(char, char) for char in word)
    return [word] if bangla == word else [word, bangla]


def ban_token_ids(tokenizer, words: list[str], limit: int) -> list[list[int]]:
    """Token sequences the decoder must never emit.

    Each banned word is compiled both bare and with a leading space, because a
    mid-sentence word carries the space into its first token.
    """
    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for word in words[:limit]:
        for variant in surface_variants(word):
            for surface in (variant, f" {variant}"):
                ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
                if ids and tuple(ids) not in seen:
                    seen.add(tuple(ids))
                    sequences.append(ids)
    return sequences


def run_hf(rows: list[dict[str, Any]], args, handle) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("the hf backend requires a CUDA GPU; use --backend openai otherwise")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    ) if args.load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=compute_dtype,
        quantization_config=quantization,
    )

    written = 0
    # bad_words_ids and min_new_tokens are per-prompt, so a batch has to share a
    # ban list and a length. Group by target words and keep batches small.
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        for row in batch:
            banned = row["banned_title_content_words"] + row["banned_body_content_words"]
            bad_words = ban_token_ids(tokenizer, banned, args.max_banned_words) if args.constrain else None
            minimum, maximum = length_budget(int(row["target_words"]), args.tokens_per_word)
            chat = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    min_new_tokens=minimum if args.force_length else None,
                    max_new_tokens=maximum,
                    bad_words_ids=bad_words,
                    do_sample=True, temperature=args.temperature, top_p=0.92,
                    repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(generated[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
            handle.write(json.dumps({
                **row, "raw_output": raw, "question": clean_question(raw),
                "generation_model": args.model, "generation_backend": "hf",
                "constrained_decoding": bool(bad_words),
                "banned_sequences": len(bad_words or []),
                "min_new_tokens": minimum if args.force_length else 0,
            }, ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
        print({"written": written, "total": len(rows)}, flush=True)
    return written


def run_openai(rows: list[dict[str, Any]], args, handle) -> int:
    """Any OpenAI-compatible chat endpoint.

    No constrained decoding here - the API exposes no token ban list - so these
    shards lean entirely on the script 62 gates and will have a lower yield
    unless the model paraphrases Bangla well on its own.
    """
    from openai import OpenAI

    key = os.environ.get("DHARA_LLM_API_KEY")
    if not key:
        raise SystemExit("set DHARA_LLM_API_KEY in the environment; never put a key in a source file")
    client = OpenAI(api_key=key, base_url=args.base_url)

    written = 0
    for row in rows:
        banned = row["banned_title_content_words"] + row["banned_body_content_words"]
        minimum, maximum = length_budget(int(row["target_words"]), args.tokens_per_word)
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                ],
                temperature=args.temperature, max_tokens=maximum,
            )
            raw = response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 - one bad row must not kill a shard
            raw = ""
            print({"prompt_id": row["prompt_id"], "error": str(error)[:200]}, flush=True)
        handle.write(json.dumps({
            **row, "raw_output": raw, "question": clean_question(raw),
            "generation_model": args.model, "generation_backend": "openai",
            "constrained_decoding": False, "banned_sequences": len(banned),
            "min_new_tokens": 0,
        }, ensure_ascii=False) + "\n")
        written += 1
        if written % 20 == 0:
            handle.flush()
            print({"written": written, "total": len(rows)}, flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--backend", choices=("hf", "openai"), default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url", default=None, help="openai backend only")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--tokens-per-word", type=float, default=DEFAULT_TOKENS_PER_WORD)
    parser.add_argument("--max-banned-words", type=int, default=80)
    parser.add_argument("--no-constrain", dest="constrain", action="store_false",
                        help="hf backend: skip bad_words_ids and rely on the prompt alone")
    parser.add_argument("--no-force-length", dest="force_length", action="store_false",
                        help="skip min_new_tokens; v1 showed the prompt alone does not control length")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing shard, skipping prompt_ids already written")
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0:
        parser.error("--start must be non-negative and --limit positive")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite raw generation shard: {args.output} (pass --resume to continue it)")

    ledger = read_jsonl(args.input)[args.start:args.start + args.limit]
    already = done_prompt_ids(args.output) if args.resume else set()
    rows = [row for row in ledger if row["prompt_id"] not in already]
    if not rows:
        raise SystemExit("nothing left to generate in this shard")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        runner = run_hf if args.backend == "hf" else run_openai
        written = runner(rows, args, handle)

    print(json.dumps({
        "output": str(args.output), "requested": len(ledger),
        "skipped_already_done": len(already), "written": written,
        "next_step": f"python scripts/62_validate_diverse_queries_v2.py --raw {args.output} --output <validated>",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
