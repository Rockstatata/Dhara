"""Topic 2 — weak-label + train the BiRNN sequence tagger (proposal, lowest priority).

    python scripts/43_train_tagger.py

docs/PIPELINE.md's own build order lists this last and names it explicitly
cuttable if time runs short; it survives here because the weak-labeling is
cheap: a section-number regex plus data/lexicon/register_map.json's formal
legal-term list, exactly the doc's own spec ("labels bootstrap by matching
corpus terms into questions, then get hand-corrected"). Hand-correction is
skipped given the timeline -- reported as a limitation below, not hidden.

Only train-split qids are used (never the 200 frozen test qids, for any
purpose, in this repo), further split internally into a weak-label train/dev
slice since there is no human-verified tag data to evaluate against. Dev
accuracy here measures self-consistency with the weak-labeling rules, not
tagging quality against ground truth -- state that plainly in the report.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.models.tagger import TAGS, TAG_TO_ID, BiRNNSequenceLabeler, save  # noqa: E402
from dhara.normalize import aggressive  # noqa: E402
from dhara.vocab import Vocab  # noqa: E402

GOLD = pathlib.Path("data/processed/gold_verified_v2.jsonl")
SPLIT = pathlib.Path("data/processed/test_split_v1.json")
LEXICON = pathlib.Path("data/lexicon/register_map.json")
VOCAB = pathlib.Path("data/processed/vocab.json")
MATRIX = pathlib.Path("models/embedding_matrix.pt")
OUT = pathlib.Path("models/tagger.pt")
RESULTS = pathlib.Path("results/runs/tagger_v3.json")

SECTION_MARKERS = {"ধারা", "অনুচ্ছেদ", "section"}
PARTY_WORDS = {
    "স্বামী", "স্ত্রী", "বাদী", "বিবাদী", "মালিক", "শ্রমিক", "ওয়ারিশ", "নমিনি",
    "ভাড়াটিয়া", "বাড়িওয়ালা", "পিতা", "মাতা", "বাবা", "মা", "ছেলে", "মেয়ে",
    "সন্তান", "নানা", "নানি", "দাদা", "দাদি", "চাচা", "মামা", "ভাই", "বোন",
    "আসামি", "সৎমা", "দৌহিত্র",
}
SECTION_NO_RE = re.compile(r"^\d+[a-z]?$")
EPOCHS = 6
BATCH_SIZE = 16
SEED = 42


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_legal_term_phrases(lexicon_path: pathlib.Path) -> list[tuple[str, ...]]:
    data = json.loads(lexicon_path.read_text(encoding="utf-8"))
    phrases = []
    for term in data.get("bn", {}):
        toks = tuple(aggressive(term).split())
        if toks:
            phrases.append(toks)
    phrases.sort(key=len, reverse=True)  # longest-match first
    return phrases


def weak_label(tokens: list[str], legal_phrases: list[tuple[str, ...]]) -> list[str]:
    n = len(tokens)
    tags = ["O"] * n

    # SECTION_NO: a digit token within 2 positions of a section marker.
    for i, tok in enumerate(tokens):
        if SECTION_NO_RE.match(tok):
            window = tokens[max(0, i - 2): i] + tokens[i + 1: i + 3]
            if any(w in SECTION_MARKERS for w in window):
                tags[i] = "SECTION_NO"

    # ACT: tokens ending in the Bangla suffix for "law/act", or "সংবিধান".
    for i, tok in enumerate(tokens):
        if tags[i] != "O":
            continue
        if tok == "সংবিধান" or tok.endswith("আইন") or tok.endswith("আইনে") or tok.endswith("আইনের"):
            tags[i] = "ACT"

    # LEGAL_TERM: greedy longest-match against the register-map phrase list.
    i = 0
    while i < n:
        if tags[i] != "O":
            i += 1
            continue
        matched = False
        for phrase in legal_phrases:
            L = len(phrase)
            if tokens[i:i + L] == list(phrase) and all(tags[i + k] == "O" for k in range(L)):
                for k in range(L):
                    tags[i + k] = "LEGAL_TERM"
                i += L
                matched = True
                break
        if not matched:
            i += 1

    # PARTY: exact token match against the role-noun list.
    for i, tok in enumerate(tokens):
        if tags[i] == "O" and tok in PARTY_WORDS:
            tags[i] = "PARTY"

    return tags


def encode(vocab: Vocab, tokens: list[str], tags: list[str]) -> tuple[list[int], list[int]]:
    ids = [vocab.stoi.get(t, vocab.unk_id) for t in tokens]
    tag_ids = [TAG_TO_ID[t] for t in tags]
    return ids, tag_ids


def pad(id_lists: list[list[int]], pad_value: int = 0) -> torch.Tensor:
    maxlen = max(len(x) for x in id_lists)
    batch = torch.full((len(id_lists), maxlen), pad_value, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def main() -> None:
    gold = read_jsonl(GOLD)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_qids = set(split["train_qids"])  # 200 frozen test qids are never touched here
    rows = [r for r in gold if r["qid"] in train_qids]

    legal_phrases = build_legal_term_phrases(LEXICON)
    print(f"{len(rows)} train-split questions, {len(legal_phrases)} legal-term phrases")

    examples = []
    for r in rows:
        tokens = aggressive(r["question_bn"]).split()
        if not tokens:
            continue
        tags = weak_label(tokens, legal_phrases)
        examples.append((r["qid"], tokens, tags))

    tag_counts = {t: 0 for t in TAGS}
    for _, _, tags in examples:
        for t in tags:
            tag_counts[t] += 1
    print(f"weak-label tag distribution: {tag_counts}")

    rng = random.Random(SEED)
    rng.shuffle(examples)
    n_dev = max(1, len(examples) // 6)
    dev, train = examples[:n_dev], examples[n_dev:]
    print(f"internal weak-label split: train {len(train)}, dev {len(dev)}")

    vocab = Vocab.load(VOCAB)
    embedding_matrix = torch.load(MATRIX, map_location="cpu", weights_only=False)
    model = BiRNNSequenceLabeler(embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start: start + BATCH_SIZE]
            batch = [train[i] for i in idx]
            id_lists, tag_lists = [], []
            for _, tokens, tags in batch:
                ids, tag_ids = encode(vocab, tokens, tags)
                id_lists.append(ids)
                tag_lists.append(tag_ids)
            ids_t = pad(id_lists, pad_value=0)
            tags_t = pad(tag_lists, pad_value=-100)  # ignore_index for padding

            logits = model(ids_t)
            loss = F.cross_entropy(logits.view(-1, len(TAGS)), tags_t.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch}: mean loss {mean_loss:.4f}")

    # Dev = self-consistency with the weak-labeling rules, not ground truth.
    # Token accuracy alone is a poor metric here regardless of that caveat:
    # 'O' dominates the tag distribution, so a model that predicts 'O' for
    # everything already scores high "accuracy" while tagging nothing. Report
    # precision/recall/F1 per entity tag (ACT/SECTION_NO/LEGAL_TERM/PARTY),
    # excluding 'O', plus their macro-F1 -- the standard way to report a
    # tagger, and the only way a reader can tell "tags well" apart from
    # "mostly says O".
    model.eval()
    entity_tags = [t for t in TAGS if t != "O"]
    tp = {t: 0 for t in entity_tags}
    fp = {t: 0 for t in entity_tags}
    fn = {t: 0 for t in entity_tags}
    correct, total = 0, 0
    with torch.no_grad():
        for _, tokens, tags in dev:
            ids, tag_ids = encode(vocab, tokens, tags)
            logits = model(torch.tensor([ids], dtype=torch.long))
            pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            for p, g in zip(pred_ids, tag_ids):
                total += 1
                if p == g:
                    correct += 1
                p_tag, g_tag = TAGS[p], TAGS[g]
                if p_tag == g_tag and p_tag != "O":
                    tp[p_tag] += 1
                else:
                    if p_tag != "O":
                        fp[p_tag] += 1
                    if g_tag != "O":
                        fn[g_tag] += 1
    token_acc = correct / max(total, 1)

    def prf(tag: str) -> tuple[float, float, float]:
        p_denom = tp[tag] + fp[tag]
        r_denom = tp[tag] + fn[tag]
        precision = tp[tag] / p_denom if p_denom else 0.0
        recall = tp[tag] / r_denom if r_denom else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    per_tag_prf = {t: dict(zip(("precision", "recall", "f1"), prf(t))) for t in entity_tags}
    entity_macro_f1 = round(sum(v["f1"] for v in per_tag_prf.values()) / len(entity_tags), 4)

    print(f"\ndev token accuracy (self-consistency, NOT ground truth, and O-dominated): {token_acc:.4f}")
    print(f"entity macro-F1 (excludes O, self-consistency only): {entity_macro_f1:.4f}")
    for t, v in per_tag_prf.items():
        print(f"  {t:12s} P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}")

    config = {"hidden": 128, "vocab_path": str(VOCAB), "embedding_matrix_path": str(MATRIX), "tags": TAGS}
    save(model, OUT, config)
    print(f"wrote {OUT}")

    results = {
        "run_id": "tagger_v3",
        "tags": TAGS,
        "n_train_questions": len(rows),
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "weak_label_tag_distribution": tag_counts,
        "dev_token_accuracy_self_consistency": round(token_acc, 4),
        "dev_entity_macro_f1_self_consistency": entity_macro_f1,
        "dev_per_tag_prf_self_consistency": per_tag_prf,
        "limitation": "Labels are weak (regex + data/lexicon/register_map.json gazetteer "
                      "matching against corpus terms), never hand-corrected, per "
                      "docs/PIPELINE.md's own explicit lowest-priority ranking for this "
                      "topic. All metrics here measure agreement with the weak-labeling "
                      "rules, not tagging quality against human-verified ground truth. "
                      "v3 vs v2: token accuracy alone was misleading (O-dominated -- a "
                      "model predicting O everywhere still scores high); replaced with "
                      "per-entity-tag precision/recall/F1 and their macro-F1, the standard "
                      "way to report a tagger. This changes what is measured, not the "
                      "self-consistency caveat, which remains and is not resolved by a "
                      "better metric.",
        "history": history,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
