# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Greenfield.** The repository currently contains only `docs/` and no commits. The two documents are the authoritative spec:

- [docs/Dhara_Proposal.md](docs/Dhara_Proposal.md) — the *why*: problem framing, scope, evaluation plan, ethics.
- [docs/Dhara_Implementation_Guide.md](docs/Dhara_Implementation_Guide.md) — the *how*: repo layout, schemas, per-phase owners, model configs, week-by-week gates.

When building any part of this project, read the corresponding section of the implementation guide first — it specifies file paths, schemas, and hyperparameters that other people's code is written against. Section references below use the guide's `§` numbering.

## What Dhara is

A Bangla legal retrieval system: a plain-Bangla citizen question in, the exact law **section (ধারা)** plus its citation out. The research claim is that a *lexical gap* separates colloquial citizen phrasing from formal legal Bangla, and that domain fine-tuning closes it. Everything in the design exists to measure that claim, so evaluation integrity outranks model performance in every tradeoff.

Corpus source: the bdlaws portal (structured HTML, no OCR). Four domains — family, land, labour, consumer — plus the Constitution as a fifth cross-cutting source. Target 800–1,200 section-level chunks.

## Architecture

Query → normalize → **BM25 (sparse)** and **fine-tuned bi-encoder (dense)** in parallel → RRF fusion → top-50 → cross-encoder rerank → top-3 sections with citations. An intent classifier runs alongside, reported separately; intent-based candidate filtering stays an ablation, never the default path (a wrong intent prediction silently destroys retrieval).

The five retrieval configurations form an "experimental ladder" (§7), and each rung is a separate report result, not a discarded prototype:

1. BM25 — tune `k1`/`b` on dev; an untuned baseline invalidates the comparison
2. Self-trained Word2Vec (gensim skip-gram, `vector_size=200, min_count=3, sg=1, epochs=30`)
3. BiLSTM dual-encoder over frozen W2V, InfoNCE with in-batch negatives
4. a) zero-shot multilingual dense **control**, b) fine-tuned bi-encoder (`MultipleNegativesRankingLoss`, max_len 256, lr 2e-5, 2–4 epochs, fp16)
5. Cross-encoder reranker over the fused top-50

Rung 4a must be locked *before* 4b runs — it is what separates "transformers work" from "our fine-tuning works."

## Planned layout (§2.1)

```
configs/     domains.yaml, models.yaml, paths.yaml, split_overrides.yaml
data/raw/    scraped HTML — never edited, never committed
data/processed/  frozen artifacts, JSONL, DO commit these
src/dhara/   normalize.py schema.py scrape.py section_split.py synth.py
             negatives.py metrics.py evaluate.py rerank.py classify.py
             cluster.py service.py retrievers/{base,bm25,word2vec,bilstm,biencoder,hybrid}.py
src/app/     app.py — Gradio UI
scripts/     01_survey_acts … 09_run_eval — thin CLIs, one job each
results/     tables/ figures/ runs/ (one JSON per experiment run)
```

`.gitignore` must cover `data/raw/`, `data/external/`, `*.bin`, `*.safetensors`, checkpoints. `data/processed/*.jsonl` **is** committed — it is small and it is what makes results reproducible.

## Commands

Nothing is implemented yet; these are the contracts the scripts must satisfy.

```bash
python scripts/01_survey_acts.py                                    # feasibility gate → results/tables/act_survey.csv
python scripts/02_scrape.py --acts 3 --out data/raw/
python scripts/03_build_corpus.py --in data/raw/ --out data/processed/corpus_v1.jsonl
python scripts/03_build_corpus.py --qa                              # → results/tables/corpus_stats.csv
python scripts/04_generate_questions.py
python scripts/05_mine_negatives.py
python scripts/06_train_biencoder.py
python scripts/07_train_crossencoder.py
python scripts/08_build_index.py                                    # → models/index_v1/
python scripts/09_run_eval.py --corpus corpus_v1.jsonl --gold gold_test_v1.jsonl --retriever bm25
python -m src.app.app                                               # Gradio demo
```

Install: `pip install -r requirements.txt`, then separately (not on PyPI, required if using BanglaBERT):

```bash
pip install git+https://github.com/csebuetnlp/normalizer
```

Compute target is a Colab/Kaggle T4; nothing here needs more.

## Non-negotiable rules

These come from the guide and are the ones most likely to be violated by an agent moving fast.

**Gold set isolation.** `gold_test_v1.jsonl` is touched exactly once, at the end. Never for debugging, never for "just checking." `scripts/09_run_eval.py` calls `assert_no_leakage(gold, train, dev)` on every run and must fail loudly.

**Split by chunk, not by question.** All synthetic questions generated from chunk X go to the same split, or near-duplicates leak across train/dev.

**Freeze then version.** `corpus_v1.jsonl` never changes after Week 3. A needed fix becomes `corpus_v2.jsonl` and every affected result is re-run or explicitly labelled.

**Two normalization levels, and they are not interchangeable** (§4.5). `normalize.light()` — NFC, ZWNJ strip, whitespace — feeds transformers. `normalize.aggressive()` — plus ZWJ strip, Bangla→ASCII digits, punctuation padding, lowercasing — feeds BM25/Word2Vec/TF-IDF. Applying aggressive normalization to transformer input is a real, measurable performance loss. Digit normalization (১২৩ ↔ 123) is load-bearing here because section numbers appear in both forms.

**Two text fields, deliberately** (§4.4). The UI shows `text_raw` — the law as printed. Models consume `text_bn`. Never display normalized text to a Bangla reader.

**E5 prefixes.** If the checkpoint is from the E5 family, `"query: "` and `"passage: "` prefixes must be applied at index time and query time, in both the zero-shot and fine-tuned runs. Omitting them silently cripples the baseline and invalidates the comparison.

**Hard negatives come from ranks 5–30**, not 1–4. The top few are often relevant-but-unlabelled; training against them teaches the model that correct answers are wrong.

**No hand-typed numbers.** Every number in the report comes from a `results/runs/*.json` emitted by a script. Notebooks are for looking at things only. Every run JSON includes `per_query` — error analysis and significance testing need it and regenerating it later means re-running everything.

**Every output carries a citation, and the system must be able to abstain.** A score threshold, calibrated on the deliberately-included unanswerable gold questions, below which it says it has no confident match. Prefer false abstentions over false confidence.

## Code contracts (§12.2)

Three interfaces are frozen in Week 1 and everything else builds against them:

- `Retriever` in `src/dhara/retrievers/base.py` — `index(chunks)` and `search(query, k) -> list[tuple[chunk_id, score]]` sorted descending. Every rung implements it; `evaluate.py` never learns about specific models. Adding a rung is one file plus one config line.
- `Chunk` dataclass in `src/dhara/schema.py` — imported everywhere, never redefined locally. Metadata *is* the citation, so schema completeness is a correctness requirement.
- `normalize.light` / `normalize.aggressive` — single source of truth. Do not write a local variant; divergent normalization produces irreproducible numbers.

Ownership is by file: A owns `normalize.py`/`scrape.py`/`section_split.py`/`schema.py`, B owns `synth.py`/`negatives.py`, C owns `metrics.py`/`evaluate.py`/classical retrievers, D owns `biencoder.py`/`hybrid.py`/`rerank.py`/`service.py`/the app. Avoid editing another role's file without a reason; `metrics.py` in particular is C's exclusively.

## Working conventions

**`DECISIONS.md` is maintained from day one.** Every non-obvious choice gets date, decision, reason. It is the raw material for the methodology chapter, and it cannot be reconstructed retroactively.

**Mock-first.** `corpus_mock.jsonl` (20 chunks) and `gold_mock.jsonl` (10 questions) in the exact final schema exist in Week 1 so downstream work is never blocked. Real files land later; integration cost is a path change in `configs/paths.yaml`.

**Unit-test `normalize.py` and `metrics.py`.** Normalization bugs are invisible — text looks fine and retrieval is quietly several points worse.

**Scraping conduct is part of the spec**, not politeness: archive raw HTML before parsing, 1.5–2s delay, honest identifying User-Agent, resumable (check file existence before fetch), record `crawl_date` per document.

Branch per role (`feat/A-scraper`, `feat/D-biencoder`), merge to `main` weekly. Anything on `main` must run.

## Known traps (§13.2)

- Fine-tuned model barely beats zero-shot → synthetic questions lexically overlap their source sections. Run the overlap audit (§5.3); this is the single most common cause of a null result here.
- Implausibly high retrieval numbers → gold leakage, or split by question instead of chunk.
- Dense model much worse than expected → missing E5 prefixes, or aggressive normalization on transformer input.
- Demo feels frozen → model reloaded per request instead of once at module level.
- Word2Vec neighbours look like nonsense → expected at 200k–400k words. Report it as a corpus-size finding; do not hide it.

Honest negative results are explicitly worth more here than fudged positive ones — the guide says so repeatedly, and that applies to code changes too: do not "fix" a disappointing number by changing what is measured.

## Ethics constraints in code

This is an information-retrieval tool, **not legal advice** — stated in the UI, the abstract, and the limitations section. Repealed/omitted sections are dropped from the corpus (retrieving one would be actual harm) and the dropped count is recorded. Crawl date is displayed. Mined real questions are stripped of names, phone numbers, NID numbers, and addresses before they enter the dataset.
