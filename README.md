# Dhara

A Bangla legal retrieval system for citizen services.

Dhara takes a plain-Bangla citizen question and returns the exact law section (ধারা), with its
official citation, that answers it. The research claim: a lexical gap separates colloquial
citizen phrasing from formal legal Bangla — and roughly 60% of the answers citizens actually
need live in **English-only** Acts a Bangla-only model cannot represent at all. Domain
fine-tuning is tested against that gap specifically, not against a generic benchmark.

**Status: built.** Corpus, annotation, zero-shot control, and fine-tuning are done — see
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the full narrative and every cited number,
[docs/DEFENSE_PREP.md](docs/DEFENSE_PREP.md) for a defense-oriented cheat sheet, and
[DECISIONS.md](DECISIONS.md) for the dated decision log. This README is the short version.

## Headline result

| stage | metric | value |
|---|---|---|
| BM25 (lexical floor) | Recall@10 | 0.067 (0% on English-only-Act gold) |
| Zero-shot BGE-m3 (locked control) | Recall@10 | 0.324 (480-pool) / 0.472 (n=36 test) |
| **Fine-tuned BGE-m3 + LoRA (v6, final)** | **Recall@10** | **0.500** (test), **+8.3pt on English-target slice** |

The claim this project stands or falls on: fine-tuning moved English-target test Recall@10 by
+8.3 points (0.333→0.417) — the exact slice the cross-lingual-gap thesis predicts should be
hardest to move. Full ladder, significance tests, and every ablation (reranker regression,
sparse/ColBERT hybrid, dual-query translation) are in the project report.

## Problem statement

A citizen might ask something like:

> পুলিশ আমাকে ধরে নিয়ে গেছে, কিছু বলছে না — আমার কী অধিকার?

The matching legal language is structured around formal phrasing such as:

> গ্রেপ্তার ও আটক সম্পর্কে রক্ষাকবচ

This is a semantic mismatch, not a spelling problem — real citizen questions share **zero**
content words with their answer provision at the median (measured, not assumed). It is not
trivial keyword matching, and a large share of the answers citizens need exist only in
English-only Acts, so it is not a monolingual problem either.

## Scope

Corpus source: the bdlaws government portal (structured HTML, not OCR) — **39,484 chunks across
1,227 Acts**. Scope is set by *frequency in ordinary civilian life*, not legal taxonomy: which
laws people actually collide with. Confirmed domains: family, land, labour, consumer,
cybercrime, constitutional, criminal procedure, and further daily-life domains added as the
project's own gold data revealed gaps. The authoritative list lives in `configs/domains.yaml`;
every addition is recorded in `DECISIONS.md` with its reason and risk tier.

Repealed or omitted provisions are dropped from the corpus and the dropped count is recorded —
retrieving one would be actual harm, not just a lower score.

## Non-goals

The system does not aim to:

- replace legal professionals or give advice in place of a lawyer,
- cover every Act in the national legal ecosystem,
- answer case-law or judicial precedent tasks,
- support multi-turn legal dialogue.

## Core architecture

```text
Bangla citizen query
        |
        v
   normalization
        |
        v
fine-tuned bi-encoder (dense, BGE-m3 + LoRA)
        |
        v
      top-50
        |
        v
 cross-encoder rerank
        |
        v
top-3 sections with citations
```

BM25 runs **alongside** as the reported lexical floor, not as a fused input — equal-weight RRF
measured worse than dense alone (R@10 0.216 vs 0.324) because BM25 returns 0% relevant
candidates whenever the gold answer is an English-only Act, so half of every fused score is
noise. An intent classifier runs alongside too, reported separately; intent-based candidate
filtering stays an ablation, never the default path, because a wrong intent prediction would
silently destroy retrieval. See [docs/PIPELINE.md](docs/PIPELINE.md) for how each syllabus
topic maps to a system artifact, and why.

## What was built, and what each stage found

Full detail, tables, and root-caused failure history are in
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md). Short version:

- **BM25** — Recall@10 0.067, 0% on English-only-Act gold. Its failure mode *is* the thesis, not
  an embarrassment to fuse away.
- **Word2Vec** (self-trained) — Recall@10 0.029. Expected at this corpus size; evidence for why
  a large pretrained multilingual transformer was necessary.
- **Classification** (Topic 1, optional per the supervisor's own guidance) — Logistic Regression
  on TF-IDF, macro-F1 0.343 best (human-only training data outperformed a 12×-larger
  synthetic-heavy pool).
- **Sequence tagger** (Topic 2) — reliably tags `ACT`/`PARTY` (F1 0.857/0.627); `SECTION_NO` and
  `LEGAL_TERM` are near-unlearnable because citizens almost never write them in plain language —
  itself a register-gap finding.
- **Language model perplexity** (Topic 3) — colloquial mean PPL 138.9 vs formal 113.8, driven by
  a heavy tail of unusually hard colloquial questions, not a uniform shift.
- **Fine-tuned BGE-m3 + LoRA** (Topic 5, the core deliverable) — Recall@10 0.324→0.500 on test,
  every intermediate failure (embedding collapse, query-space hubness, a stale-file bug, a
  train-set-dropping policy mismatch) root-caused and fixed, not shrugged at.
- **Cross-encoder reranker** — a real, still-open regression (R@10 0.500→0.361) after training;
  partially root-caused, reported as unresolved rather than hidden.

## Data pipeline

1. **Act survey** — which Acts have Bangla legal text, how many sections, whether the corpus is
   large enough to proceed.
2. **Scraping** — raw HTML archived before parsing; 1.5–2s delay, honest User-Agent, resumable,
   `crawl_date` recorded per document.
3. **Section splitting** — retrieval unit is the section (ধারা), not the Act. Repealed/omitted
   provisions dropped.
4. **Normalization** — two levels, never interchangeable: `normalize.light()` (NFC, ZWNJ strip,
   whitespace) feeds transformers; `normalize.aggressive()` (+ZWJ strip, digit normalization,
   punctuation padding, lowercasing) feeds BM25/Word2Vec/TF-IDF.
5. **Question collection and annotation** — real questions mined from newspaper legal-advice
   columns (verbatim text git-ignored, only a paraphrase ships publicly) plus quality-gated
   synthetic questions, adjudicated into a 552-row gold pool (480 answerable, 72 deliberately
   unanswerable for abstention calibration). Gold set is touched exactly once per model-selection
   cycle, never for debugging.

## Dataset and schema

Frozen, versioned artifacts under `data/processed/` — `corpus_v1.jsonl` (39,484 chunks),
`gold_verified_v3.jsonl` (552 rows), `train_retrieval_v7.jsonl` / `dev_retrieval_v4.jsonl` /
`test_retrieval_v4.jsonl` (the final human-aware, provision-connected-component split), plus the
mined hard-negative pools. Large generated artifacts (raw scrapes, model checkpoints, full
corpus/synthetic pools over ~50MB) are git-ignored and kept off GitHub — see `.gitignore`.

The `Chunk` schema (`src/dhara/schema.py`) carries the citation metadata as a correctness
requirement: `chunk_id`, `act_name_bn`, `act_year`, `chapter`, `section_no`, `section_title_bn`,
`text_raw`, `text_bn`, `domain`, `source_url`, `crawl_date`, `char_len`. `text_raw` (the law as
printed) is what the UI shows a human reader; `text_bn` is what models consume — never the
reverse.

## Repository structure

```text
configs/     domains.yaml, abstention.json, synth_*.yaml, split configs
data/raw/    scraped HTML — never committed
data/processed/  frozen JSONL artifacts — committed when under GitHub's size limit
src/dhara/   normalize.py, schema.py, scrape.py, section_split.py, synth.py, negatives.py,
             metrics.py, evaluate.py, classify.py, service.py, annotation_round2.py,
             retrievers/{base,bm25,word2vec,biencoder}.py
src/app/     app.py — Gradio demo (Search / Law Corpus / Gold Q&A tabs)
scripts/     01–76, thin CLIs — one job each (numbering has drifted from the implementation
             guide's plan; see CLAUDE.md's "Commands" section for the mapping)
notebooks/   colab_bge_m3_finetune.ipynb, colab_acttitle_and_rerank.ipynb, and later
             fine-tuning/summarization notebook iterations
results/     tables/ figures/ runs/ — one JSON per experiment run, every claim traces here
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
# BanglaBERT workflows need this separately (not on PyPI):
pip install git+https://github.com/csebuetnlp/normalizer
```

Compute target is a Colab/Kaggle T4; nothing here needs more.

## Commands

See [CLAUDE.md](CLAUDE.md)'s "Commands" section for the current, accurate script list — corpus
build, annotation merge, dense-index evaluation, the synthetic-training-data chain, hard-negative
mining, and diagnostics (hubness audit, rerank scoring, question-diversity). Every citable number
comes out of `scripts/13_eval_dense_index.py` + `scripts/18_compare_runs.py`, never hand-typed.

```bash
python -m src.app.app   # Gradio demo — Search / Law Corpus / Gold Q&A
```

## Evaluation methodology

- **Split by provision-connected-component, not by question** — correlated citizen phrasings of
  the same provision stay together, so none leak across train/dev/test.
- **Every retrieval run JSON carries `per_query`**, so `scripts/18_compare_runs.py` can run a
  paired bootstrap significance test on any claimed delta — a point estimate alone is never
  treated as a real result.
- **Gold set isolation**, enforced by an automatic leakage assertion at data-load time in every
  training notebook.
- At the current test size (n=36), one question is worth ~2.8 points of Recall@10 — every delta
  in the project report is read against that, and several are stated as not-yet-CI-confirmed
  rather than oversold.

## Non-negotiable rules

The full list (gold set isolation, split-by-chunk, freeze-then-version, the two normalization
levels, E5 prefix handling, hard negatives from ranks 5–30, no hand-typed numbers, risk-tier
framing without touching ranking) lives in [CLAUDE.md](CLAUDE.md) — treat it as binding on any
code change, not just documentation.

## Ethics and safety

This is an information-retrieval tool, **not legal advice** — stated in the UI, and in every
document above. Repealed/omitted legislation is excluded from the corpus and the dropped count is
recorded; crawl date is displayed; mined real questions are stripped of names, phone numbers, NID
numbers, and addresses before they enter the dataset; high-risk domains (family, cybercrime,
constitutional, criminal procedure, women & children) get a legal-aid referral banner above
results and a raised abstention threshold — framing only, never a change to ranking logic.

## Team

Two credited members: Sarwad and Iftiaq. Role-based file ownership (normalize/scrape/schema;
synth/negatives; metrics/evaluate/classical retrievers; biencoder/rerank/service/app) is recorded
in CLAUDE.md; `DECISIONS.md` is the running log of who decided what and why.

## Documentation map

- [docs/Dhara_Proposal.md](docs/Dhara_Proposal.md) — the *why*: problem framing, scope, ethics.
- [docs/Dhara_Implementation_Guide.md](docs/Dhara_Implementation_Guide.md) — the *how*: schemas,
  hyperparameters, per-phase owners.
- [docs/PIPELINE.md](docs/PIPELINE.md) — maps syllabus topics to system artifacts.
- [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) — the full narrative, every cited number.
- [docs/DEFENSE_PREP.md](docs/DEFENSE_PREP.md) — defense-oriented cheat sheet.
- [docs/TECH_STACK.md](docs/TECH_STACK.md), [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) —
  reference material for onboarding into the codebase.
- [DECISIONS.md](DECISIONS.md) — dated decision log, the raw material for the methodology chapter.
