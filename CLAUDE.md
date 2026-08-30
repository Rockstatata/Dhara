# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Greenfield.** The repository contains `docs/`, `README.md`, and `AGENTS.md` — no `src/`, `scripts/`, or data yet. The two documents under `docs/` are the authoritative spec:

- [docs/Dhara_Proposal.md](docs/Dhara_Proposal.md) — the *why*: problem framing, scope, evaluation plan, ethics.
- [docs/Dhara_Implementation_Guide.md](docs/Dhara_Implementation_Guide.md) — the *how*: repo layout, schemas, per-phase owners, model configs, week-by-week gates.

When building any part of this project, read the corresponding section of the implementation guide first — it specifies file paths, schemas, and hyperparameters that other people's code is written against. Section references below use the guide's `§` numbering.

## What Dhara is

A Bangla legal retrieval system: a plain-Bangla citizen question in, the exact law **section (ধারা)** plus its citation out. The research claim is that a *lexical gap* separates colloquial citizen phrasing from formal legal Bangla, and that domain fine-tuning closes it. Everything in the design exists to measure that claim, so evaluation integrity outranks model performance in every tradeoff.

Corpus source: the bdlaws portal (structured HTML, no OCR). The scope rule is **frequency in ordinary civilian life**, not tidy legal taxonomy: cover the law people actually collide with. Confirmed domains are family, land, labour, consumer, **cybercrime**, and constitutional; further daily-life domains are under active expansion (see [DECISIONS.md](DECISIONS.md)). Corpus target scales with the domain count — 800–1,200 chunks covered the original four, so a wider scope means a proportionally larger corpus.

There are two audiences for the output. The near one is the course submission in late September / October 2026. The far one is a **conference paper**, which raises the bar on baselines, significance testing, dataset release, and the ethics statement — build for the paper, and the course requirements come free.

## Architecture

Query → normalize → **fine-tuned bi-encoder (dense)** → top-50 → cross-encoder rerank → top-3 sections with citations. **BM25 runs alongside as the reported lexical floor, not as a fused input:** equal-weight RRF measured *worse* than dense alone on the full corpus (R@10 0.216 vs 0.324) because BM25 returns 0% relevant candidates on the 60% of questions whose gold answer is an English-only Act, so half of every fused score is noise. Weighted, dense-dominant fusion is an open ablation, not the default (DECISIONS.md 2026-08-31). An intent classifier runs alongside, reported separately; intent-based candidate filtering stays an ablation, never the default path (a wrong intent prediction silently destroys retrieval).

**The build is organised by course syllabus topic, not by a retrieval ladder.** Read [docs/PIPELINE.md](docs/PIPELINE.md) before building anything — it maps every topic to the artifact in this system that needs it, and explains why each stage exists.

| Topic | Artifact | Job in the system |
|---|---|---|
| 0 | `build_pretrained_embedding_matrix()`, Word2Vec vs general vectors | embedding matrix every RNN below starts from; NN-table is a result |
| 1 | NB + LogReg on TF-IDF, `VanillaRNNClassifier`, `StackedBiLSTMClassifier` | predict `domain`; generative vs discriminative on identical features |
| 2 | `BiRNNSequenceLabeler` | tag ACT / SECTION_NO / LEGAL_TERM / PARTY — extracts "১০৩ ধারা" to boost that provision |
| 3 | `StackedLSTMLanguageModel` | **measuring instrument**: perplexity on colloquial vs formal quantifies the register gap without a retriever |
| 4 | `Seq2SeqTranslation` | **register** translation, colloquial→formal Bangla, trained on the `PAIR` annotations; used as query reformulation |
| 5 | fine-tuned pretrained transformers only | no transformer is written from scratch — Topic 0 answers "explain your embeddings" |

**What the supervisor actually requires** (written guidance, recorded in DECISIONS.md): classification is **not** mandatory — "you are not bound to do classification works, it's an open ended project." The supervised core is the retrieval fine-tuning itself (contrastive learning on labelled question→provision pairs). Topics 1–4 are syllabus demonstrations that earn their place, not requirements. One of generative/discriminative is enough (we keep both; NB is ~20 lines). No GUI required beyond a clear input→output box. Unsupervised clustering "can add value". Data collection **must** be documented.

Build order is Topic 0 → Topic 5 → 3 → 4 → 1 → 2. Topic 5 comes early on purpose: if the central claim fails, find out now.

Two checkpoints, two jobs: `csebuetnlp/banglabert` for classification (ELECTRA, strongest Bangla-specific, needs its own normalizer); a **multilingual** sentence encoder for retrieval, because retrieval must reach the English-only Acts that a Bangla-only model cannot represent.

**BM25 stays but is one row in one table.** It is a TF-IDF-family ranking function, not a language model. It exists because a neural result with no non-neural floor is unpublishable, and because its specific failure — it cannot match a Bangla question to an English provision at all — is one of the things this project measures. Tune `k1`/`b` on dev; an untuned baseline invalidates the comparison. The zero-shot dense control must be locked *before* fine-tuning runs — it is what separates "transformers work" from "our fine-tuning works."

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

**The gold set is human-adjudicated, and the candidate list is lexical on purpose.** 200 questions. An agent may propose candidates but never decides the answer — otherwise the evaluation measures agreement with the same model family that wrote the training questions, and the headline claim becomes circular. Candidates come from **BM25**, not from a dense model: that biases the gold set toward provisions a lexical retriever can find, which makes the project's own thesis *harder* to prove. 15 questions are annotated with no candidate list at all, to measure how often the BM25 list missed the true answer; that miss rate is reported. Every gold question records `annotation_mode` ∈ {`assisted`, `unaided`} and `source` ∈ {`mined`, `authored`} — an authored question honestly labelled is fine, an authored question passed off as mined is not.

**Agreement is measured, not skipped.** With one annotator, inter-annotator κ does not exist; 30 questions are re-annotated blind after a gap and reported as intra-annotator (test–retest) reliability. When a second annotator joins, they double-annotate 50 and both numbers are reported. "We did not measure agreement" is the answer that costs marks and reviewers.

**Differences are claimed only with a confidence interval.** At 200 gold questions the 95% CI on Recall@5 is roughly ±7 points, so adjacent rungs will sometimes be indistinguishable. `metrics.py` provides a **paired bootstrap over `per_query`** and every rung-to-rung comparison reports the CI on the difference. A rung pair honestly reported as within noise reads as more credible, not less. Never narrate a bar chart as if the gaps were all real.

**Split by chunk, not by question.** All synthetic questions generated from chunk X go to the same split, or near-duplicates leak across train/dev.

**Freeze then version.** `corpus_v1.jsonl` never changes after Week 3. A needed fix becomes `corpus_v2.jsonl` and every affected result is re-run or explicitly labelled.

**Two normalization levels, and they are not interchangeable** (§4.5). `normalize.light()` — NFC, ZWNJ strip, whitespace — feeds transformers. `normalize.aggressive()` — plus ZWJ strip, Bangla→ASCII digits, punctuation padding, lowercasing — feeds BM25/Word2Vec/TF-IDF. Applying aggressive normalization to transformer input is a real, measurable performance loss. Digit normalization (১২৩ ↔ 123) is load-bearing here because section numbers appear in both forms.

**Two text fields, deliberately** (§4.4). The UI shows `text_raw` — the law as printed. Models consume `text_bn`. Never display normalized text to a Bangla reader.

**E5 prefixes.** If the checkpoint is from the E5 family, `"query: "` and `"passage: "` prefixes must be applied at index time and query time, in both the zero-shot and fine-tuned runs. Omitting them silently cripples the baseline and invalidates the comparison.

**Hard negatives come from ranks 5–30**, not 1–4. The top few are often relevant-but-unlabelled; training against them teaches the model that correct answers are wrong.

**No hand-typed numbers.** Every number in the report comes from a `results/runs/*.json` emitted by a script. Notebooks are for looking at things only. Every run JSON includes `per_query` — error analysis and significance testing need it and regenerating it later means re-running everything.

**Risk tier drives presentation, never retrieval.** Every domain in `configs/domains.yaml` carries `risk_tier` ∈ {high, medium, low}. For high-tier domains — family, cybercrime, constitutional, criminal procedure, women & children — the UI leads with a legal-aid referral **above** the retrieved provisions, and the abstention threshold is raised. Ranking logic is identical across tiers; only framing and the threshold change. Do not let risk tier filter or reorder candidates.

**Mined questions: verbatim stays local, paraphrase ships.** Newspaper legal-advice columns are copyrighted. Every mined question stores `source_url`, `text_verbatim` (git-ignored, never released), and `text_bn` — a paraphrase preserving register while changing wording — plus a `paraphrased` boolean. The public dataset carries the paraphrase, the URL, and the provision label only.

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

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/` in this repo — not GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, each label string equal to its name (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
