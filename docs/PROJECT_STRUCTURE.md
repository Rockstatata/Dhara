# Dhara — project structure: the whole repo, mapped

Written 2026-09-19. This is the "hold the whole project in your head" document: what every
directory is for, how the pieces connect at runtime, and — honestly — which parts
`docs/TECHNICAL_DEEPDIVE.md` already covers in depth versus which parts are real, working code
that hasn't had its own deep-dive pass yet. Read this first if you're new to the repo; read
`TECHNICAL_DEEPDIVE.md` for how a specific piece works; read `PROJECT_REPORT.md` for results.

## Coverage honesty check, up front

While building this document, a full script inventory turned up **one entire sub-project that
neither `TECHNICAL_DEEPDIVE.md` nor `PROJECT_REPORT.md` currently documents**: provision
**summarization** (`scripts/70_build_provision_summaries.py`, `71_split_summaries.py`,
`72_summary_extractive_baselines.py`, `notebooks/ssh-summarize-finetune.ipynb`). It's real,
working code — hand-authored provision summaries with the same honesty-labelling discipline as
the authored questions (`label_source: llm_authored_v1`, `verified: false` until a human checks
it), two non-neural extractive baselines (first-sentence, TextRank) scored with ROUGE-1/2/L and
chrF against a frozen 19-row test split, and an mT5/BanglaT5 fine-tuning notebook — but it has
not had a `TECHNICAL_DEEPDIVE.md`-style pass. Flagged here rather than silently left out; ask for
a follow-up pass on it specifically if it needs to be defense-ready.

`TECHNICAL_DEEPDIVE.md` also does not narrate every one of the ~75 scripts individually — many
are superseded iterations of the same idea (there are, for instance, three generations of
"build a leakage-safe retrieval split" script). The **script index** in §3 below accounts for
every script in the repo by name, with a one-line purpose and a current/superseded/diagnostic
tag, specifically so nothing is invisible even where it doesn't get full prose treatment.

---

## 1. Top-level layout

```
Dhara/
├── configs/        Frozen, versioned settings — domains, model checkpoints, abstention thresholds
├── data/           raw/ (scraped, never edited) → interim/ (git-ignored, PII-bearing) →
│                   processed/ (frozen, committed JSONL) → annotation/ (CSVs) → lexicon/
├── docs/           This document, TECHNICAL_DEEPDIVE.md, PROJECT_REPORT.md, DEFENSE_PREP.md,
│                   the original proposal + implementation guide, PIPELINE.md
├── models/         Trained checkpoints and dense indexes (embeddings.npy + chunk_ids.json +
│                   manifest.json per index) — gitignored, large binaries
├── notebooks/      GPU-bound training runs (LoRA fine-tune, summarization fine-tune) that don't
│                   fit this repo's CPU-only dev machine
├── results/        runs/ (one JSON per experiment, per-query data included), tables/, figures/
├── scripts/        ~75 numbered, single-purpose CLIs — the pipeline, in execution order
├── src/app/        The Gradio demo (3 tabs)
├── src/dhara/      The actual library code every script and the app import from
├── tests/          pytest unit tests (currently: normalize.py, metrics.py — the two modules
│                   CLAUDE.md itself calls out as needing tests, since their bugs are invisible)
├── scratch/        Ad hoc one-off analysis scripts, not part of the pipeline
├── DECISIONS.md    Dated, append-only decision log — the raw material this project's own
│                   "why" answers are pulled from, never rewritten, only corrected forward
└── CLAUDE.md       The project's own operating rules (non-negotiables, known traps, code
                    ownership) — read this first if you're an agent, not a human
```

## 2. `src/dhara/` — the library, and how its modules depend on each other

```
schema.py         <- imported by everything. Chunk dataclass. No other module redefines it.
normalize.py      <- imported by everything that touches text. light()/aggressive()/content_tokens().
vocab.py          <- imported by classify.py, models/tagger.py, models/language_model.py.
                     Vocab + build_pretrained_embedding_matrix() (Topic 0's shared artifact).

scrape.py  ─┐
blad.py    ─┴─> corpus construction. scrape.py fetches HTML; blad.py parses BOTH scraped HTML
                and the BLAD dataset into Chunk objects (provision splitting, chunking, repeal
                detection, dedup). Neither imports the other; scripts/02 and /03 glue them.

mine.py    ─┐
questions.py ┴─> real citizen question collection. mine.py fetches source articles;
                 questions.py splits them into individual Question objects, PII-scrubs, tracks
                 split confidence.

synth.py        <- synthetic question generation (topic-template matching, overlap audit,
                    narrative scaffolding, intent-based positive selection). Imports normalize.py
                    for content_overlap(), nothing else from this package.

annotation_round2.py <- the double-annotation/adjudication data model (tested in tests/).

word2vec.py           <- from-scratch skip-gram trainer. Standalone; only vocab.py is shared.
classify.py           <- TF-IDF NB/LogReg + VanillaRNNClassifier/StackedBiLSTMClassifier.
                          Imports normalize.py (aggressive/content_tokens) and vocab.py.
models/tagger.py           <- BiRNNSequenceLabeler. Imports vocab.py.
models/language_model.py   <- StackedLSTMLanguageModel + perplexity(). Imports vocab.py.

metrics.py            <- Recall@k / MRR / paired_bootstrap. Imports nothing from this package
                          (pure functions over ranks) — deliberately dependency-free so any
                          script can import it without pulling in a model.

retrievers/base.py         <- Retriever ABC: index(chunks), search(query, k). Every retriever
                               below implements this so evaluate.py / eval scripts never need
                               to know which concrete retriever they're scoring.
retrievers/bm25.py          <- implements base.Retriever. Imports normalize.py (content_tokens).
retrievers/word2vec.py      <- implements base.Retriever. Wraps word2vec.py's trained centre
                               table. Imports normalize.py (aggressive, NOT content_tokens).
retrievers/biencoder.py     <- implements base.Retriever. Wraps sentence_transformers. Imports
                               normalize.py (light) and schema.py (Chunk).

service.py             <- the ONE place retrieval + abstention + risk-tier framing meet.
                           Imports schema.py and normalize.py directly (not the retrievers/
                           package — it loads a precomputed index + a raw SentenceTransformer
                           itself, rather than going through BiEncoderRetriever, because the
                           demo needs abstention/risk-tier logic BiEncoderRetriever doesn't have).
                           This is the module src/app/app.py calls.
```

**Two retrieval code paths exist and don't share an implementation** — worth knowing precisely
which is which: `retrievers/biencoder.py`'s `BiEncoderRetriever` is the "pure" `Retriever`
interface implementation, used by evaluation scripts (`13_eval_dense_index.py`,
`26_eval_bm25.py`, `36_eval_word2vec.py`) to score any rung interchangeably. `service.py`'s
`Dhara` class is a separate, purpose-built serving path with its own index-loading, abstention,
and risk-tier logic that the demo app actually calls — it does not go through
`BiEncoderRetriever`. This is intentional (serving needs abstention/citation/risk-framing that a
generic `Retriever.search()` contract has no place for) but means a change to encoding logic in
one does not automatically propagate to the other.

## 3. `scripts/` — full index, every script accounted for, organized by pipeline stage

Numbering is roughly chronological, not strictly staged — later scripts often supersede earlier
ones for the same job (three generations of split-builder, three of negative-miner). **Current**
= what the live pipeline actually uses today. **Superseded** = real, working code from an earlier
iteration, kept for history/reproducibility, not what current results come from. **Diagnostic** =
one-off investigation, not re-run routinely.

| script | purpose | status |
|---|---|---|
| `02_scrape.py` | fetch bdlaws HTML, archive-first, resumable | current |
| `03_build_corpus.py` | BLAD + scraped HTML → `corpus_v1.jsonl` via `blad.py` | current (frozen output) |
| `04_mine_questions.py` | fetch real citizen questions from the 4 news sources | current |
| `05_split_questions.py` | article text → individual `Question`s via `questions.py` | current |
| `06_make_annotation_sheets.py` | build the annotator CSVs | current |
| `07_search_corpus.py` | BM25 candidate list for annotators (never dense — keeps the gold set unbiased toward what a lexical retriever can find) | current |
| `08_merge_gold.py` | merge annotated sheets → `gold_verified_*.jsonl` | current |
| `09_autolabel_pass1.py` | first-pass weak labelling to speed up manual annotation | superseded (early bootstrap, output hand-corrected since) |
| `10_train_word2vec.py` | train the from-scratch skip-gram model | current |
| `11_extract_answers.py` | pull candidate answer provisions during annotation | current |
| `12_zeroshot_probe.py` | early zero-shot recall probe | superseded by `13` |
| `13_eval_dense_index.py` | **the** dense-retriever eval entrypoint, any checkpoint | current |
| `14_generate_questions.py` | synthetic question generation (topic-template pipeline, `synth.py`) | current |
| `15_overlap_audit.py` | §5.3 lexical-overlap quality gate, standalone report | current |
| `16_split_training.py` | original topic-disjoint train/dev/test split | superseded by `51`/`68` for retrieval; still the reference implementation the later ones evolved from |
| `17_mine_negatives.py` | original hard-negative miner | superseded by `67`/`71`/`74` |
| `18_compare_runs.py` | paired bootstrap significance test between two runs | current — the single most-used script in the repo |
| `19_build_exclusions.py` | repealed-provision exclusion list for index + serving | current |
| `20_hubness_audit.py` | query-space hubness diagnostic (root-caused fine-tune attempt 2's flatness) | diagnostic, one-off |
| `21_eval_rerank.py` | score an arbitrary reranked candidate list | diagnostic |
| `22_question_diversity.py` | question-set structural-shape audit | diagnostic |
| `23_preflight.py` | pre-run sanity checks | utility |
| `24_gold_audit.py` | gold-set integrity checks | utility |
| `25_make_verification_sheets.py` | annotator verification-pass CSVs | current |
| `26_eval_bm25.py` | BM25 k1/b tuning + eval | current |
| `27_label_triage.py` | flags likely-mislabelled rows for review | utility |
| `28_result_slices.py` | language/domain slice reporting over a run | current |
| `29_build_test_split.py` | builds `test_split_v1.json` — the frozen 352/200 split `classify.py`/`tagger.py` both key off of | current |
| `30_make_paraphrase_sheets.py` | paraphrase annotation CSVs (verbatim→released text) | current |
| `31_finalize_annotation_round2.py` | double-annotation adjudication finalize | current |
| `32_build_probes.py` | build probe-question sets for eval | current |
| `33_merge_attempt4_checkpoint.py` | merge a specific fine-tune attempt's LoRA weights | superseded (attempt4 branch is dead) |
| `34_embed_probes.py` | cache probe-question embeddings | utility |
| `35_build_attempt4_corpus_embeddings.py` | corpus embeddings for the (now dead) attempt4 checkpoint | superseded |
| `36_eval_word2vec.py` | Word2Vec-as-retriever eval | current |
| `39_train_language_model.py` | train `StackedLSTMLanguageModel` | current |
| `40_measure_perplexity.py` | colloquial-vs-formal perplexity measurement | current |
| `42_train_classifiers.py` | Topic 1: NB/LogReg/VanillaRNN/StackedBiLSTM, 4 data variants | current |
| `43_train_tagger.py` | Topic 2: weak-label `BiRNNSequenceLabeler` | current |
| `46_calibrate_abstention.py` | sweep abstention threshold vs. answerable/unanswerable gold | current in design, **stale output** as of 2026-09-19 (calibrated against a retired checkpoint; `configs/abstention.json` currently carries a provisional manual recalibration pending a clean re-run — see that file's own note) |
| `47_curate_retrieval_training.py` | early retrieval-training curation pass | superseded |
| `48_build_retrieval_colab_splits.py` | early Colab-targeted split builder | superseded by `51`/`68` |
| `49_promote_approved_retrieval_pairs.py` | silver→gold promotion of reviewed title-pairs (`--approve-all`) | current — still feeds the v6/v7 pools |
| `50_build_full_coverage_synthetic_v2.py` | intermediate synthetic-coverage generation | superseded |
| `51_build_human_aware_retrieval_splits_v3.py` | 2nd-gen leakage-aware split builder | superseded by `68` (v4) — this is the version whose provision-overlap policy mismatched `66`'s and caused the 52-row human-train bug |
| `52_audit_retrieval_distribution_v1.py` | training-pool distribution audit | diagnostic |
| `53_prepare_diverse_augmentation_prompts_v1.py` … `58_promote_diverse_augmentation_v1.py` | LLM-augmentation generation pipeline v1 (prepare→generate→validate→promote) | explored, **not in the final v6 training pool** (`augmentation_train` is empty in the current data) — a real, working pipeline that ended up unused |
| `59_generate_validate_all_v1.py` | batch driver for the v1 augmentation pipeline | same status as above |
| `60_prepare_diverse_augmentation_prompts_v2.py` … `63_promote_diverse_augmentation_v2.py` | augmentation pipeline v2 | same status — explored, not in the final pool |
| `64_generate_coverage_questions_v3.py` | 3rd-gen coverage-question generation | superseded/intermediate |
| `65_build_authored_questions.py` | hand-written question pool, quality-gated like mined questions | current — feeds v6/v7 |
| `66_merge_authored_into_pool.py` | merge authored questions into the training pool | superseded by `70` (v6) — historically the script whose provision-exclusion policy silently dropped 199/251 human rows |
| `67_mine_negatives_v5.py` | negative miner, v5 pool | superseded by `71`/`74` |
| `68_build_human_aware_retrieval_splits_v4.py` | **current** split builder — connected-component leakage prevention + component-size cap + domain round-robin fill | current |
| `70_build_provision_summaries.py` | **summarization sub-project** — hand-authored provision summaries, scored not generated | current, undocumented elsewhere (see coverage note above) |
| `70_merge_pool_v6.py` | **the current retrieval training pool builder** (v6: 2,685 approved + 942 authored + 408 human) | current |
| `71_mine_negatives_v6.py` | hard-negative miner for the v6 pool | current |
| `71_split_summaries.py` | summarization train/dev/test split | current, undocumented elsewhere |
| `72_pool_eval_v4.py` | pooled dev+test eval-set builder | superseded — its output was contaminated by `73_generate_bilingual_query_expansions.py`'s leak and is quarantined in `data/processed/_archived_leaked_bilingual/` |
| `72_summary_extractive_baselines.py` | first-sentence + TextRank summarization baselines, ROUGE/chrF | current, undocumented elsewhere |
| `73_build_balanced_pool_v7.py` | quality-filtered, size-capped alternative training pool (v7) | current as an artifact/ablation — tested, scored slightly worse than v6 (test R@10 0.472 vs 0.500), not the deployed pool |
| `73_generate_bilingual_query_expansions.py` | **RETRACTED** — hardcodes gold Act/section into query text (answer leakage). Guarded with a hard `SystemExit`; do not run | retracted, guarded |
| `74_mine_negatives_v7.py` | hard-negative miner for the v7 pool | current (feeds the v7 ablation) |
| `75_merge_annotator_identities.py` | one-time migration: reattribute two early contributors' annotation rows to the two credited project members | current (one-time, already run) |
| `76_build_final_demo_index.py` | build the demo's dense index from the final v6 checkpoint, full 39,484-chunk corpus | current |

## 4. `data/` — provenance chain

```
data/raw/          scraped HTML, never edited, never committed
data/interim/       git-ignored. PII-bearing verbatim mined questions live here and ONLY here.
data/processed/     frozen, committed JSONL. corpus_v1.jsonl, gold_verified_v3.jsonl,
                     train/dev/test_retrieval_v*.jsonl, classify/tagger results, etc.
data/annotation/     the human annotation CSVs (verify_*.csv, paraphrase_*.csv,
                     adjudication_round2.csv) and their archived pre-merge originals
data/lexicon/        register_map.json (formal-legal-term gazetteer, used by the tagger and
                     synth.py's tier-b lexicon rewrite)
data/external/       third-party reference data (BLAD dataset, etc.)
data/finetune-results/  raw artifacts pulled back from the SSH GPU box
```

## 5. `models/` — checkpoints and indexes

Every dense index directory follows the same schema (`embeddings.npy`, `chunk_ids.json`,
`manifest.json` — the manifest records the exact checkpoint path, document template, and
normalization used to build it, so an index is self-describing and can't silently drift from
what built it). `models/bge_m3_finetuned_v6/` is the actual merged LoRA checkpoint (a full
sentence-transformers export); `models/index_bge_m3_finetuned_v6/` is the corpus encoded through
it — two different things that are easy to conflate by name alone.

## 6. Request flow at serving time

```
Bangla question
    │
    ▼
src/app/app.py (Gradio "Search" tab)
    │  respond(query, k) → dhara.search(query, k, show_all=True)
    ▼
src/dhara/service.py — Dhara.search()
    │  1. light-normalize the query
    │  2. encode with the fine-tuned checkpoint (models/bge_m3_finetuned_v6)
    │  3. dot-product against the precomputed corpus matrix (models/index_bge_m3_finetuned_v6)
    │  4. over-fetch, dedupe to one hit per provision_id
    │  5. detect risk tier from a FIXED lookahead (independent of k — a fixed bug)
    │  6. compare top score against configs/abstention.json's threshold for that tier
    │  7. return an Answer: hits, abstained flag, top_score, threshold, risk framing
    ▼
src/app/app.py renders hit cards + (if abstained) a soft low-confidence banner,
    never a hard block — hits are shown either way per explicit instruction
```

The **Law Corpus** and **Gold Q&A** tabs are a separate, read-only path (`src/app/browse.py`)
that loads `corpus_v1.jsonl` and `gold_verified_v3.jsonl` directly at app startup — they never
touch the retrieval model at all, which is why they load in ~1 second while the search tab's
model load takes longer.
