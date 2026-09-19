# Session handoff

**CURRENT STATUS — 2026-09-16: the lab pipeline is complete.** The dated log
below is historical context only. Do not follow its early “pending” commands
without first checking the final artifacts listed here.

- Final human + LLM annotation: `data/processed/gold_verified_v2.jsonl` and
  `data/processed/questions_release_v2.jsonl`; validate with
  `python scripts/31_finalize_annotation_round2.py --dry-run`.
- Comparable retrieval evaluation: use the 480-probe v2 artifacts and
  `results/tables/ladder_v2_complete.csv`, with register/domain slices beside
  it. Never mix a run with a different qid population into this table;
  `scripts/28_result_slices.py` now enforces that invariant.
- Completed supporting NLP artifacts: `models/bilstm_encoder.pt`,
  `models/language_model.pt`, `models/seq2seq_register.pt`, and
  `models/tagger.pt`, with their run JSONs under `results/runs/`.
- Demo default: `models/index_bge_m3_acttitle_v1`, served through
  `src/dhara/service.py` and `src/app/app.py`, with calibrated thresholds in
  `configs/abstention.json`.
- The Seq2Seq reformulator, weak-label tagger, and clustering experiment are
  completed ablations, not demo defaults. Their limitations are documented in
  `DECISIONS.md` and their run JSONs.

Run the demo from the repository root with:

```bash
python -m src.app.app
```

Run final checks with:

```bash
python scripts/31_finalize_annotation_round2.py --dry-run
pytest -q -p no:cacheprovider
```

**Read this section first.** The rest of the file is a dated log of earlier
sessions, appended in order; where it disagrees with this section, this section
is current.

## STATUS as of 2026-09-04 — everything is built; two Colab runs are pending

### The state of the central claim

| run | R@10 (all) | what happened |
|---|---|---|
| BM25 (lexical floor) | 0.067 | cross-language wall: 0.000 on English-gold |
| **BGE-m3 zero-shot (the control)** | **0.324** | locked 2026-08-31, never rebuilt |
| equal-weight RRF fusion | 0.216 | worse than dense alone; not the default path |
| fine-tune attempt 1 | 0.060 | embedding collapse — two bugs, both fixed |
| fine-tune attempt 2 | 0.330 | flat; root-caused to query-space hubness |
| fine-tune attempt 3 | — | **not yet run** — data and notebook ready |
| act-title + rerank | — | **not yet run** — notebook ready |

Fine-tuning has not beaten the zero-shot control. Two attempts, each with a
diagnosed cause rather than a shrug. That is a publishable result as it stands.

### The two things to run, in this order

**1. `notebooks/colab_acttitle_and_rerank.ipynb` — do this one first.**
No training. Upload `corpus_v1.jsonl.gz` and `probe_questions.jsonl`, run all.
It is where the measured headroom actually is: 30.3% of questions have their
gold provision sitting in ranks 11-100 with no second stage to pull it up, and
a perfect reranker over the top 100 would take R@10 from 0.324 to 0.627. It
also tests putting the Act title in the embedded document, which the zero-shot
index never did.

Then in the repo:
```bash
mkdir -p models/index_bge_m3_acttitle_v1
mv embeddings.npy chunk_ids.json probe_query.npy manifest.json models/index_bge_m3_acttitle_v1/
python scripts/13_eval_dense_index.py --index models/index_bge_m3_acttitle_v1 --run-id bge_m3_acttitle
python scripts/18_compare_runs.py --a results/runs/bge_m3_acttitle.json --b results/runs/bge_m3_zeroshot.json

mv rerank_bge_v2m3_top100.jsonl data/processed/
python scripts/21_eval_rerank.py --rerank data/processed/rerank_bge_v2m3_top100.jsonl --run-id bge_m3_acttitle_rerank
python scripts/18_compare_runs.py --a results/runs/bge_m3_acttitle_rerank.json --b results/runs/bge_m3_zeroshot.json
```

**2. `notebooks/colab_bge_m3_finetune.ipynb` — attempt 3.**
Training data was rewritten as narratives, then filtered to hand-anchored labels
only after ~80% of the keyword-matched positives were found to be wrong
(DECISIONS.md 2026-09-04 late). Upload the four files listed in the notebook's
own header — the **anchor** variant now, not strict:
`train_anchor_v1_negatives.jsonl` (5,332 pairs / 826 questions / 173 positive
provisions), `dev_anchor_v1.jsonl`, `probe_questions.jsonl`, `corpus_v1.jsonl.gz`.
Run `python scripts/23_preflight.py --variant anchor` first and check the printed
SHA-1s against what lands in Colab. Watch two printed numbers
after training: canary cosine (want 0.7-0.98) and query-concentration delta
(want under +0.06). Then:
```bash
mkdir -p models/index_bge_m3_finetuned_v3
mv embeddings.npy chunk_ids.json probe_query.npy manifest.json models/index_bge_m3_finetuned_v3/
python scripts/13_eval_dense_index.py --index models/index_bge_m3_finetuned_v3 --run-id bge_m3_finetuned_anchor_v3
python scripts/18_compare_runs.py --a results/runs/bge_m3_finetuned_anchor_v3.json --b results/runs/bge_m3_zeroshot.json
python scripts/20_hubness_audit.py \
    --a models/index_bge_m3_finetuned_v3 --a-id bge_m3_finetuned_anchor_v3 \
    --b models/index_bge_m3_zeroshot_v1  --b-id bge_m3_zeroshot
```

**Delete the Colab runtime between runs.** The upload cell skips silently if
files with the same names are already on disk from a previous session, which
will quietly train attempt 3 on attempt 2's data.

### What changed on 2026-09-04

- Training questions rewritten as narrative letters. Real questions have median
  74 words and sd 56.3; the old templates were one-liners at median 11 / sd
  3.41. Now median 55 / sd 30.3. Distinct questions 413 -> 1,239, pairs
  3,774 -> 11,395. `configs/synth_narrative.yaml` + `Narrative` in
  `src/dhara/synth.py`.
- **Correction to the 2026-09-03 entry**: it blamed attempt 2's concentration on
  training-question homogeneity. Measured against the base encoder that is
  wrong — the short templates were no more concentrated than the real questions
  (0.6983 vs 0.6990). The concentration came from training dynamics, so
  attempt 3 also drops to 1 epoch at LR 1e-5. Two separate defects, two separate
  fixes; the narrative rewrite is not claimed to fix hubness.
- New instruments: `scripts/20_hubness_audit.py` (diagnoses hubness after a
  fine-tune), `scripts/21_eval_rerank.py` (scores a reranked list into the same
  schema as `13`), `scripts/22_question_diversity.py` (measures question-set
  shape on CPU, before spending GPU time).
- Full regeneration chain re-run: 14 -> 15 -> 16 -> 17. Audit verdict unchanged
  (tier_a passes at median 0.000, tier_b/c still fail), leakage assertions pass.
- Emotional background clauses gated to sensitive domains only (a
  "stayed silent out of shame" clause was attaching to deed-correction
  questions).
- Fixed a cache-invalidation bug in `17_mine_negatives.py`: the query-embedding
  cache validated on question *count*, and the regeneration left the count at
  1,239 while every string changed. It now fingerprints the question list.
- **Run `python scripts/23_preflight.py` before any GPU run.** It checks
  negatives/split alignment, gold isolation, train-dev disjointness, negative
  contamination, tier approval, question shape, and prints SHA-1s of the four
  upload files. Every check in it corresponds to a failure that has actually
  happened.

### One open decision for you

`data/processed/` now holds ~150 MB of **derived** training files
(`synth_questions_v1` 46 MB, `train_full_v1` 34 MB, the two `_negatives` files
~48 MB, `train_strict_v1` 25 MB). CLAUDE.md says "`data/processed/*.jsonl` **is**
committed — it is small and it is what makes results reproducible", and that
rule was written when these files did not exist.

They are all deterministically regenerable from the corpus plus
`configs/synth_*.yaml`, `data/lexicon/`, and scripts 14-17 — verified
byte-identical on a re-run — and every one of those inputs is small and
committed. So committing the derived files costs ~150 MB for no reproducibility
gain. The corpus itself is a different case: it cannot be regenerated without
re-scraping, so it earns its place.

Not changed unilaterally, because it is a repo-policy call and CLAUDE.md states
the rule explicitly. If you agree, add to `.gitignore`:

```
data/processed/synth_questions_v1.jsonl
data/processed/train_*.jsonl
data/processed/dev_*.jsonl
```

and amend the CLAUDE.md rule to say committed *frozen* artifacts, not derived
ones.

### Known-good invariants — do not break these

- The zero-shot control (`models/index_bge_m3_zeroshot_v1`,
  `results/runs/bge_m3_zeroshot.json`) is frozen. Every comparison is against it.
- Every rung is scored by `13_eval_dense_index.py` or `21_eval_rerank.py`, both
  emitting the same schema, and compared only by `18_compare_runs.py` with a
  paired bootstrap. Never hand-compare two recall tables.
- Failed attempts stay on disk with their own run-ids and index directories.
  Nothing overwrites anything.
- `gold_test_v1.jsonl` has still never been touched.

### Verified before the GPU runs (2026-09-04, late)

The whole synthetic chain was replayed after a further opener fix, and both
scoring paths were tested without a GPU, so the Colab output lands on code that
is already known to work.

**`scripts/23_preflight.py --variant strict` is clean** — 12 hard checks pass,
no warnings. Question shape now reads median 50 words / sd 26.1 against the real
questions' 67 / 50.9 (preflight splits on whitespace; `22_question_diversity.py`
reports 57 / 30.3 against 74 / 56.3 because it counts `normalize.aggressive()`
tokens — same data, two tokenizations, so compare each instrument only to
itself).

Upload fingerprints — check these on the Colab side if a run looks wrong:

| sha1 (first 12) | file | size |
|---|---|---|
| `875c2e13ee2b` | `train_strict_v1.jsonl` | 16.9 MB |
| `3ed038ebddf0` | `train_strict_v1_negatives.jsonl` | 18.7 MB |
| `7d7ab283aaba` | `dev_strict_v1.jsonl` | 2.2 MB |
| `c810e7d42787` | `probe_questions.jsonl` | 0.8 MB |

Current artifacts: 28,953 synthetic pairs, 7,586 train / 983 dev, 826 distinct
training questions, 60,633 mined negatives (mean 7.99 per pair, 0 pairs empty,
contamination under 1%). The stale-cache guard fired as designed — the query
cache was invalidated by content hash, not row count, and re-encoded.

**`21_eval_rerank.py` and `18_compare_runs.py` were smoke-tested against an
identity reranker** (the zero-shot index's own top-100, reordered by nothing).
It reproduced the control exactly — R@10 0.324, recoverable ceiling 0.627 — and
the paired bootstrap reported 0 of 24 comparisons outside the noise floor, which
is the correct answer when a run is compared against itself. The rerank scoring
path is therefore known-good before any cross-encoder output exists.

### BLOCKER on the evaluation set (2026-09-04, late)

`scripts/24_gold_audit.py` over `data/annotation/annotate_*.csv` found the gold
set unverifiable from what is in the repo: 794/794 rows machine-filled, 0 rows
with `verified` set, 0 human-written notes, and **0 differing answers across 270
double-annotated annotator pairs**. Kappa is undefined on that; the 1.000
exact-match rate is an artifact of identical pre-filled answers, not agreement.

The verification pass *was* done — in a shared Google Sheet that was never
exported back. The CSVs in the repo are the pre-verification copies.

**Next action, and it gates what the paper may claim about the evaluation set:**

1. Export each annotator's tab from the Google Sheet as CSV into a new directory,
   e.g. `data/annotation/verified/`, keeping the `annotate_<name>.csv` naming.
2. ```bash
   python scripts/24_gold_audit.py --verified data/annotation/verified
   ```
   This prints the per-annotator edit rate (how often a human overrode the
   machine proposal) and Cohen's kappa on `verdict`/`register`/`domain` plus
   provision-set exact/overlap/Jaccard for the double-annotated block.
3. Re-merge into the gold set with `scripts/08_merge_gold.py` pointed at the
   exported directory, then re-check `probe_questions.jsonl` against it.

If the edit rate is healthy, the gold set is adjudicated and the kappa is real —
report both. If it is near zero, the fallback is already built:
`verify_{Iftiaq,Sarwad}.csv (two other early contributors' sheets later merged in, 2026-09-19)` (55 rows each, `answer` column empty,
100 VERIFY + 60 DOUBLE questions) with the prior labels held in
`data/annotation/verification_key_v1.jsonl`.

Until then, describe the gold set as "machine-proposed, verification not recorded
in the repository". No measured result changes; the claim about the instrument does.

## Session ruleset in effect

CAVEMAN MODE (full) was active — terse prose in chat, normal prose in files. Not
a project requirement, just a chat-session setting. Turn off with "stop caveman"
or ignore it in a new chat.

## Project in one paragraph

Dhara: Bangla legal retrieval. Plain-Bangla citizen question in → exact law
section (ধারা) + citation out. Research claim: a lexical gap separates colloquial
citizen phrasing from formal legal Bangla, and domain fine-tuning closes it.
Corpus: bdlaws portal, `data/processed/corpus_v1.jsonl`, **39,484 chunks, 1,227
acts, frozen**. Two audiences: course submission (late Sept/Oct 2026) and a
conference paper. Evaluation integrity outranks model performance in every
tradeoff. Authoritative spec: `docs/Dhara_Proposal.md`,
`docs/Dhara_Implementation_Guide.md`, `CLAUDE.md`, `DECISIONS.md`.

## What this session did

### 1. LLM-assisted annotation pass (complete)

- Claude proposed a provision answer for **all 569 unique assisted questions**
  across the annotator sheets (`data/annotation/annotate_{Iftiaq,Sarwad}.csv`; two other early contributors' sheets were later merged into these two credited names, 2026-09-19),
  using its own legal knowledge + the question + the recovered newspaper advocate
  reply, with dense (e5) + BM25 candidate bundles only as a search aid.
- Suggestions first went into a separate lane (`llm_suggest` / `llm_rationale` /
  `llm_confidence` / `llm_flag`) to avoid the circularity of an LLM-verified gold
  set. See DECISIONS.md 2026-08-27 entries.
- **The 4 annotators then independently answered each question themselves, then
  cross-checked against Claude's suggestion.** They report Claude's answers
  matched theirs well. So the labels are genuinely human-adjudicated, not
  rubber-stamped.
- Claude then merged `llm_suggest` → the real `answer` column and dropped the 4
  `llm_*` scratch columns. `llm_rationale` folded into `notes` (prefixed
  `[llm-assist]`, plus safety/method flags). `confidence` filled from
  `llm_confidence` where blank. `pass1_auto=1` kept (provenance: machine-proposed,
  human-verified). Backup of the pre-merge sheets:
  `<scratchpad>/annotation_bak_20260831_010200/`.
- `verified` was **NOT** set by Claude — the annotators set it per row in their
  own pass. `08_merge_gold.py` gates the headline agreement + miss-rate stats on
  `verified` and currently (correctly) warns "every stat below is the
  auto-labeller measured against itself" until rows are marked.
- 20 UNAIDED rows: `answer` left blank for the annotators' own unaided search
  (~15 unique qids; some now filled with plain-language citations that need
  `08_merge_gold.py`'s `resolve()` to map — 15 showed as "unresolved" when
  building the probe set).
- `data/processed/gold_test_v1.jsonl` was regenerated by `08_merge_gold.py` as a
  side effect — treat as a **draft**, not frozen. Version it before freezing.
- `data/processed/probe_questions.jsonl` created — 552 answerable questions
  (qid, question_bn, gold_chunk_ids, gold_provision_ids, lang_tag), the fixed
  eval set. Excludes 8 unanswerable + 9 "none" + 15 unresolved.

### 2. BM25 diagnosis (done — it is NOT broken)

BM25 misses the gold provision ~93% of the time (recall@10 = 0.07 over the 552).
Checked: `dhara.retrievers.bm25` + `dhara.normalize.content_tokens` apply
symmetric aggressive normalization, tokenizer works, digits normalized both
sides. The miss is real and decomposes:

| provision-level recall@10 | BM25 | e5-base zero-shot |
|---|---|---|
| overall (552) | 0.07 | 0.15 |
| English-gold (333 = 60%) | 0.00 | 0.05 |
| Bengali-gold (108) | 0.27 | 0.36 |
| mixed (111) | 0.12 | 0.27 |

- **Cross-language wall:** 60% of gold answers are English-only Acts (MFLO 1961,
  DMMA 1939, Succession Act, Penal Code, CrPC, Contract Act, TPA, Evidence Act —
  the un-replaced colonial/early-independence private law). A Bangla question
  cannot lexically touch English statute text.
- **Bangla→Bangla register gap:** even Bengali-gold questions miss 73% at rank
  20. Corpus has তালাক in 3 chunks, যৌতুক in 8, দেনমোহর in 3, ওয়ারিশ in 1 —
  citizen vocabulary barely appears in statute text.
- Secondary: ~10 near-identical local-government Acts share "সম্পত্তি সম্পর্কিত
  ঘোষণা" boilerplate that wins any family-property question.

Both gaps are the project thesis, now measured separately. Report the miss rate.

### 3. Dense retriever switched: multilingual-e5-base → BAAI/bge-m3 (decided)

e5-base failed the load-bearing assumption that a multilingual encoder reaches
the English Acts (5% English-gold recall@10). BGE-m3 zero-shot, on an 8.3k probe
pool (all gold-provision chunks + 8,000 distractors):

| recall@10 (8.3k pool) | e5-base | BGE-m3 |
|---|---|---|
| overall | 0.22 | **0.48** |
| English-gold | 0.08 | **0.35** |
| Bengali-gold | 0.53 | **0.72** |
| mixed | 0.34 | **0.61** |

BGE-m3 quadruples English-gold recall. Cross-language gap is bridgeable; e5-base
was too weak. No corpus translation needed as a prerequisite. See DECISIONS.md
2026-08-28.

Cost: BGE-m3 is 568M params / 1024-dim (e5-base: 278M / 768). Corpus embed needs
a T4 (~10 min), not the CPU box (~5 hrs). Fine-tune with LoRA or small batch.

## Deliverable produced this session

`notebooks/colab_bge_m3_eval.ipynb` — Colab/Kaggle T4 notebook. Upload
`data/processed/corpus_v1.jsonl.gz` (~12 MB — the plain 138 MB file truncates in
Colab's upload widget) + `probe_questions.jsonl`, Run all (~15 min). The loader
handles `.gz` and asserts 39484/552 so a bad upload fails loudly. Full-corpus
BGE-m3 embed + BM25 + RRF hybrid, provision-level recall@{1,5,10,20,50,100}
overall and by gold language. Saves `bge_m3_corpus.npy` (39484×1024 fp16, the
production dense index), `bge_m3_corpus_ids.json`, `bge_m3_query.npy`,
`eval_results.json`. The BM25 in the notebook is an inlined approximation of
`content_tokens` — rerun with repo code for the paper number.

## Update — 2026-08-31: steps 1-3 are DONE

The T4 run completed and its outputs are organised into the repo:

- `models/index_bge_m3_zeroshot_v1/` — `embeddings.npy` (39484x1024 fp16),
  `chunk_ids.json`, `probe_query.npy` (552x1024), `manifest.json`,
  `colab_eval_results.json`. Gitignored (`models/`), frozen, never rebuilt in place.
- `results/runs/bge_m3_zeroshot.json` — the citable run, **with `per_query`**
  (rank, top_score, gold_score_max, top-20 chunk ids per question).
- `results/tables/bge_m3_zeroshot_recall.csv`
- `scripts/13_eval_dense_index.py` — re-derives all of it from the cached vectors
  on CPU in seconds. Reproduces the notebook to within 0.01.
- `src/dhara/metrics.py` — recall@k, MRR, paired bootstrap over `per_query`.
- `configs/models.yaml` — created; BGE-m3 is the default in
  `src/dhara/retrievers/biencoder.py` (512 tokens, no prefix).

Locked zero-shot control, full 39,484-chunk corpus, n=552, provision-level:
overall R@1 0.096 / R@5 0.239 / R@10 0.324 / R@50 0.534.
English-gold R@10 0.222, Bengali-gold 0.546, mixed 0.414.
The 8.3k-pool numbers (R@10 0.48) are retired — pool overestimate, as expected.

**New finding: equal-weight RRF loses to dense alone** (R@10 0.216 vs 0.324, on
every language slice). RRF weights by rank only, and BM25 is near-random on the
60% English-gold slice, so fusion injects noise. Dense-only is now the default
path into the reranker; weighted fusion is an ablation. CLAUDE.md architecture
line and DECISIONS.md updated.

## Update — 2026-08-31 (later): synthetic training data built

There was no training set: all 794 annotated rows are `source: mined` and the
552 clean ones are the eval set. Built one. Full reasoning in DECISIONS.md
2026-08-31 "Synthetic training data".

**Generator inverts the usual direction.** Questions are hand-authored in citizen
register first (`configs/synth_topics.yaml` 36 topics +
`configs/synth_templates_extra.yaml`), then *matched* to provisions — so a
template cannot leak a provision's vocabulary. Positives are hand-anchored
(`configs/synth_anchors.yaml`, 367 chunks, every act/section verified against
corpus_v1) because signature matching alone got ~half the sections wrong (talaq
question → MFLO s6 Polygamy instead of s7 Talaq).

**The §5.3 audit is the important output.** Real mined questions share a
**median of 0.0** content words with their gold provision (mean 0.041, p90
0.122). That is the lexical gap measured directly, no retriever involved — a
headline number for the paper. Against it:

| tier | n | median overlap | verdict |
|---|---|---|---|
| real mined (reference) | 552 | 0.000 | — |
| tier_a (anchored) | 4,585 | 0.000 | pass |
| tier_a_expansion | 1,356 | 0.000 | pass |
| tier_b (title+lexicon) | 5,244 | 0.375 | fail |
| tier_c (title only) | 12,464 | 0.500 | fail |

54,434 pairs generated, **5,941 survived (11%)**. Tier B/C are title-derived and
degenerate into title matching — the §5.3 trap, reproduced and caught. Kept in
the dataset, labelled, as an ablation.

**Ready to train:** `train_strict_v1_negatives.jsonl` (3,774 pairs, 8 hard
negatives each from ranks 5–30, no gold provision among positives),
`dev_strict_v1.jsonl` (490 pairs, 6 held-out topics). `full` variant also built
as the ablation. Split unit is the **topic**, not the chunk — the near-duplicate
axis runs along topics in this generator. Leakage assertions pass.

**Ethics gap closed:** 561 repealed/omitted chunks (not 887 — that over-count
included `রহিতকরণ ও হেফাজত` sections, which perform a repeal and are live law)
are listed in `data/processed/excluded_repealed_v1.jsonl` for index/serving to
apply. Zero are gold answers. Only 1 of 552 queries ever surfaced one in top-10.

**New code:** `src/dhara/synth.py`, `src/dhara/metrics.py`, scripts
`13_eval_dense_index` `14_generate_questions` `15_overlap_audit`
`16_split_training` `17_mine_negatives` `18_compare_runs` `19_build_exclusions`,
`notebooks/colab_bge_m3_finetune.ipynb`.

## Update — 2026-09-03: fine-tune attempt 1 collapsed, root cause fixed

Ran `notebooks/colab_bge_m3_finetune.ipynb` on a T4. It completed without error
and produced a full index, but scoring it (`13_eval_dense_index.py` +
`18_compare_runs.py`, same code as the control) showed **R@10 collapsed from
0.324 to 0.060, significant on all 24 cutoff/slice comparisons, all in the
wrong direction.** Not a null result — diagnosed as an actual training bug:
fine-tuned corpus embeddings sat at **mean cosine 0.16** against zero-shot on
the identical chunks (near-orthogonal, not "shifted by training").

Full diagnosis and fix in DECISIONS.md 2026-09-03 "Fine-tune attempt 1
collapsed". Two compounding bugs, both fixed:

1. **`scripts/17_mine_negatives.py` only excluded a candidate from being a
   hard negative if it was relevant to the *same topic*.** Never checked if it
   was a *different* topic's actual positive. 7.7% of mined negatives (2,335 /
   30,162) were literally another pair's correct answer — MNRL then pushed a
   correct embedding both toward and away from different anchors in nearby
   batches. Fixed: exclusion is now global across the whole training file.
   Re-ran for both variants (strict skips: 6,367 → 9,827; every pair still
   fills to 8 negatives).
2. **The notebook exploded each pair into 4 duplicate triplet rows** (same
   anchor+positive, 4 different negatives) at `batch_size=8`, making
   same-batch collisions routine. Fixed: one `InputExample` per pair with all
   4 negatives packed in (`texts=[anchor, positive, neg1..neg4]`), with an
   assertion that every example has the same text count (sentence-transformers
   collates by column across the whole dataset — ragged lengths silently
   break, not just look ugly).

Also while in there: narrowed LoRA `target_modules` from
`["query","key","value","dense"]` to `["query","value"]` — "dense" substring-matched
3 extra Linear layers per transformer block, way more capacity than 3,774
pairs should carry without drifting the encoder. Raised batch 8→16. Added a
**canary gate**: right after training, before the 10-minute full-corpus embed,
encode 200 sample chunks with base vs fine-tuned and abort if mean cosine <
0.5. This attempt's collapse would have been caught in under a minute instead
of after a full embed + full eval.

Attempt 1 preserved, not deleted: index at
`models/index_bge_m3_finetuned_v1_attempt1_broken/`, scored run at
`results/runs/bge_m3_finetuned_strict.json`. Recorded as a training-recipe
failure, not evidence about the thesis either way.

## Update — 2026-09-03 (still later): attempt 2 ran clean, result is a wash

Re-ran the fixed notebook. Canary: 0.8268 (healthy, no collapse). Full-corpus
cosine to zero-shot confirmed independently: mean 0.8267, range 0.696–0.973,
zero chunks below 0.5. Manifest confirms the fixes actually took this time
(`train_examples: 3774` == `train_pairs`, `target_modules: [query, value]`).

Scored (`13_eval_dense_index.py` → `bge_m3_finetuned_strict_v2`), compared
(`18_compare_runs.py`, 10,000 resamples) against the zero-shot control:

| n=552 | zero-shot | fine-tuned v2 | 95% CI on diff |
|---|---|---|---|
| R@10 (all) | 0.324 | 0.330 | within noise |
| R@1 (english) | 0.054 | 0.024 | **significant, worse** |
| R@5 (english) | 0.159 | 0.111 | **significant, worse** |
| R@50 (mixed) | 0.649 | 0.784 | **significant, better** |
| R@100 (mixed) | 0.739 | 0.847 | **significant, better** |

**No headline win.** All-slice recall is flat at every cutoff. Only 4 of 24
comparisons cleared the noise floor (uncorrected — ~1 expected by chance at
this count), and they point opposite directions: English got worse at the TOP
of the ranked list (R@1/R@5 — what a top-3 UI shows), mixed-language got
better DEEP in the list (R@50/R@100 — below what any UI surfaces). Net visible
effect: nothing. Quick error check: 54 English questions zero-shot ranked
correctly at ≤10 (many at rank 1) got demoted by the fine-tune, some sharply
(rank 1 → 18). Not root-caused yet.

Leading hypothesis, not confirmed: **413 distinct training questions is thin**
(flagged before this run happened). The encoder moved substantially (0.83
cosine is real movement, not a no-op) so it learned *something* — probably
just not enough distinct phrasing to teach new cross-lingual associations for
English specifically.

Full writeup: DECISIONS.md 2026-09-03 "Fine-tune attempt 2: no collapse, no
headline win". Both attempts preserved: `models/index_bge_m3_finetuned_v1/`
(v2, current), `models/index_bge_m3_finetuned_v1_attempt1_broken/` (v1).

## Immediate next steps

1. **Root-cause the English R@1/R@5 regression** before doing anything else —
   pull the 54 demoted questions' `per_query` records from both
   `bge_m3_finetuned_strict_v2.json` and `bge_m3_zeroshot.json`, look for a
   shared pattern (same Act? same topic overrepresented in training? same
   provision now attracting everything?). A concentrated cause (e.g. one
   overfit topic hoovering up nearby rankings) is a very different fix than
   diffuse noise from thin training data.
2. **Widen training data diversity** — 413 distinct questions across 30
   topics is the prime suspect for "moved the space but didn't teach anything
   new." Add more phrasings to `configs/synth_templates_extra.yaml` (or new
   topics to `configs/synth_topics.yaml` + `configs/synth_anchors.yaml`),
   re-run the four-command replay (`14`→`15`→`16`→`17`), re-run the notebook
   as attempt 3. Do not skip the overlap audit (`15`) on the new batch — it's
   what keeps new templates from drifting into the tier-B/C trap.
3. Try the `full` variant as a bounded ablation (5,376 pairs, includes
   gold-overlapping positives) to see if more data alone — even
   memorization-tainted — moves the needle, as a diagnostic only. Never cite
   its number as the headline result; `strict` is the only variant that
   isolates transfer from memorization.
4. `scripts/12_zeroshot_probe.py` still builds an e5 biencoder by name in
   `build_retriever` — point it at `configs/models.yaml`.
5. Re-tune BM25 `k1`/`b` on dev against the 39,484-chunk corpus (defaults are
   on old 6k-chunk values). Does not change the cross-language decomposition.
6. Annotators: mark `verified` per row; resolve the ~15 UNAIDED
   plain-language citations; decide `none` vs `unanswerable` on the ~13
   flagged rows (load-bearing for abstention calibration).
7. **Paraphrasing** of the mined questions (register-preserving,
   copyright-safe) — independent of all the above, safe to run in parallel.
   Do NOT freeze `gold_test_v1` / `corpus_v1` until steps 4–6 settle.
8. Cross-encoder rerank (`bge-reranker-v2-m3` or fine-tuned) once the
   dense-retriever question above is settled — reranking on top of a fine-tune
   that isn't clearly better yet is premature.

## Build order (from CLAUDE.md)

Topic 0 (embeddings, Word2Vec) → Topic 5 (retrieval fine-tuning — the central
claim, done early on purpose) → Topic 3 (LM perplexity — the register-gap
measuring instrument) → Topic 4 (seq2seq register translation) → Topic 1
(classification) → Topic 2 (sequence labeling). Classification is NOT mandatory
(supervisor guidance). Honest negative results are explicitly worth more than
fudged positive ones.

## Non-negotiables (do not violate under time pressure)

- Gold set isolation: `gold_test_v1.jsonl` touched once, at the end. Never for
  debugging. `09_run_eval.py` asserts no leakage.
- Split by chunk, not by question (near-duplicate synthetic questions leak).
- Freeze then version: `corpus_v1.jsonl` never changes after Week 3; a fix
  becomes `corpus_v2.jsonl`.
- Two normalization levels: `normalize.light()` for transformers,
  `normalize.aggressive()` for BM25/Word2Vec. Not interchangeable.
- E5 prefixes if any E5 checkpoint is used (query:/passage:). BGE-m3 dense needs
  no prefix.
- Hard negatives from ranks 5–30, not 1–4.
- No hand-typed numbers in the report — every number from a `results/runs/*.json`
  with `per_query`.
- Every rung-to-rung comparison reports a paired-bootstrap CI on the difference.
  At 200 gold questions the 95% CI on Recall@5 is ~±7 points.
- Risk tier drives presentation only, never retrieval ranking.

## Scratchpad artifacts (this session, not in repo)

`<scratchpad>` = `C:\Users\sarwa\AppData\Local\Temp\claude\f--CODE-NLP-Dhara\6bb5a90f-9060-4571-9a16-f4e954b0a223\scratchpad\`

- `corpus_e5base.npy` (39484×768) + `corpus_ids.json` — cached e5-base corpus
  embeddings (max_seq_length 160, `passage: {title} {text_bn}`)
- `bge_pool.npy`, `bge_queries.npy` — BGE-m3 on the 8.3k probe pool
- `probe_pool.json` — the 8.3k pool definition
- `answers.jsonl` — the 569 Claude answers (qid, suggest, rationale, confidence, flag)
- `annotation_bak_20260831_010200/` — pre-merge annotation sheets
- helper scripts: `bm25_check.py`, `bm25_breakdown.py`, `zeroshot_probe.py`,
  `bge_probe.py`, `pool_build.py`, `finalize_answers.py`, `merge_llm_suggest.py`,
  `make_probe_questions.py`

## Two safety-flagged gold rows (in notes)

- `prot_0112_02` — debtor contemplating suicide over loan penalties. Note says
  civil debt default is not imprisonable + crisis-support/legal-aid referral.
- `ajke_0184_01` — sexual assault + abduction + extortion of a student. HIGH-TIER
  framing, urge prompt FIR + medical exam + OCC/legal-aid referral.
