# Defense prep: every result, every shortcoming, why

Built 2026-09-19. This is a reference for defending the project to a supervisor/committee —
every number here is pulled from a `results/runs/*.json` or a direct file audit run this
session, not recalled from memory. Where a number is small-n or uncertain, that is stated,
not hidden. See `DECISIONS.md` for the dated, narrative version of how each finding was
reached; this document is the organized summary for defense.

## 1. Corpus and scope

- 39,484 chunks, 1,227 Acts, scraped from the bdlaws portal (structured HTML, no OCR).
- Scope rule: frequency in ordinary civilian life, not legal taxonomy — confirmed domains
  are family, land, labour, consumer, cybercrime, constitutional; further domains under
  expansion (`DECISIONS.md`).
- **~60% of gold questions' correct answer is an English-only Act** (no Bangla version
  exists for many Acts). This single fact drives most of the retrieval ceiling discussed
  below — it is a corpus property, not a modeling failure, and it is the reason the project
  uses a multilingual encoder instead of a Bangla-only one.
- Repealed/omitted provisions are dropped from the corpus and the dropped count is recorded
  (ethics requirement — retrieving a repealed section would be actual harm).

## 2. Data annotation and the gold set — including a real limitation just discovered

**Team:** the project is credited to two members, Sarwad and Iftiaq (`data/annotation/verify_{name}.csv`,
290 rows each after merge). This is not a single-annotator project; that earlier assumption
in this session's chat was wrong and is corrected here. Two other early contributors'
verification work was folded into these two names (`scripts/75_merge_annotator_identities.py`,
`DECISIONS.md` 2026-09-19) — reattributed, not dropped, preserving the double-annotation
structure the reliability numbers below depend on.

**Process:** each annotator verified questions independently (`verify_*.csv`), with small
deliberate cross-annotation overlaps for reliability checking, plus a dedicated
double-annotation-with-adjudication round (`adjudication_round2.csv`) where two annotators
independently proposed an answer and a `final_verdict`/`final_answer` reconciled disagreement.

**Measured agreement (computed 2026-09-19, previously stated as "undefined" — that was
wrong; it just hadn't been computed yet). Numbers below exclude pairs that became
self-comparisons after the identity merge (same person "vs themselves"), reported instead
of silently kept:**

| comparison | n | verdict agreement | exact answer agreement |
|---|---|---|---|
| adjudication_round2 | 12 | 8/12 (66.7%) | 0/12 (0%) |
| verify cross-check overlap | 14 | 13/14 (92.9%) | not separately re-measured post-merge |

**This is a real limitation, stated plainly:** raw pre-adjudication agreement on the
specific correct provision is poor. This is exactly why the adjudication step exists — a
third pass reconciles disagreement into `final_answer`, and only the adjudicated result
enters `gold_verified_v2.jsonl`. The honest framing for defense: *initial annotator
agreement on exact provision was low, which is itself informative about how ambiguous many
real citizen questions are (a scenario can plausibly cite multiple provisions across
different Acts); the project's response was to add a mandatory adjudication step rather than
trust single-annotator judgments, and to measure and disclose the disagreement rate rather
than hide it.* n=12-14 is below CLAUDE.md's own 50-question double-annotation target — this
number is a first measurement, not a final reliability figure, and should be expanded before
the paper's final reliability claim.

**Total pool:** 552 gold-verified rows, 480 answerable with a labelled positive chunk, 72
deliberately unanswerable (for abstention-threshold calibration). Confirmed: every row across
all four `verify_*.csv` files and `adjudication_round2.csv` is already merged into
`gold_verified_v2.jsonl` — there is no unmerged annotation sitting idle. Growing the gold set
beyond 480 answerable rows requires new annotation sessions, not a data-pipeline fix.

**Mined-question ethics:** newspaper legal-advice questions are paraphrased before entering
the public dataset (`paraphrase_*.csv`), verbatim text stays local and git-ignored, PII
scrubbing tracked per row.

## 3. Train/dev/test split — is it standard size?

**As a proportion: yes, close to standard (80/10/10).** Of the 480 real annotated rows:
train 408 (85%), dev 36 (7.5%), test 36 (7.5%).

**As an absolute count: no, thin — but this is a data-volume ceiling, not a methodology
bug.** 36 dev / 36 test rows is small next to typical IR benchmark eval sets (often
hundreds+). The reason: **the entire project has only 480 real human-verified questions,
ever, full stop** — confirmed above, nothing more exists to draw from right now. Comparing
dev/test (36 each) against the full *training pool* (4,035 rows, which includes 2,685
synthetic approved pairs + 942 authored questions) is an apples-to-oranges comparison: dev
and test are drawn only from the real human-verified population by design (an eval set built
from synthetic data would not measure what the project claims to measure), so they can only
ever be as large as that population allows.

**The actual fix, if pursued:** more real annotation sessions from the existing 4-person
team, growing the 480-row pool. That is a project-scheduling decision, not something fixable
by re-running a script. What *was* fixable and got fixed this session: the *split
methodology* — an earlier version let a single 386-row provision-connected component (real
citizens ask many correlated variants of popular questions) dominate whichever split it
landed in, producing an unstable, misleading dev-vs-test gap. Fixed by capping which cluster
sizes can enter dev/test and round-robining across domains (`scripts/68_build_human_aware_retrieval_splits_v4.py`,
`DECISIONS.md` 2026-09-19). Post-fix: max same-Act cluster in dev/test dropped from 5 to 4,
distinct-Act coverage improved (dev 19→21, test 23→24 Acts), same target sizes.

**Consequence for reading any dev/test number:** at n=36, one question is worth ~2.8
percentage points. Every reported delta needs a paired bootstrap CI on `per_query`
(`scripts/18_compare_runs.py`) before being claimed as real, not read at face value.

## 4. Preprocessing and embedding

- Two normalization levels used correctly and separately: `normalize.light()` (NFC, ZWNJ
  strip, whitespace) feeds transformers; `normalize.aggressive()` (+ZWJ strip, digit
  normalization, lowercasing) feeds BM25/Word2Vec/TF-IDF. Never mixed.
- Two text fields: `text_raw` shown to users, `text_bn` fed to models — never displayed
  normalized text to a reader.
- Retriever checkpoint: `BAAI/bge-m3`, multilingual (needed to reach English-only Acts a
  Bangla-only model cannot represent). Classification checkpoint: `csebuetnlp/banglabert`.
- Zero-shot dense control locked before any fine-tuning ran (R@10 = 0.324 on the original
  480-question mixed pool) — this is what separates "transformers work at all" from "our
  fine-tuning specifically works," and it has never been silently rebuilt.

## 5. Why retrieval is hard here — the actual thesis, not a bug list

- **Lexical gap is median-zero**: real citizen questions share **zero** content words with
  their answer provision on median (measured, session memory `lexical-gap-is-median-zero`).
  This is the premise the whole project measures, not a defect.
- **BM25 (lexical floor): R@10 = 0.067, 0% on English-only gold.** A pure lexical retriever
  cannot bridge Bangla-question-to-English-Act at all. This number is *supposed* to be bad —
  it is the floor the neural approach is measured against.
- **Cross-lingual wall, quantified on the current test set (n=36):** Bengali-target R@100 =
  0.708, English-target R@100 = 0.417. That gap is the retrieval *ceiling* before any
  reranking or fine-tuning touches it, and no amount of Bangla-question fine-tuning teaches
  the model English legal vocabulary it was never exposed to.
- **Word2Vec neighbours are near-nonsense at this corpus size** (R@10 = 0.029 overall,
  0.0033 on English) — expected at 200k-400k words per CLAUDE.md's own documented trap,
  reported as a corpus-size finding, not hidden.

## 6. Fine-tuning attempts — full history, each failure with its specific cause

| attempt | result | root cause |
|---|---|---|
| Zero-shot BGE-m3 (control) | R@10 0.324 (480-pool) | — locked baseline |
| Equal-weight RRF (BM25+dense) | R@10 0.216 | BM25 returns 0% relevant on English-gold; half of every fused score is noise |
| Fine-tune attempt 1 | R@10 0.060 | embedding collapse — two bugs, both fixed |
| Fine-tune attempt 2 | R@10 0.330 (flat) | query-space hubness, root-caused |
| Fine-tune attempt 3 | R@10 ~0.35 | beat zero-shot, largest gain on English-gold slice |
| Fine-tune "attempt 4" (2026-09-18, v5 data) | Dev R@10 0.325→0.342, R@10 flat, R@100 *dropped* | **stale train file**: notebook loaded `train_retrieval_v4.jsonl` (zero hard negatives), not the newly-mined v5 pool — the mining pipeline had just been built but was never wired in |
| Same day, negatives wired in | Still flat | **52-row human-training floor**: `66_merge_authored_into_pool.py`'s provision-level exclusion policy silently dropped 199 of 251 real human rows that an earlier split (`51_build_human_aware_retrieval_splits_v3.py`) had deliberately allowed to overlap dev/test provisions — a policy mismatch between two scripts, not a modeling failure |
| Same day, split rebuilt (v4 splits) | Dev R@10 0.325→0.368 (+4.3pt), test R@10 flat at exact tie (0.0000 delta) but R@1/MRR/nDCG all moved up together | test tie is a small-n (n=35-36) boundary artifact, not evidence of failure — other ranking metrics moved in the same direction on the same set |
| 2026-09-19, v6 pool (408 real rows, split-clumping fixed) | **Test R@10 0.472→0.500 (+2.8pt); English-target test R@10 0.333→0.417 (+8.3pt)** | real human training signal grew from 52 to 408 rows by fixing the split/merge policy conflict above; this is the first attempt where the English-target slice — the actual hard part of the thesis — moved |

**Why fine-tuning "failed" three times before working, specifically:** not because the
approach is wrong, but because of a chain of *data-pipeline* bugs, each with a name: a stale
file reference, a policy mismatch between two scripts written at different times, and (for
attempts 1-2) embedding collapse from cross-topic negative contamination and query-space
hubness (both documented and fixed in `DECISIONS.md` 2026-09-03/04). None of these are "we
don't know why it didn't work" — every one has a diagnosed, fixed, documented mechanism.
That is the defensible claim: not "it worked the first time," but "every failure was
root-caused, not shrugged at."

**Why it hasn't been "fixed" to a large number, and won't be by more of the same:** the
remaining gap is structural (the cross-lingual wall in §5), not a bug. R@100 for the
fine-tuned model is *identical* before and after fine-tuning (0.6111 on test) — the LoRA
adapter (0.28% of parameters, deliberately small for T4 memory limits) re-ranks within the
existing candidate pool, it does not expand which documents are reachable. Closing that
requires either a reranker with real headroom to work with, or fundamentally more/different
training signal than 408 rows can provide.

## 7. Reranker — a real regression, found and root-caused same session

**What happened:** a zero-shot (untrained) `BAAI/bge-reranker-v2-m3` on an earlier, larger
probe pool moved R@1 significantly (+3.6pt) but not R@10 (+1.8pt, not significant). A
*trained* reranker (on this session's mined pairs) was then evaluated and made things
**worse** — test R@10 dropped from 0.500 to 0.361, R@1 nearly halved.

**Root cause, found the same day:** `retrieval_metrics()` deduped stage-1 candidates to one
entry per provision but discarded *which specific chunk* earned that rank. The reranker
evaluation code then reconstructed a chunk per provision by picking whichever chunk appeared
first in `corpus_v1.jsonl`'s file order — unrelated to relevance. The reranker was *trained*
on the specific labelled `positive_chunk_ids[0]` for each question but *shown a different,
arbitrary chunk* for the same provision at evaluation time. A real train/eval mismatch, not
evidence the reranking approach itself is unsound.

**Fixed:** `retrieval_metrics()` now tracks the actual best-scoring chunk per provision and
passes it through; reranker evaluation uses that same chunk instead of the file-order
substitute. Not yet re-run to confirm the fix restores expected behavior — that is the
immediate next action, not yet a claimed result.

**Real headroom that motivates fixing this rather than abandoning it:** on a larger 552-probe
pool, 30.3% of questions have their gold provision sitting in ranks 11-100 — invisible to
top-10 without a second stage. A perfect reranker over that pool would take R@10 from 0.324
to 0.627. That is the ceiling the reranker is chasing; the zero-shot reranker realized only a
small fraction of it, which is exactly why training one (correctly, this time) is worth
doing.

## 8. Classification (Topic 1) — done, honest, not a retrieval lever

**Not mandatory** (supervisor's written guidance, `DECISIONS.md` 2026-08-23: "you are not
bound to do classification works, it's an open ended project"), done anyway as cheap
syllabus coverage.

**14 domains, `other` at 66% of all 552 gold questions** (363/552) — a real corpus-composition
imbalance, not a labeling bug. 7 of 14 domains (`constitutional`, `civil_registration`,
`cybercrime`, `local_government`, `money_recovery`, `road_transport`, `tenancy`) have 1-6
examples **total across the entire pool**, not just one split.

| model | v2 macro-F1 | v3 (rebalanced) | v3 merged rare domains |
|---|---|---|---|
| Naive Bayes | 0.080 | 0.073 (regression) | 0.100 |
| Logistic Regression | 0.203 | 0.203 (unchanged — already balanced) | **0.269 (best)** |
| Vanilla RNN | 0.096 | 0.071 (regression) | 0.131 |
| Stacked BiLSTM | 0.093 | 0.068 (regression) | 0.162 |

**Two rebalancing attempts genuinely backfired, reported as such, not hidden:**
1. Forcing Naive Bayes to a uniform class prior made it slightly worse (0.080→0.073) — NB's
   problem was never the prior, it's that likelihood estimates from 1-6 examples are
   inherently noisy; the learned prior was accidentally stabilizing.
2. Class-weighted cross-entropy on both RNNs made them **much** worse in isolation (accuracy
   cratered to 20-31%) — classes with 1-2 examples get ~25x loss weight, and 8 epochs of
   Adam at lr=1e-3 on 352 rows is not stable against gradient spikes that large.

**What actually worked:** merging the 7 near-empty domains into one `rare_other` bucket. This
is the only intervention that helped every model, and combined with the (otherwise-harmful)
RNN weighting, netted ahead of the original baseline for the RNNs too.

**Best model: Logistic Regression on merged labels, macro-F1 0.269** — confirms the
generative-vs-discriminative comparison the proposal wants (discriminative wins under this
imbalance), and confirms the ceiling was data volume, not fixable by any reweighting
mathematics.

**Deliberately not wired into retrieval**: CLAUDE.md bans domain-filtering retrieval
candidates by default — a wrong prediction (55% accuracy at best on the hardest classes)
would silently destroy retrieval. Reported as its own, separate result.

## 9. Other syllabus topics — honest one-line status each

| topic | artifact | result | status |
|---|---|---|---|
> **2026-09-19: Topic 4 (Seq2Seq), the from-scratch BiLSTM dual-encoder retriever ablation, and
> unsupervised clustering were dropped from the project's reported scope** — none produced a
> result worth keeping, and their code/artifacts were removed from the repo. The tagger row
> below is also stale: `results/runs/tagger_v2.json`'s 97.5% "accuracy" was replaced by
> `tagger_v3.json`'s per-entity F1 (0.857 ACT / 0.627 PARTY / 0.000 SECTION_NO / 0.000
> LEGAL_TERM, macro 0.371), which is the real, more informative number. See
> `docs/PROJECT_REPORT.md` §4 for the current, authoritative version of this table.

| topic | artifact | result | status |
|---|---|---|---|
| Topic 0 (embeddings) | zero-shot BGE-m3 control | R@10 0.324 locked | done, never silently rebuilt |
| Topic 2 (sequence tagger) | weak-label BiLSTM tagger | entity macro-F1 0.371 (ACT 0.857, PARTY 0.627, SECTION_NO/LEGAL_TERM 0.000) | superseded metric, see note above; still measures agreement with its own weak labels, not human ground truth |
| Topic 3 (perplexity LM) | register-gap instrument | colloquial mean PPL 138.9 (median 89.5) vs formal mean 113.8 (median 100.3) | nuanced, not a clean win: colloquial's *median* question is actually slightly easier than formal's, but a heavy tail of very hard colloquial questions (max PPL 1575 vs 358) pulls the mean up — the register gap is driven by outliers, not a uniform shift |
| Word2Vec retriever (Topic 0 baseline) | from-scratch word vectors as a retriever | R@10 0.029 overall | expected at this corpus size (200-400k words), documented corpus-size finding per CLAUDE.md's own known trap |

## 10. The three real ceilings, for the limitations section

1. **Cross-lingual retrieval wall** — measured via the Bengali/English R@100 gap (0.708 vs
   0.417 on current test), not fixable by more Bangla-question training data; the actual
   research finding of the project, not a defect to apologize for.
2. **Real annotated data volume capped at 480 questions**, confirmed exhausted (nothing
   unmerged remains). Limits both training signal and eval precision; growing it needs new
   annotation sessions, a scheduling decision.
3. **Small eval set (36-71 rows) → wide CI.** Every point-estimate delta needs
   `scripts/18_compare_runs.py`'s paired bootstrap before being claimed as real. Several
   numbers in this document (the test R@10 gains) are directionally consistent across
   multiple metrics but not yet CI-confirmed — stated as such, not oversold.

None of these three are "we didn't try hard enough." Two are corpus/data properties outside
this session's control; the third is a measurement-rigor discipline already being followed
(that is what CI-gating every claim *is*).
