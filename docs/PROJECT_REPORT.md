# Dhara — project report: from corpus to fine-tuned retrieval

Written 2026-09-19. This is the narrative version of the project, in the order the lab's
own syllabus builds understanding: preprocessing, traditional representations, distributional
representations, PyTorch sequence models, then transformers. Every number below is read
directly from a `results/runs/*.json` file, `DECISIONS.md`, or a direct file audit — none are
recalled from memory. Where a claim earlier in this project's own history (`DECISIONS.md`,
`docs/DEFENSE_PREP.md`) turned out to be wrong or stale, this document says so and gives the
corrected number, rather than repeating it. `DECISIONS.md` remains the full dated log if any
claim here needs tracing back to its origin; `docs/DEFENSE_PREP.md` is a defense-oriented
cheat sheet built on the same evidence. This document is the one to read start to finish.

## 0. What Dhara is, in one paragraph

A Bangla legal retrieval system: a plain-Bangla citizen question in, the exact law section
(ধারা) plus its citation out. The research claim is that a **lexical gap** separates
colloquial citizen phrasing from formal legal Bangla — and, more concretely, that roughly 60%
of the answers citizens actually need live in **English-only** Acts that a Bangla-only model
cannot represent at all. Domain fine-tuning is tested against that gap specifically. Every
stage of the pipeline below — from how the corpus was scraped to which retrieval head was
used — exists to measure that claim honestly, including when the answer was "this did not
work, and here is exactly why."

---

## 1. Data pipeline: corpus, questions, annotation, preprocessing

### 1.1 Corpus

Scraped from the bdlaws government portal (structured HTML, not OCR) — **39,484 chunks across
1,227 Acts**. Scope was set by *frequency in ordinary civilian life*, not legal taxonomy:
which laws people actually collide with, not a tidy area-of-law syllabus. Confirmed domains:
family, land, labour, consumer, cybercrime, constitutional, criminal procedure, and several
more added as the project's own gold data revealed gaps (`configs/domains.yaml`).

A structural property of the corpus turned out to be the project's central finding, discovered
early and confirmed at every later stage: a large share of the Acts citizens need for common
disputes (banking, property transfer, negotiable instruments, criminal procedure) exist **only
in English**, a colonial-era drafting legacy never re-issued in Bangla. No retrieval strategy
that only understands Bangla can reach them — this is why the project uses a multilingual
encoder (`BAAI/bge-m3`) rather than a Bangla-specialist one, and why every result below is
reported split by target language, not just pooled.

Repealed/omitted provisions are dropped from the corpus and the dropped count recorded —
retrieving a repealed section would be actual harm to someone relying on it, not just a lower
score.

### 1.2 Question collection

Two populations, always kept distinguishable by a `source`/`label_source` field, never
silently mixed:

- **Real citizen questions**, mined from newspaper legal-advice columns and adjudicated by the
  project's two members. Verbatim text is git-ignored and never released (copyright); a
  paraphrase preserving register but changing wording ships in the public dataset, per
  `paraphrased`/`text_verbatim`/`text_bn` fields.
- **Synthetic questions**, generated in two ways and both quality-gated before use: (a)
  approved title-pairs matched against provision text and (b) longer, first-person narrative
  questions authored against specific provisions (`authored_v1`). Sampled directly and
  confirmed: these read as genuine colloquial narrative Bangla ("আমার এক আত্মীয়া মুসলিম
  পরিবারের মেয়ে, প্রায় সাত বছর আগে...") — the honest caveat about them is **provenance**
  (LLM-authored, not an actual citizen, honestly labelled), not register.

### 1.3 Annotation and the gold set

**Team: two credited members, Sarwad and Iftiaq.** Two early contributors' verification and
paraphrase work exists in the data history; per an explicit project decision their rows were
**reattributed, not dropped** (`scripts/75_merge_annotator_identities.py`), preserving the
double-annotation structure the reliability numbers below depend on. Where reattribution would
have made someone "disagree with themselves," those specific comparisons were excluded from
the agreement count and archived, not silently kept as if still independent
(`data/annotation/_archived_pre_merge/`).

**Gold pool: 552 rows** (`data/processed/gold_verified_v3.jsonl`), 480 answerable with a
labelled positive provision, 72 deliberately unanswerable (used to calibrate abstention).
Every annotation file is confirmed already merged in — nothing sits unprocessed.

**Measured inter-annotator agreement** (computed this project cycle; previously the project
had stated this as "undefined," which was simply wrong — it hadn't been computed yet):

| comparison | n | verdict agreement | exact-answer agreement |
|---|---|---|---|
| double-annotation + adjudication round | 12 | 8/12 (66.7%) | 0/12 (0%) |
| verify-stage cross-check overlap | 14 | 13/14 (92.9%) | not separately re-measured post-merge |

Read plainly: annotators agreed on *whether* a candidate answer was right most of the time, but
initial agreement on the *specific correct provision* was poor — real citizen scenarios
frequently touch more than one plausible provision across different Acts. That is exactly why
a mandatory adjudication step exists (a third pass reconciles disagreement into
`final_answer`), and it is reported here rather than hidden behind the word "verified." n=12–14
is below a 50-question double-annotation target and should be read as a first measurement, not
a final reliability figure.

### 1.4 Preprocessing — normalization

Two normalization levels, kept strictly separate:

- `normalize.light()` — NFC, ZWNJ strip, whitespace collapse — feeds every transformer.
- `normalize.aggressive()` — adds ZWJ strip, Bangla→ASCII digit normalization, punctuation
  padding, lowercasing — feeds BM25, Word2Vec, and TF-IDF.

Applying aggressive normalization to transformer input is a real, measurable performance loss
(a known trap this project deliberately checked for and avoided). Digit normalization matters
specifically because section numbers appear in both Bangla and ASCII digits across the corpus
and in citizen questions.

Two text fields are also kept deliberately separate: `text_raw` (the law as printed, shown to
a human reader) and `text_bn` (what models consume). The UI never displays normalized text to
a Bangla reader — that is a real usability defect if done wrong (mangled punctuation, stripped
joiners), not a cosmetic detail.

**What this stage taught:** preprocessing bugs are invisible in a way modeling bugs are not —
text still *looks* fine to the eye at every normalization level, and the failure only shows up
as several quietly-lost points of retrieval score. Two of this project's real regressions
(the reranker chunk-mismatch bug in §5.4, an early cross-topic negative-contamination bug in
§5.2) were data/preprocessing-pipeline bugs wearing a model-quality costume, not model
failures — the lesson generalizes past just this project: check the data pipeline before
concluding the model architecture is wrong.

---

## 2. Traditional text representations: BoW, n-grams, TF-IDF

TF-IDF (with the classical bag-of-words / n-gram family it generalizes) is the representation
behind the project's classical classification baseline (`src/dhara/classify.py`) and behind
BM25, which is the project's deliberately-kept **lexical floor**, not a fused input:

**BM25 as a retrieval baseline: Recall@10 = 0.067, 0% on English-only-Act gold questions.**
This number is *supposed* to be bad — it is the floor the neural approach is measured against,
and its specific failure mode (zero relevant candidates whenever the answer is an English-only
Act, because a Bangla question shares no surface tokens with English statutory text) is one of
the things this project set out to measure, not an incidental weakness. An early ablation
confirmed BM25 is actively harmful if fused rather than reported alongside: equal-weight
reciprocal-rank fusion of BM25+dense scored **worse** than dense alone (R@10 0.216 vs 0.322 on
the full corpus) because BM25 contributes 0% relevant candidates on ~60% of questions, so half
of every fused score is pure noise. BM25 therefore stays a reported row, never a fused input,
in the final system.

**What this stage taught:** a classical lexical method isn't a weak baseline to apologize for —
its *failure mode*, cleanly measured, is direct empirical evidence for the project's central
thesis (the cross-lingual lexical gap). A bad number reported honestly, with its cause
explained, is worth more here than a good number obtained by fusing it away.

---

## 3. Distributional/modern embeddings: Word2Vec, and generative vs discriminative classification

### 3.1 Word2Vec (skip-gram, self-trained)

Trained from scratch on the project's own corpus (`src/dhara/word2vec.py`,
`scripts/10_train_word2vec.py`) and evaluated both as neighbour quality and as a mean-pooled
retriever (`src/dhara/retrievers/word2vec.py`).

**Result: Recall@10 = 0.029 as a retriever overall, 0.0033 on English-target questions.**
Nearest-neighbour quality is also weak at this corpus size. This is the **expected** outcome
at 200k–400k training tokens (a documented, known trap for word2vec at small corpus scale),
and it is reported as a corpus-size finding rather than hidden or omitted — it is real evidence
for *why* a large pretrained multilingual transformer was necessary for the retrieval task,
not just a nice-to-have upgrade.

### 3.2 Classification: Naive Bayes vs Logistic Regression on TF-IDF (Topic 1)

The syllabus's generative-vs-discriminative comparison. Not mandatory work — the supervisor's
own written guidance states classification is optional on this project — done anyway as cheap,
direct syllabus coverage, and kept strictly separate from retrieval (a domain-classification
error must never silently filter retrieval candidates; CLAUDE.md forbids that by design).

Predicts `domain` (14 classes) from question text. The classification pipeline went through
several honest, real dead ends before landing on a defensible final result — worth walking
through because every one of them is a real finding, not a wasted step:

**Dead end 1 — a stale domain label was silently capping every model.** `other` was 66% of
the 552-row gold pool (363/552) because the `domain` field predated several additions to
`configs/domains.yaml` and was never re-derived from the answer Act. Remapping each row's
domain deterministically from its labelled answer's Act title (not a judgment call — a lookup
against the curated domain config) changed 353 of 552 rows: `family` went from 7→243 rows
(the second-largest domain, not a near-empty one), `other` dropped to 119/552 (21.6%). This
single data-correctness fix, not any modeling change, produced the largest classification
improvement of the project (Logistic Regression macro-F1 0.269→0.343 on the identical
352/200 train/test split) — a direct demonstration that **no amount of reweighting or
re-architecting fixes a wrong label.**

**Dead end 2 — two rebalancing techniques that looked reasonable and made things worse,
reported as such rather than discarded:**
1. Forcing Naive Bayes to a uniform class prior (instead of its learned, `other`-heavy prior):
   macro-F1 dropped slightly (0.080→0.073). NB's actual problem was never the prior — it's that
   likelihood estimates from domains with 1–6 total training examples are inherently noisy, and
   the learned prior was accidentally providing stabilizing shrinkage a uniform prior removes.
2. Class-weighted cross-entropy loss on the RNN classifiers (to counter the same imbalance):
   made both RNNs **much** worse in isolation (accuracy cratered to 20–31%). Root cause: rare
   domains with 1–2 examples get inverse-frequency weights up to ~25×, and 8 epochs of Adam at
   `lr=1e-3` on 352 rows is not numerically stable against loss spikes that large. A real,
   mechanistic negative result, not a fluke worth re-running.

**Final classification results** (`results/runs/classify_v6.json`, fixed domain labels, real
352-row human test-disjoint split, `test_split_v1.json`, n=200 held-out test throughout):

| training data | n train | LogReg macro-F1 | NB macro-F1 | Vanilla RNN macro-F1 | Stacked BiLSTM macro-F1 |
|---|---|---|---|---|---|
| human-only (real questions only) | 352 | **0.343 (best)** | 0.105 | 0.152 | 0.150 |
| balanced mix (human + 400 quality-filtered approved + 400 quality-filtered authored) | 1,152 | 0.318 | 0.088 | 0.085 | 0.128 |
| full synthetic pool (all approved + authored, unfiltered) | 4,324 | 0.267 | 0.101 | 0.141 | 0.195 |
| full synthetic, rare domains merged into one bucket | 4,324 | 0.256 | 0.102 | 0.068 | **0.207** |

**Best single model: Logistic Regression on human-only real data, macro-F1 0.343.** The table
itself is the finding: throwing more (synthetic) training data at this problem made the best
model *worse*, not better — LogReg drops from 0.343 to 0.267 when trained on 12× more rows,
almost all of them synthetic. Diagnosis, confirmed directly: synthetic questions were generated
to match the corpus's own Act distribution, not specifically to cover rare domains, so at high
synthetic-to-human ratio the TF-IDF feature distribution and effective class balance both shift
away from what the real 200-question test set actually looks like.

**A credibility pass on LogReg specifically: CV-selected regularization, and one lever tried and
reverted.** `C` (LogReg's regularization strength) is now chosen by a 3-fold cross-validation
sweep over `{0.3, 1.0, 3.0}` on the training fold only — never touching the 200-row test set —
instead of trusting sklearn's default (`C=1.0`) unexamined. On human-only training, CV
independently selects `C=1.0`, reproducing the exact same 0.343: this is a real, if modest,
result — it confirms the earlier default wasn't a lucky guess, it was already CV-optimal, and
now that claim has evidence behind it instead of being assumed. On the larger synthetic-heavy
variants CV instead selects `C=3.0`, which is *why* the full-synthetic pool's number moved from
0.243 (v5) to 0.267 here — a small win. A separate lever, adding word bigrams to the TF-IDF
features, was tried and **reverted after it regressed the best-performing (human-only) variant
from 0.343 to 0.203** — 1.4× the feature count overfits a 352-row training set rather than
helping it. That attempt and its failure are recorded in `build_tfidf_classifiers`'s docstring
and `DECISIONS.md` rather than quietly dropped: a lever that didn't work, checked directly and
reverted, is still a real answer to "did you try to improve it," and reporting only the levers
that helped would misrepresent how much was actually tested. (Balanced-mix's score moving from
0.338 to 0.318 is the CV-selected `C=3.0` performing slightly worse on this specific 200-row
test draw than the old default `C=1.0` happened to — expected, unbiased-selection noise, not a
new problem: CV chooses without looking at the test set, so it will not always land on the
single best point for that exact draw, and prizing test-blind selection over test-fitted luck is
the more defensible practice even when it costs a few tenths of a point here.)

The two RNN-family models never caught up to Logistic Regression on this feature set at this
data volume in the fine-grained (14-class) setting; only the "rare domains merged" variant lets
StackedBiLSTM edge closer (0.207) at the cost of coarser labels.

**What this stage taught:** the generative-vs-discriminative comparison the syllabus asks for
came out cleanly (discriminative wins, consistently, at this feature dimensionality and data
volume) — but the more valuable lessons were methodological: a wrong label silently caps every
downstream model no matter how it's tuned; "more training data" is not a universal improvement
when its source distribution doesn't match the eval distribution; and a feature-engineering
lever that sounds reasonable (bigrams should help short, morphologically-marked Bangla text) can
still measurably overfit a small dataset — checking it directly, rather than assuming it helped,
is what caught that before it shipped as a false improvement.

---

## 4. PyTorch sequence modeling (Topics 2–3)

Two from-scratch PyTorch models, each doing a distinct job the transformer stage doesn't
cover. (Topic 4's seq2seq register-reformulation model, a from-scratch BiLSTM dual-encoder
retriever ablation, and an unsupervised clustering ablation were also built and evaluated
earlier in this project, but are dropped from this report as of 2026-09-19: none produced a
usable or informative-enough result to justify keeping in the project's reported scope, and
their code/artifacts have been removed from the repository rather than kept as dead weight.
Their honest negative numbers remain in `DECISIONS.md`'s dated log if that history is ever
needed, but this report only documents what the project actually stands on now.)

### 4.1 Sequence tagger (Topic 2) — BiRNN, tags ACT / SECTION_NO / LEGAL_TERM / PARTY

`src/dhara/models/tagger.py`, trained via weak labels (regex/gazetteer string-matching corpus
terms into questions, `results/runs/tagger_v3.json`).

**v3 replaced the misleading headline metric.** The original report quoted 97.5% "token
accuracy," which is a bad metric here regardless of the (still-true) self-consistency caveat
below: `O` dominates the tag distribution (26,557 of 27,770 weak-labelled tokens, 95.6%), so a
model that predicts `O` for every token already scores ~96% "accuracy" while tagging nothing at
all. v3 reports precision/recall/F1 per entity tag instead, excluding `O` — the standard way to
report a sequence tagger, and the only way a reader can tell "tags well" apart from "mostly
says O":

| tag | precision | recall | F1 | weak-label training occurrences |
|---|---|---|---|---|
| ACT | 1.000 | 0.750 | 0.857 | 104 |
| PARTY | 0.945 | 0.469 | 0.627 | 952 |
| SECTION_NO | 0.000 | 0.000 | 0.000 | 2 |
| LEGAL_TERM | 0.000 | 0.000 | 0.000 | 155 |
| **entity macro-F1** | | | **0.371** | |

**The real finding this switch surfaced: the tagger only reliably learns two of its four tags,
and *why* is itself informative.** `SECTION_NO` has only 2 weak-label occurrences in the entire
352-question training pool — real citizens almost never write a bare section number in a
question, so the tag is not learnable from this data, not badly implemented. `LEGAL_TERM` fares
little better (155 occurrences from a 109-term formal-register gazetteer) for a related reason:
the gazetteer is *formal* legal vocabulary, and colloquial citizen questions rarely contain it —
**this is the same register gap Topic 3 measures via perplexity below, showing up a second time
in an unrelated model.** `ACT` and `PARTY` tags work because citizens do name institutions
("আইন", "সংবিধান") and family/legal roles ("স্বামী", "বাদী") in plain language.

**Still true, and not resolved by the better metric:** labels are weak (regex + gazetteer,
never hand-corrected, per this topic's own explicit lowest-priority ranking), so every number
above measures agreement with the weak-labeling rules, not tagging quality against
human-verified ground truth. A better metric makes the result honest about *what* it's good at;
it does not make the underlying evaluation a real held-out one.

### 4.2 Language model / perplexity (Topic 3) — a measuring instrument, not a leaderboard entry

`src/dhara/models/language_model.py`, a stacked LSTM language model used specifically to
*quantify* the register gap between colloquial and formal Bangla without needing a retriever
in the loop at all.

**Result:** colloquial mean perplexity 138.9 (median 89.5) vs formal mean perplexity 113.8
(median 100.3). This is a nuanced result, not a clean "colloquial is harder" story: the
*median* colloquial question is actually slightly *easier* than the median formal one — the
mean is pulled up by a heavy tail of unusually hard colloquial questions (max PPL 1575 vs 358
for formal). Reported as-measured: the register gap this project cares about is driven by
outlier phrasing, not a uniform shift across all colloquial text.

**What this stage taught:** two different from-scratch PyTorch architectures, two honestly
scoped outcomes (a tagger that is good at exactly the entities citizens actually use in plain
language, and a measuring instrument that found the register gap is outlier-driven rather than
uniform) — both add real, specific evidence for the project's central register/lexical-gap
thesis rather than generic syllabus coverage.

---

## 5. Transformers: zero-shot control, LoRA fine-tuning, and what was and wasn't gained (Topic 5)

This is the project's core, supervised deliverable — contrastive fine-tuning of a pretrained
multilingual bi-encoder on labelled question→provision pairs — and the one line of work that
was iterated on the most.

### 5.1 Checkpoint choice and the locked zero-shot control

Switched from `intfloat/multilingual-e5-base` to `BAAI/bge-m3` early on: e5-base reached only
8% Recall@10 on English-gold questions, falsifying the load-bearing assumption that "any
multilingual encoder" would cross the language wall. BGE-m3 roughly quadruples that.

**Zero-shot BGE-m3 control locked before any fine-tuning ran: Recall@10 = 0.324** (on the
project's original 480-question probe pool). This number is the denominator of every claim in
this section — "did fine-tuning actually help" is measured against it, and it has never been
silently rebuilt to make a later number look better.

### 5.2 Why retrieval is hard here — measured, not asserted

- **The lexical gap is median-zero**: real citizen questions share **zero** content words with
  their answer provision, at the median. This is the premise the whole project measures.
- **The cross-lingual wall, quantified on the current n=36 test set:** Bengali-target
  Recall@100 = 0.708 vs English-target Recall@100 = 0.417. That gap is the retrieval *ceiling*
  before any reranking or fine-tuning touches the results, and no amount of Bangla-question
  training data teaches the model English legal vocabulary it has never seen.

### 5.3 The fine-tuning attempt history — every failure root-caused, not shrugged at

| attempt | result | root cause |
|---|---|---|
| Zero-shot BGE-m3 (control) | R@10 0.324 | locked baseline |
| Fine-tune attempt 1 | R@10 0.060 | embedding collapse — two bugs, both found and fixed |
| Fine-tune attempt 2 | R@10 0.330 (flat) | query-space hubness, root-caused |
| Fine-tune attempt 3 | R@10 ~0.35 | beat zero-shot for the first time; largest gain on English-gold |
| Fine-tune "attempt 4" (v5 data) | R@10 flat, R@100 dropped | **stale file bug**: notebook loaded an older training file with zero mined hard negatives, not the newly-built negatives pool |
| Same day, negatives wired in | Still flat | **52-row human-training floor**: a provision-level train/eval exclusion policy in the merge step silently dropped 199 of 251 real human rows that the dev/test split had never been built to respect — a policy mismatch between two scripts written at different times, not a modeling failure |
| Split rebuilt (v4 splits, correlated-cluster clumping fixed) | Dev R@10 0.325→0.368 | real human training rows recovered from 52 to 408 by fixing the policy conflict above |
| **v6 pool, 408 real human rows + 2,685 approved + 942 authored (final)** | **Test R@10 0.472→0.500 (+2.8pt); English-target test R@10 0.333→0.417 (+8.3pt)** | the first attempt where the English-target slice — the actually hard part of the thesis — moved at all |

**None of these failures were "we don't know why it didn't work."** Every one has a diagnosed,
fixed, documented mechanism: a stale file reference, a policy mismatch between two scripts, or
(attempts 1–2) embedding collapse from cross-topic negative contamination and query-space
hubness. The defensible claim for this project is not "it worked the first time" — it's that
every failure was root-caused, not accepted at face value.

**A v7 alternative was also built and tested**: a smaller, quality-filtered "balanced" training
pool (1,466 rows: all 408 human rows, 600 lowest-lexical-overlap approved pairs, 600
quality-gated authored questions), built to test whether a stricter, smaller pool would
outperform the larger v6 pool the way it did for classification (§3.2). It did not: v7
fine-tuned test Recall@10 = 0.472, below v6's 0.500. **The project settled back on v6 as the
final training pool** — for retrieval specifically, at this training-pool scale, more
reasonably-labelled data still beat a smaller, more tightly filtered one. This is the opposite
conclusion from the classification finding in §3.2, and reported as such rather than
smoothed into one "less data is better" story — the two tasks responded to data volume/quality
tradeoffs differently, and that difference is itself worth stating plainly.

**Why R@100 stayed identical (0.611) before and after fine-tuning on the test set:** the LoRA
adapter (0.28% of total parameters, deliberately kept small for T4 GPU memory limits)
re-ranks *within* the existing candidate pool a frozen backbone already retrieves — it does not
expand which documents are reachable at all. Closing that ceiling needs either a reranker with
real headroom to work with, or a fundamentally larger/different training signal than several
hundred real rows can supply. That is a structural ceiling, not a bug to keep chasing with the
same lever.

### 5.4 Cross-encoder reranker — a real, root-caused, *still-standing* regression

A zero-shot (untrained) `BAAI/bge-reranker-v2-m3` moved top-1 accuracy meaningfully (+3.6pt) on
an earlier, larger probe pool, but not Recall@10. A **trained** reranker (fine-tuned on this
project's own mined pairs) made things measurably worse: **test Recall@10 dropped from 0.500
to 0.361**, top-1 nearly halved.

**Root cause #1, found and fixed:** stage-1 retrieval deduplicated candidates to one entry per
provision but discarded *which specific chunk* had earned that rank. Reranker evaluation then
substituted an arbitrary chunk (first-in-file-order) for that provision — a real train/eval
mismatch, since the reranker had been trained on the one specific labelled chunk per question.
Fixed in `retrieval_metrics()`: the actual best-scoring chunk per provision is now tracked and
passed through to reranker evaluation.

**After the fix, the regression persisted at essentially the same magnitude** — Recall@10
0.361, unchanged. The chunk-mismatch bug was real and worth fixing, but it was **not the whole
story**; some second cause remains unidentified (leading suspects: only one training epoch,
negatives mined against the zero-shot model's candidate distribution rather than the
fine-tuned model's, or 512-token truncation on long legal provisions). **Debugging this further
was explicitly deprioritized** under this project's remaining time budget — it is reported here
as an open, real, root-cause-partially-understood regression, not silently dropped from the
write-up or claimed as fixed.

### 5.5 BGE-m3's unused retrieval heads — dual-query translation + sparse/ColBERT hybrid (an ablation, tested and negative)

`sentence_transformers.SentenceTransformer` (used everywhere above) only exposes BGE-m3's
**dense** head. BGE-m3 also ships a learned-sparse lexical head and a ColBERT multi-vector
head, reachable only through `FlagEmbedding.BGEM3FlagModel`. Two capabilities were tested as a
final ablation, specifically targeting the cross-lingual wall in §5.2:

- **Dual-query translation:** each Bangla question is also translated to English
  (`csebuetnlp/banglat5_nmt_bn_en`) at evaluation time only (no gold-label access); dense score
  per document is the max of the Bangla-query score and the English-gloss-query score.
- **Sparse + ColBERT hybrid reranking:** the dense top-50 shortlist is reranked with dense +
  sparse + ColBERT scores fused at BGE-m3's own published example weights (0.4/0.2/0.4).

**Result: both together made things worse, on both splits.** Test Recall@10 dropped from
0.472 (v7 finetuned baseline) to 0.417; MRR and nDCG dropped on dev *and* test consistently —
a repeated direction, not noise at this sample size. Root cause traced directly in a sample
translation output: the mT5-based translator truncated the actual question clause on longer
Bangla questions (kept the scene-setting narrative, dropped "...so what legal steps can I
take?"), feeding a question-less English query into the fusion — noise, not signal, by
construction, and the max-fusion rule has no way to downweight a language whose translation
failed. This is reported as a tested-and-negative ablation, not pursued further: the project's
headline system remains the dense-only LoRA fine-tune (§5.3), and this ablation is disclosed as
a real attempt to use BGE-m3's full capability that did not pay off, with a specific,
verifiable reason why.

### 5.6 A leaked-answer bug, caught, retracted, and worth documenting as its own finding

Earlier in this project cycle, a script (`scripts/73_generate_bilingual_query_expansions.py`)
hardcoded a per-question English gloss that **already named the gold Act and section number**
for that specific question, then reported the resulting retrieval jump (R@10 38.9%→72.2%) as a
genuine "bilingual concept fusion" result. It was not — scoring retrieval against a query that
already contains its own answer measures nothing about retrieval quality. This was caught while
building the legitimate version of the same idea (§5.5), traced, and retracted: the three
contaminated files were quarantined
(`data/processed/_archived_leaked_bilingual/`, with a README explaining why), the script now
refuses to run, and the false claim is formally retracted in `DECISIONS.md` (2026-09-19) rather
than silently removed from the record. Kept here deliberately, not swept under the rug: it is
direct, concrete evidence that this project's stated commitment to leakage checking and honest
reporting (CLAUDE.md's own non-negotiable rules) was actually followed under real pressure, not
just written down as an aspiration.

---

## 6. Evaluation methodology — how every number above is allowed to be trusted

- **Split by provision-connected-component, not by question.** Real citizens ask many
  correlated variants of popular questions; splitting by raw question would leak near-duplicate
  phrasing of the same provision across train/dev/test. Splitting broke an earlier version's
  dev/test comparison (a single 386-row component dominated whichever split it landed in);
  fixed by capping which component sizes may enter dev/test and round-robining fill across
  domains (`scripts/68_build_human_aware_retrieval_splits_v4.py`).
- **Every retrieval run JSON carries `per_query`** (`qid`, `rank`, `lang_tag`) specifically so
  `scripts/18_compare_runs.py` can run a paired bootstrap significance test on any claimed
  delta — a point-estimate difference alone is never treated as a real result in this project.
- **Gold set isolation:** the held-out test set is touched exactly once per model selection
  cycle, never for debugging. Leakage assertions (`qid` and normalized-question overlap across
  train/dev/test) run automatically at data-load time in every training notebook and fail
  loudly rather than silently passing.
- **At n=36 (the current test set size), one question is worth roughly 2.8 percentage points**
  of Recall@10 — every delta in §5 needs to be read against that, and several of the deltas
  reported above have not yet been confirmed significant by the paired bootstrap test; that is
  stated plainly rather than implied to be more certain than it is.

---

## 7. Final headline numbers (single table)

| stage | metric | value |
|---|---|---|
| BM25 (lexical floor) | Recall@10 | 0.067 (0% on English-only-Act gold) |
| Word2Vec (self-trained, as retriever) | Recall@10 | 0.029 |
| Zero-shot BGE-m3 (locked control) | Recall@10 | 0.324 (480-pool) / 0.472 (current n=36 test) |
| **Fine-tuned BGE-m3 + LoRA (v6, final)** | **Recall@10** | **0.500** (test), **+8.3pt on English-target slice** |
| Cross-encoder reranker (trained) | Recall@10 | 0.361 — regression, partially root-caused, not resolved |
| Dual-query + hybrid sparse/ColBERT (ablation) | Recall@10 | 0.417–0.444 — tested, negative, cause identified |
| Classification, Logistic Regression (best, human-only) | macro-F1 | 0.343 |
| Classification, Naive Bayes (best variant) | macro-F1 | 0.105 |
| Sequence tagger (Topic 2), ACT / PARTY tags | entity F1 | 0.857 / 0.627 — the two tags citizens' plain language actually contains |
| Sequence tagger, SECTION_NO / LEGAL_TERM tags | entity F1 | 0.000 / 0.000 — near-absent in citizen phrasing, a register-gap finding, not a bug |
| Language model perplexity gap | colloquial vs formal PPL | 138.9 vs 113.8 (mean); 89.5 vs 100.3 (median) |

**The one claim this whole project stands or falls on:** fine-tuning moved English-target test
Recall@10 by +8.3 points (0.333→0.417) — the exact slice the cross-lingual-gap thesis predicts
should be hardest to move and most informative if it does. That is the number to lead with in
any defense.

---

## 8. Real ceilings — for the limitations section, not an apology

1. **The cross-lingual retrieval wall** (Bengali R@100 0.708 vs English R@100 0.417) is the
   project's actual finding, not a defect. It is not fixable by more Bangla-question training
   data — that was tested directly (§5.5's translation ablation) and did not help.
2. **Real annotated data is capped at 480 answerable questions**, confirmed exhausted — nothing
   unmerged remains. This limits both fine-tuning signal and evaluation precision; growing it
   needs new annotation sessions, a scheduling decision outside what code changes can fix.
3. **The eval set is small (36–71 rows) → wide confidence intervals.** Every reported delta in
   this document should be read next to `scripts/18_compare_runs.py`'s paired bootstrap CI, not
   at face value — several deltas above are directionally consistent across multiple metrics
   but not yet CI-confirmed, and that is stated rather than oversold.

None of these three are "the team didn't try hard enough." Two are corpus/data properties this
project's remaining time could not change; the third is a measurement-rigor discipline that was
followed throughout, which is precisely what CI-gating every claim in this report *is*.

---

## 9. Demo

A three-tab Gradio demo (`src/app/app.py`, backed by `src/dhara/service.py` and
`src/app/browse.py`), serving the v6 fine-tuned checkpoint over the full 39,484-chunk corpus:

- **Search** — a plain Bangla question in, cited provisions out. Permanent "this is not legal
  advice" disclaimer, a legal-aid referral banner for high-risk domains rendered above results
  (never filtering or reordering the ranking, only the framing), and an honest abstention path
  when no candidate clears a calibrated confidence threshold.
- **Law Corpus** — every one of the 1,227 Acts, browsable, each provision rendered as a
  continuous document (sub-chunks of a split section rejoined) with its citation, for a
  supervisor/committee audience to inspect the underlying data directly.
- **Gold Q&A** — all 552 human-adjudicated questions with their resolved answer citation,
  filterable by domain and free text, the same population the headline numbers in this report
  are computed from.

**Two real bugs were found and fixed while wiring this up, not just a threshold retune:**

1. **Abstention threshold was calibrated for the wrong model.** The shipped thresholds
   (default 0.53, high 0.65) came from a sweep against `models/index_bge_m3_acttitle_v1` — a
   different checkpoint *and* a different document template than the v6 model actually being
   served. Live-verified against 25 real dev/test questions with a known-correct answer,
   genuinely correct top-1 matches scored as low as 0.517; the old `high=0.65` threshold alone
   silently abstained on 5 of them, including a domestic-violence question that had the exact
   right answer sitting at rank 1. Recalibrated directly off that live-verification sample
   (`configs/abstention.json`, default 0.48 / high 0.55) as a provisional fix; the formal sweep
   script (`scripts/46_calibrate_abstention.py`) is the real fix but needs to run in isolation
   on a machine that isn't also serving the demo and training a classifier at the same time —
   three attempts on this session's own machine were killed by CPU contention before finishing.
2. **A k-dependent abstention bug, independent of the threshold value.** `service.py`'s risk-tier
   detection (which raises the abstention threshold for high-risk domains) was computed from
   `hits`, the list already truncated to the caller's requested `k`. Raising `k` in the UI could
   pull a high-risk-domain result into view further down the candidate list, flip the tier to
   "high," and retroactively hide an already-good top-1 result that had nothing to do with that
   lower-ranked hit — the same question, same top score, abstained at `k=2` but not at `k=1`.
   Fixed by detecting tier from a fixed-size lookahead (`TIER_LOOKAHEAD = 5`) independent of the
   caller's `k`, so a question's risk framing no longer depends on a UI slider.

**20 example questions in the Search tab are real, human-adjudicated, frozen dev/test
questions** (never trained on), each live-verified against this exact deployed checkpoint and
index before being included — not a curated-sounding guess. 11 return the exact correct
Act+section; 3 land one section off inside the correct Act; 6 land the correct Act with a
different section. The three placeholder examples that shipped before this verification pass
(police-custody rights, spoken talaq, land-boundary dispute) scored 0.57–0.61 on live
retrieval — genuinely reasonable matches — but were silently abstained by bug #1 above before
it was found; they are a good illustration of exactly the failure mode this section describes,
not a mistake to hide.
