# The Dhara pipeline, end to end

This document explains what the system does, stage by stage, and where each thing
you were taught in the course actually sits in it. Read it once before building
anything; it is the map the rest of the repo assumes.

---

## What the system does, in one paragraph

A citizen types a question in ordinary Bangla. The system finds the exact
provision of Bangladeshi law that answers it and shows that provision with its
citation. It never writes an answer of its own. The research question is whether
models can bridge two gaps at once: the **register gap** between how citizens
speak and how law is written, and a **language gap**, because much of the law
that citizens ask about is written in English and has never been translated.

---

## Stage 1 — Preprocessing

**In:** raw legal text and raw citizen questions.
**Out:** normalized strings.
**Code:** `src/dhara/normalize.py`

Bangla text arrives with problems that are invisible on screen and fatal to
matching:

| Problem | Why it breaks things |
|---|---|
| Multiple byte encodings for one visible character | "সাপ্তাহিক" typed two ways will not compare equal. Fixed by Unicode NFC. |
| ZWNJ / ZWJ (U+200C / U+200D) | Invisible joiners inserted inconsistently by web editors. They split tokens silently. |
| Two danda characters (U+0964 `।` and U+09F7 `৷`) | bdlaws uses both, sometimes in one Act. |
| Bangla vs ASCII digits | A user writes "১০৩ ধারা" or "section 103" and means the same provision. |

**The key design decision: two levels, not one.**

- `light()` — NFC, strip ZWNJ, unify danda, tidy whitespace. **For transformers.**
  Their tokenizers were trained on natural text; aggressive cleaning moves your
  input off the distribution the model learned and costs real accuracy.
- `aggressive()` — everything above plus ZWJ removal, Bangla→ASCII digits,
  punctuation padding, lowercasing. **For BoW / TF-IDF / BM25 / Word2Vec.**
  These match surface forms, so every unnormalized variant is a missed match.
- `content_tokens()` — `aggressive()` plus stopword removal, for lexical models
  only.

Using the wrong one is a real, measurable performance loss and it is invisible
when you read the output. This is why `normalize.py` is unit-tested.

---

## Stage 2 — The corpus

**In:** BLAD (a published dataset of Bangladeshi acts) + the bdlaws portal.
**Out:** `corpus_v1.jsonl` — one record per retrievable chunk.
**Code:** `src/dhara/blad.py`, `src/dhara/scrape.py`, `src/dhara/schema.py`

The unit of retrieval is the **provision** — one ধারা of an Act, or one অনুচ্ছেদ
of the Constitution. One provision = one embedding = one citable answer. A
provision longer than 400 words is split into overlapping sub-chunks that share
one citation, because a 1,200-word provision averaged into a single vector
matches nothing well.

Every record carries its citation: act title, provision number in both numeral
systems, source URL, crawl date, language, domain, risk tier. **The metadata is
the citation** — a retrieved provision that cannot be cited is useless here, so
schema completeness is a correctness requirement, not bookkeeping.

Two text fields, deliberately: `text_raw` is what a human is shown (the law as
printed), `text_bn` is what models consume. Never show a Bangla reader normalized
text.

---

## Stage 3 — Questions

**In:** newspaper legal-advice columns.
**Out:** a pool of real citizen questions.
**Code:** `src/dhara/mine.py`, `src/dhara/questions.py`

Real questions matter because **register is the thing being measured**, and no
model invents authentic register reliably. A person writes "বাড়িওয়ালা কি ইচ্ছা
করলেই আমাকে বের করে দিতে পারবে?" — a researcher imagining a citizen writes
something tidier, and the whole experiment becomes circular.

Three kinds of question, used differently:

| Kind | Source | Used for |
|---|---|---|
| **Mined** | newspaper columns | gold test set + some training |
| **Synthetic** | generated from provisions | training only, never evaluation |
| **Formal counterpart** | annotator rewrites a real question in legal register | the controlled register contrast, and Seq2Seq training |

---

## Stage 4 — Annotation

**In:** the question pool + BM25 candidates.
**Out:** `gold_test_v1.jsonl`.
**Code:** `scripts/06_make_annotation_sheets.py`, `docs/annotation_guideline.md`

A human decides every gold label. Candidates come from BM25 rather than a neural
model on purpose: that biases the gold set toward what a *lexical* retriever can
find, which makes this project's own claim **harder** to prove. Biasing in the
conservative direction is what makes the final numbers defensible.

The gold set is touched exactly once, at the end. Everything else uses dev.

---

## Stage 5 — Representations and models

This is where the course syllabus lives. Each topic below is a real artifact in
this project, not an exercise bolted on.

### Topic 0 — Pretrained vectors and the embedding matrix

`build_pretrained_embedding_matrix()` maps your vocabulary onto pretrained
vectors, producing a `(vocab_size, dim)` float tensor used to initialise
`nn.Embedding`. Every RNN model below starts here.

Two vector sources, compared:
- **Self-trained Word2Vec** on the legal corpus (gensim, skip-gram).
- **General-purpose Bangla vectors** (`api.load(...)`).

The comparison is a result in itself: nearest neighbours of আটক, নামজারি, খারিজ,
জামিন differ sharply between a legal-trained and a general-trained model, and
that table is the most direct human-readable evidence of what an embedding
learned. Expect the self-trained vectors to be *weak* — a few hundred thousand
words is small for Word2Vec. Report that honestly as a corpus-size finding.

### Topic 1 — Classification

Predict the legal `domain` of a question. Four models on identical labels:

| Model | Representation | Why it is here |
|---|---|---|
| Multinomial Naive Bayes | TF-IDF | **generative** counterpart |
| Logistic Regression | TF-IDF | discriminative baseline on the same features |
| `VanillaRNNClassifier` | Topic 0 embeddings | `nn.RNN`, `h_n.squeeze(0)` |
| `StackedBiLSTMClassifier` | Topic 0 embeddings | `nn.LSTM(num_layers=2, bidirectional=True)`, `torch.cat((h_fwd, h_bwd), dim=1)` |

Report macro-F1, not accuracy — accuracy hides failure on small domains.

### Topic 2 — Sequence tagging

`BiRNNSequenceLabeler` over citizen questions, tagging each token:
`ACT` / `SECTION_NO` / `LEGAL_TERM` / `PARTY` / `O`.

Not an exercise — it does a job. When a question contains "১০৩ ধারা", the tagger
extracts it and the system boosts that provision directly instead of hoping
similarity finds it. `nn.RNN(bidirectional=True)`, `logits.view(-1, C)`,
`CrossEntropyLoss(ignore_index=0)` so padding is not trained on.

Labels bootstrap by matching corpus terms into questions, then get hand-corrected.

### Topic 3 — Language modelling

`StackedLSTMLanguageModel` trained on formal legal Bangla.

Its job here is **measurement**. Train on legal text, then compute perplexity on
colloquial questions versus formal ones. If colloquial questions are far more
surprising to a model of legal language, that *is* the register gap — measured
independently of any retrieval system. It is a second, orthogonal piece of
evidence for the central claim, and it comes from a model you built.

Generation (`torch.multinomial`) is used only to show samples of what the model
learned about legal Bangla. It never produces answers for users; the system has
no generation layer.

### Topic 4 — Sequence to sequence

`Seq2SeqTranslation`, but **register translation, not language translation**:
colloquial Bangla → formal legal Bangla.

There is no parallel Bangla–English legal corpus to train on, because no Act
exists in both languages. But we *do* have parallel register pairs: the 100 `PAIR`
annotations where a human rewrote a real citizen question in legal register.

Then use it: translate the query into legal register first, retrieve second. This
is query reformulation, a real IR technique, and it attacks the register gap
head-on rather than sitting beside the project. Encoder passes its `(h, c)`
context to the decoder; training uses teacher forcing.

### Topic 5 — Transformers

**Pretrained and fine-tuned. Nothing is written from scratch here.**

The "explain how you did your embeddings" requirement is answered by Word2Vec in
Topic 0, which is ours line by line — the objective, the frequency subsampling,
the unigram^0.75 negative distribution — and whose learned geometry can be shown
directly as a nearest-neighbour table. A hand-written transformer would add weeks
and explain less, because its learned structure is not inspectable in the same
human-readable way.

Two different checkpoints for two different jobs:

| Job | Checkpoint | Why |
|---|---|---|
| Classification (Topic 1) | `csebuetnlp/banglabert` | ELECTRA discriminator, strongest Bangla-specific model. Requires its own normalizer. |
| Retrieval | a multilingual sentence encoder (LaBSE / BGE-M3 / multilingual-E5) | must place Bangla questions and **English** provisions in one space |

A **bi-encoder** encodes the question and the provision separately and compares
the two vectors by cosine similarity. Because provisions are encoded once and
cached, search is fast. Fine-tuning uses a contrastive objective: pull a question
and its correct provision together, push everything else apart.

A **cross-encoder** instead reads the question and one provision *together* and
scores the pair. It is far more accurate because the two texts attend to each
other, and far slower, so it only reranks the top candidates.

---

## Stage 6 — Retrieval and serving

```
question
   │
   ├─ normalize (light for neural, aggressive for lexical)
   ├─ tag (Topic 2) → pull out any explicit ধারা number
   ├─ reformulate (Topic 4) → colloquial rewritten as formal
   │
   ├──────────────┬──────────────────┐
   │              │                  │
 BM25         bi-encoder        explicit provision
 (lexical)    (semantic)        number match
   │              │                  │
   └──────────────┴──────────────────┘
                  │  fuse (reciprocal rank fusion)
             top 50 candidates
                  │
           cross-encoder rerank
                  │
        top 3 provisions + citations
                  │
        score below threshold? → abstain
```

**Why BM25 is still here.** It is a TF-IDF-family ranking function, not a
language model, and it is one row in one table. It exists because a neural result
with no non-neural floor is unpublishable: "our model scores 68%" means nothing
until you know what keyword matching alone scores. It also has a specific
weakness this project measures — it cannot match a Bangla question to an English
provision at all.

**Abstention.** Below a calibrated score threshold the system says it has no
confident match. In a legal tool, a false abstention is an annoyance and a
confident wrong answer is the actual harm.

---

## Stage 7 — Evaluation

**Code:** `src/dhara/metrics.py`, `src/dhara/evaluate.py`

Metrics: Recall@k, Hit@k, MRR@10, nDCG@10 for retrieval; accuracy and macro-F1
for classification; perplexity for the language model.

Three rules that make the numbers mean anything:

1. **The gold set is never trained on, tuned on, or peeked at.** A leakage
   assertion runs on every evaluation.
2. **Split by provision, not by question.** All questions derived from one
   provision go to the same split, or near-duplicates leak across train and dev.
3. **No difference is claimed without a confidence interval.** A paired bootstrap
   over per-query results; two models within noise are reported as within noise.

Every number in the report comes from a script writing a JSON file. No number is
ever typed by hand.

---

## What is required, and what is optional

The supervisor's written guidance settles this, and it is looser than it looked:

> "You are not bound to do classification works. It's an open ended project."

**The supervised core of this project is the retrieval fine-tuning itself** —
contrastive learning on labelled question→provision pairs. That satisfies
"primarily supervised" with no classifier at all. Everything in the table below
that is marked *demonstration* is there for syllabus coverage and because it
happens to earn its place, not because a requirement forces it.

| Piece | Status | Why |
|---|---|---|
| Corpus + documented collection | **required** | Sir: describe sources, methods, preprocessing |
| Word2Vec, self-trained | **required in practice** | this is the "explain your embeddings" answer |
| Pretrained-vector comparison | **recommended by Sir** | "a balanced approach" |
| Retrieval fine-tuning | **required** | the supervised core, and the thesis |
| Gold set + register measurement | **required** | the contribution |
| Topic 3 LM perplexity | high-value demonstration | second, independent measure of the register gap |
| Topic 4 Seq2Seq reformulation | high-value demonstration | improves retrieval directly |
| Topic 1 classifiers | cheap demonstration | generative vs discriminative, ~20 lines for NB |
| Topic 2 tagger | lowest-value demonstration | useful, but cut first if time runs short |
| Clustering (ARI/NMI) | optional | Sir: unsupervised "can add value" |
| Gradio demo | **required, deliberately plain** | Sir: "a fancy GUI is not required" |

## Build order

Each step reuses the previous one's vocabulary and embedding matrix, so order
matters:

1. **Topic 0** — vocabulary, Word2Vec, embedding matrix, + pretrained comparison
2. **Topic 5** — fine-tuned bi-encoder (the headline result; do it early, because
   if the central claim fails you want to know now, not in week six)
3. **Topic 3** — LSTM language model + the perplexity measurement
4. **Topic 4** — Seq2Seq register translation
5. **Topic 1** — NB, LR, VanillaRNN, StackedBiLSTM classifiers
6. **Topic 2** — BiRNN tagger
7. Fusion, reranking, abstention, clustering, demo

BM25 and the evaluation harness exist already and are the floor everything else
is measured against.
