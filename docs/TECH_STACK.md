# Dhara — technologies and models: what's used, and why each one specifically

Written 2026-09-19. Every choice below is either a measured decision (an alternative was tried
and lost) or a constraint-driven one (a hard requirement ruled competitors out) — stated as
which, not just asserted as "the right tool." Where a library or model is configured but not
actually wired into running code, that's stated too, rather than left to look more finished than
it is.

---

## 1. Core language and ML framework

**Python** — the whole project, including hand-written model architectures, not just glue code
around a higher-level framework.

**PyTorch**, not a higher-level wrapper like Keras — chosen specifically because three of this
project's models are hand-written `nn.Module` architectures with non-trivial internals that
needed to be built and explained line-by-line for the syllabus's own requirements: the skip-gram
negative-sampling Word2Vec model (`src/dhara/word2vec.py`), the RNN/BiLSTM classifiers
(`src/dhara/classify.py`), the BiRNN sequence tagger (`src/dhara/models/tagger.py`), and the
causal LSTM language model (`src/dhara/models/language_model.py`). A framework that abstracts
away the forward pass would have made "explain your architecture's tensor shapes" (a real
committee-level question this project is built to answer, see
`docs/TECHNICAL_DEEPDIVE.md` §4) much harder to answer honestly.

## 2. Transformer / retrieval stack

**`sentence-transformers`** wraps the pretrained bi-encoder checkpoint (`SentenceTransformer`
class) for both the zero-shot baseline and the LoRA fine-tune, and provides
`SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` for the actual fine-tuning loop
(`notebooks/ssh-retrieval-finetune.ipynb`). Chosen over hand-writing the training loop directly
against `transformers` because the contrastive-loss machinery (in-batch negative construction,
`NO_DUPLICATES` batch sampling) is exactly what this project needs and is easy to get subtly
wrong by hand — using a maintained implementation here is a "call a library" decision, not a
"pretend I invented this" one, unlike the from-scratch models in §1, which is a deliberate,
task-appropriate split (§8 explains the general principle).

**`transformers`** (Hugging Face) — the underlying model/tokenizer loading library everything
above sits on. Also loads `csebuetnlp/banglat5_nmt_bn_en` directly (`AutoModelForSeq2SeqLM`) for
the dual-query translation ablation.

**`peft`** — LoRA. `LoraConfig(r=16, lora_alpha=32, target_modules=['query','value'])` +
`get_peft_model()`. Chosen because full fine-tuning of a 568M-parameter model does not fit a
single T4 GPU's memory budget at a workable batch size; LoRA trains 0.28% of the parameters and
still moved the headline metric (+2.8pt test Recall@10, +8.3pt on the English-target slice) — the
constraint (GPU memory) picked the method, and the method was then verified to actually work
before being trusted.

**`FlagEmbedding`** — the *only* library that exposes BGE-m3's sparse-lexical and ColBERT
multi-vector heads (`BGEM3FlagModel`). `sentence-transformers`' generic wrapper only ever reaches
the dense head. Used exclusively in the Stage-3 hybrid-retrieval ablation
(`docs/PROJECT_REPORT.md` §5.5) — tested, found not to help at this data scale, kept in the
dependency list because the ablation is a real, reported result, not because it's part of the
deployed path.

**`huggingface_hub`** (`snapshot_download`) — used narrowly, to pull `sparse_linear.pt` /
`colbert_linear.pt` from the original pretrained `BAAI/bge-m3` release and attach them to the
locally fine-tuned backbone before loading it as a `BGEM3FlagModel` (the LoRA fine-tune never
touched those two heads, and `SentenceTransformer.save()` never writes them — see
`TECHNICAL_DEEPDIVE.md` §3.4/§3.5's discussion of this).

## 3. Classical / lexical retrieval

**`rank_bm25`** (`BM25Okapi`) — a small, dependency-light BM25 implementation, chosen because BM25
here is deliberately kept simple: it's the reported lexical floor, not a component under active
development, so a heavyweight search-engine library (Elasticsearch, Whoosh) would be unjustified
complexity for a role that's explicitly "the un-improved baseline."

**No FAISS or vector database.** At 39,484–47,000 chunk vectors, a brute-force
`embeddings @ query_vector` matmul on CPU or GPU is, per `retrievers/biencoder.py`'s own
docstring, "milliseconds" — an ANN index (FAISS, Annoy, a hosted vector DB) would add real
operational complexity (index build/rebuild logic, approximate-search recall loss to reason
about) for zero measurable latency benefit at this scale. This is a right-sized-for-the-data
decision, not an oversight — the corpus would need to be roughly two to three orders of magnitude
larger before an ANN index would pay for itself.

## 4. Classical NLP / classical ML

**`scikit-learn`** — `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `KFold` +
`cross_val_score` for the classifier's regularization sweep, and the metric functions
(`f1_score`, `accuracy_score`, `confusion_matrix`) behind every classification number reported.
The standard, well-tested choice for the classical (non-neural) half of the generative-vs-
discriminative comparison the syllabus asks for — reimplementing TF-IDF or logistic regression by
hand here would add risk of a subtle bug without adding any pedagogical value the syllabus
actually wants demonstrated (unlike the RNN architectures in §1, which *are* the point).

**No gensim**, despite gensim being the default choice almost anyone would reach for to train
Word2Vec. Two reasons, one practical and one deliberate: no gensim wheel was available for the
Python version this project's dev environment runs, and — independently of that — the project
wanted to *explain* the skip-gram negative-sampling objective at the level of its actual loss
function (`docs/TECHNICAL_DEEPDIVE.md` §3.2 walks through it), which is much easier to do
honestly for ~60 lines of hand-written PyTorch than for a library call whose internals weren't
written by the project.

## 5. Scraping and data collection

**`BeautifulSoup4` + `lxml`** — HTML parsing for both the bdlaws portal scraper and the news-site
question miners. `lxml` specifically as the parser backend for speed at corpus scale (39k+ pages
parsed cumulatively across the crawl).

**`requests`** — HTTP client, with a real identifying `User-Agent` string and a fixed per-request
delay (`1.5s`) — scraping conduct treated as part of the spec, not an afterthought, per
`CLAUDE.md`'s own non-negotiable rules.

**`PyYAML`** — every configuration file (`configs/*.yaml`: domain definitions with risk tiers,
model checkpoint registry, synthetic-question topic inventories) is YAML rather than JSON or a
Python config module, specifically so a non-engineer collaborator can read and edit domain/topic
definitions without touching code.

## 6. Serving / demo

**Gradio**, not Streamlit, Flask, or a custom frontend — chosen against the supervisor's own
explicit guidance recorded in `DECISIONS.md` (2026-08-23): "a fancy GUI is not required... there
should be a clear input-output system." Gradio's `Blocks`/`Tab` API gets a working multi-tab
input→output UI (search box, corpus browser, gold-set browser) running with a fraction of the
code a custom frontend+backend split would need, which matters when UI polish was never the
graded part of the project. The model loads once at module level
(`dhara = Dhara.load()` outside any request handler) specifically to avoid the "demo feels
frozen" trap `CLAUDE.md`'s known-traps table calls out by name (reloading a 568M-parameter
transformer per request would make every query take tens of seconds instead of milliseconds).

## 7. Summarization sub-project (separate stack, not yet fully documented elsewhere)

**`networkx`** — graph-based TextRank, one of the two required non-neural extractive summarization
baselines (`scripts/72_summary_extractive_baselines.py`), scored before any GPU time is spent on
a neural summarizer, mirroring the same "a neural result needs a non-neural floor to be
publishable" rule BM25 follows for retrieval.

**`rouge-score`, `sacrebleu`** — ROUGE-1/2/L and chrF, the two language-agnostic, no-download-
required metrics used to score summaries offline (BERTScore and human-usefulness scoring are
explicitly noted in that script's own docstring as needing a downloaded scoring model / a human
annotator respectively, and are left for the fine-tuned summarizer stage to be judged against,
not this floor).

---

## 8. Pretrained models — the registry, and why each

### Retrieval

| checkpoint | role | why |
|---|---|---|
| **`BAAI/bge-m3`** | primary dense bi-encoder (zero-shot control + LoRA fine-tune base) | Multilingual, 568M params, 1024-dim, no query/passage prefix needed. Chosen over the alternative below after a **measured** comparison, not by reputation. |
| `intfloat/multilingual-e5-base` | retired, kept as a reported comparison row | 278M params, 768-dim, needs `"query: "`/`"passage: "` prefixes. Reached only 8% Recall@10 on the English-only-Act gold slice — close enough to BM25's 0% that it falsified the working assumption that "any multilingual encoder" would cross the cross-lingual wall. BGE-m3 reached 22% on the same slice. Kept in `configs/models.yaml` and reported, not deleted, because a rejected alternative with its number attached is stronger evidence than an unstated assumption. |
| **`BAAI/bge-reranker-v2-m3`** | cross-encoder, stage-2 reranking | Same model family as the bi-encoder (both from BAAI's BGE line), reducing the number of distinct architectures/tokenizers the project has to reason about, while being architecturally the *opposite* of the bi-encoder (joint attention over query+document, see `TECHNICAL_DEEPDIVE.md` §3.5) — exactly the complementary capability a two-stage retrieval pipeline needs. |

### Cross-lingual / register (used in the Stage-3 hybrid ablation)

| checkpoint | role | why |
|---|---|---|
| **`csebuetnlp/banglat5_nmt_bn_en`** | bn→en query translation | Same research group (csebuetnlp) already trusted elsewhere in the project for Bangla-specific tooling; a dedicated NMT fine-tune of BanglaT5 rather than a generic multilingual translation model, on the theory that Bangla-specific pretraining would handle citizen-register colloquial Bangla better than a generic model — tested directly (not just assumed): it measurably improved nothing and a specific truncation bug in its output was found and documented (`TECHNICAL_DEEPDIVE.md`/`PROJECT_REPORT.md` §5.5), a real, disclosed limitation of this specific model choice at the sentence lengths this project's questions actually have. |
| **`csebuetnlp/normalizer`** | Bangla-specific text normalization, required by the csebuetnlp model family (not on PyPI, installed via `git+https://github.com/csebuetnlp/normalizer`) | Needed as a preprocessing step ahead of any csebuetnlp checkpoint (BanglaBERT, BanglaT5) — those models were trained against this specific normalizer's output distribution, and skipping it would silently feed them out-of-distribution input. |

### Classification — a real config/code gap, stated plainly

**`configs/models.yaml` names `csebuetnlp/banglabert`** (an ELECTRA-architecture,
Bangla-specific pretrained transformer) as "the classification checkpoint," with a comment citing
it as "the strongest Bangla-specific" option available. **The actual classification code
(`src/dhara/classify.py`) does not load or use it** — Topic 1's classifiers are TF-IDF+Naive
Bayes/Logistic Regression and from-scratch RNN/BiLSTM models over the project's own Word2Vec
embedding matrix, not a fine-tuned BanglaBERT. This is worth being direct about rather than
glossing over: the config entry documents an intended/considered checkpoint for a
transformer-based classification variant that was never built, most plausibly because
classification was explicitly flagged as *not required* by the supervisor
(`DECISIONS.md`, 2026-08-23: "you are not bound to do classification works") and the
TF-IDF/RNN comparison already satisfies the syllabus's generative-vs-discriminative requirement
on its own — spending a GPU fine-tuning run on a non-required subsystem wasn't prioritized. If
asked "why does the config mention BanglaBERT when the code doesn't use it," this is the honest
answer: it's a recorded intention, not a currently-running path.

### Self-trained, not pretrained

Four models are trained entirely from scratch on this project's own data, sharing one artifact
(the Word2Vec-derived embedding matrix, `build_pretrained_embedding_matrix()`,
`TECHNICAL_DEEPDIVE.md` §4.4) as their only "pretrained" input:

- **Word2Vec** (`src/dhara/word2vec.py`) — skip-gram with negative sampling, 200-dim, trained on
  the project's own 39,484-chunk corpus.
- **`VanillaRNNClassifier` / `StackedBiLSTMClassifier`** (`src/dhara/classify.py`) — Topic 1.
- **`BiRNNSequenceLabeler`** (`src/dhara/models/tagger.py`) — Topic 2.
- **`StackedLSTMLanguageModel`** (`src/dhara/models/language_model.py`) — Topic 3.

None of these import a pretrained checkpoint from Hugging Face or anywhere else — they exist
specifically to demonstrate the syllabus's from-scratch architectures rather than to compete with
the transformer stage on raw performance, and their honestly-reported, more modest numbers
(§7 of `PROJECT_REPORT.md`) reflect that they were never meant to.

## 9. What was deliberately NOT used, and why

- **A commercial embeddings API** (OpenAI, Cohere, Google) — ruled out on three independent
  grounds: reproducibility (every number in this project must be regenerable offline, without a
  paid API key or rate limit, per `CLAUDE.md`'s own non-negotiable evaluation-integrity rules),
  cost at corpus scale (39,484 chunks embedded repeatedly across a full experiment history), and
  privacy (real citizen questions, even paraphrased, are not something this project sends to a
  third-party API without a stated reason to).
- **A vector database** — see §3 above; unjustified complexity at this corpus size.
- **spaCy / NLTK for tokenization** — Bangla-specific normalization needs (ZWJ/ZWNJ handling,
  danda variants, Bangla-digit/ASCII-digit equivalence, the specific stopword list tuned against
  this project's own BM25 failure modes) aren't well served by a generic multilingual tokenizer;
  `src/dhara/normalize.py` was built as a small, fully-understood, project-specific module
  instead of adding a large general-purpose NLP library for a handful of specific rules.
- **A from-scratch transformer** — explicitly out of scope by the syllabus's own build-order logic
  (`docs/PIPELINE.md`): "no transformer is written from scratch — Topic 0 [[the embedding-matrix
  work]] answers 'explain your embeddings'" instead. The project demonstrates transformer
  *understanding* through fine-tuning, prefix logic, and architecture comparison (bi- vs
  cross-encoder), not through reimplementing attention.
