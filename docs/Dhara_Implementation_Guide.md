# ধারা (Dhara) — Complete Implementation Guide

**A Bangla Legal Retrieval System for Citizen Services**
*From empty repository to working demo, in 8 weeks, with 4 people*

---

## §0 · How to use this document

This is an execution manual, not a proposal. It assumes the proposal is approved and you are starting Monday of Week 1.

Read §0–§2 as a team, together, before anyone writes code. After that, each person mainly lives in their own phase sections, but **everyone must read §12 (Integration Contracts)** — that section is what lets four people work in parallel without blocking each other.

Three rules that govern everything below:

1. **The toy pipeline runs end-to-end in Week 1.** 50 sections, 30 questions, BM25 only, ugly output. Before anything is scaled, polished, or optimized. Teams fail this kind of project by perfecting components in isolation and discovering integration bugs in Week 6.
2. **Freeze artifacts, then version them.** `corpus_v1.jsonl` never changes after Week 2. If it must change, it becomes `corpus_v2.jsonl` and every result computed on v1 is re-run or explicitly labelled. Silent data drift makes results incomparable and is the most common way student projects produce numbers nobody can defend.
3. **The gold test set is touched exactly once, at the end.** Not for debugging. Not for "just checking." Use the dev split for everything else. Every time you peek, you leak.

---

## §1 · Team roles

Four people, four vertical slices. Each owns files nobody else edits, and consumes files produced by others through the frozen contracts in §12.

| Role | Person | Owns | Primary sections |
|---|---|---|---|
| **A — Data Engineer** | | Scraping, section splitting, normalization, corpus QA | §4 |
| **B — Dataset & Annotation Lead** | | Question mining, synthetic generation, gold set, intent labels, hard negatives | §5 |
| **C — Classical Models & Evaluation** | | Eval harness, BM25, Word2Vec, BiLSTM, Naive Bayes, clustering | §6, §7.1–7.3, §7.8–7.9 |
| **D — Transformers & System** | | Bi-encoder, cross-encoder, fusion, inference service, UI, figures | §7.4–7.7, §8, §9, §10.3 |

**Why this split works:** A and B are both data roles but have almost no overlap in tooling — A writes HTML parsers, B writes prompts and annotation guidelines. C and D are both modelling roles, but C owns the *measurement* infrastructure that D's work is judged by, which keeps D honest. The person building the fancy model should not be the person who wrote the metric.

**Load balance warning.** A's work is front-loaded (Weeks 1–3) and D's is back-loaded (Weeks 5–8). This is deliberate but must be managed: from Week 4, A shifts to writing the methodology chapter and running corpus QA; from Week 6, C shifts to error analysis and results tables. Nobody idles, but nobody is at 100% for all eight weeks either.

**Everyone, regardless of role:**
- Writes their own section of the report as they go, not in Week 8.
- Maintains their part of the decision log (§2.5).
- Reviews at least one other person's pull request per week.

---

## §2 · Environment and repository setup

*Owner: D, in Week 1, Day 1. Everyone else is blocked until this exists — do it first.*

### 2.1 Repository structure

```
dhara/
├── README.md
├── requirements.txt
├── .gitignore
├── DECISIONS.md              # the decision log — see §2.5
│
├── configs/
│   ├── domains.yaml          # which Acts, which domains
│   ├── models.yaml           # checkpoint names, hyperparameters
│   └── paths.yaml            # single source of truth for file locations
│
├── data/
│   ├── raw/                  # scraped HTML, NEVER edited, NEVER committed
│   ├── interim/              # intermediate parses
│   ├── processed/            # frozen artifacts (§11)
│   │   ├── corpus_v1.jsonl
│   │   ├── train_pairs_v1.jsonl
│   │   ├── dev_pairs_v1.jsonl
│   │   ├── gold_test_v1.jsonl
│   │   └── hard_negatives_v1.jsonl
│   └── external/             # pretrained vectors, downloaded models
│
├── src/
│   ├── dhara/
│   │   ├── __init__.py
│   │   ├── normalize.py      # A owns
│   │   ├── scrape.py         # A owns
│   │   ├── section_split.py  # A owns
│   │   ├── schema.py         # A owns — shared dataclasses
│   │   ├── synth.py          # B owns
│   │   ├── negatives.py      # B owns
│   │   ├── metrics.py        # C owns — the eval harness
│   │   ├── evaluate.py       # C owns
│   │   ├── retrievers/
│   │   │   ├── base.py       # C owns — the Retriever interface
│   │   │   ├── bm25.py       # C
│   │   │   ├── word2vec.py   # C
│   │   │   ├── bilstm.py     # C
│   │   │   ├── biencoder.py  # D
│   │   │   └── hybrid.py     # D
│   │   ├── rerank.py         # D
│   │   ├── classify.py       # C
│   │   ├── cluster.py        # C
│   │   └── service.py        # D — inference orchestration
│   └── app/
│       ├── app.py            # D — Gradio UI
│       └── assets/
│
├── scripts/                  # thin CLI wrappers, one job each
│   ├── 01_survey_acts.py
│   ├── 02_scrape.py
│   ├── 03_build_corpus.py
│   ├── 04_generate_questions.py
│   ├── 05_mine_negatives.py
│   ├── 06_train_biencoder.py
│   ├── 07_train_crossencoder.py
│   ├── 08_build_index.py
│   └── 09_run_eval.py
│
├── notebooks/                # exploration only — never the source of a result
├── results/
│   ├── tables/               # CSV, auto-generated
│   ├── figures/
│   └── runs/                 # one JSON per experiment run
└── report/
```

**One hard rule about `notebooks/`:** notebooks are for looking at things. Every number that appears in the report comes from a script in `scripts/` writing a file to `results/`. A number that exists only in a notebook cell cannot be reproduced, and in Week 8 you will not remember which cell produced it.

### 2.2 Dependencies

```txt
# requirements.txt
python>=3.10

# data
requests==2.32.*
beautifulsoup4==4.12.*
lxml
pyyaml
pandas
tqdm

# classical
rank-bm25
gensim==4.3.*
scikit-learn==1.5.*

# neural
torch>=2.2
transformers>=4.44
sentence-transformers>=3.0
datasets
accelerate

# viz + app
matplotlib
seaborn
umap-learn
gradio>=4.44
```

Plus, installed separately because it is not on PyPI:

```bash
pip install git+https://github.com/csebuetnlp/normalizer
```

This is the normalization utility published by the BanglaBERT authors. **If you use BanglaBERT, you must use their normalizer** — the model was trained on text processed this way, and substituting your own pipeline will silently cost you performance you will then misattribute to the model.

### 2.3 Compute

Colab or Kaggle free tier (T4, 16GB) is sufficient for everything in this project. Nothing here requires more.

Practical Colab discipline:
- **Mount Drive and checkpoint every epoch.** Sessions die. Budget for it rather than being surprised by it.
- Keep `data/processed/` in Drive, sync to local disk at session start — reading JSONL from Drive directly is slow.
- Set `max_seq_length=256` for the bi-encoder unless your section-length audit (§3.6) says otherwise. Going to 512 roughly doubles time and memory for marginal gain on text this short.
- Use `fp16=True` in training args. Free speedup on T4.

### 2.4 Git workflow

Branch per person: `feat/A-scraper`, `feat/D-biencoder`. Merge to `main` weekly, minimum. Four people on one branch for eight weeks is how you get a Week 7 merge conflict in the eval harness.

`.gitignore` must include `data/raw/`, `data/external/`, `*.bin`, `*.safetensors`, checkpoints. **`data/processed/*.jsonl` should be committed** — it is small (a few MB), and it is the thing you most need to reproduce.

### 2.5 The decision log

Create `DECISIONS.md` on day one. Every non-obvious choice gets three lines: date, decision, reason.

```markdown
## 2026-08-24 — Section is the retrieval unit, not the Act
Acts run to hundreds of sections; retrieving a whole Act is useless to a
citizen and unciteable. Cost: sections that reference each other lose that
link. Accepted; noted as a limitation.

## 2026-08-26 — Dropped Penal Code from scope
500+ sections would be 40% of the corpus and skew domain balance. Criminal
procedure questions are also the highest-risk category for a non-advice tool.
Revisit only if corpus falls short of 800 sections.
```

This file is the raw material for your methodology chapter. Writing methodology in Week 8 from memory is where good projects lose their details — you will remember *what* you did and have forgotten *why*, and the why is what gets marked.

---

## §3 · Phase 0 — The Week 1 spike

*Owner: everyone, Week 1. This is the most important week of the project.*

The goal of Week 1 is **not** to build anything good. It is to prove that a question can travel all the way through the system and come out the other side, and to find out — while it is still cheap — whether your data assumptions hold.

### 3.1 The feasibility gate (blocking, Day 1–2)

*Owner: A*

**Before anything else, verify that the Bangla text you are planning to use actually exists.** The bdlaws portal publishes many older Acts in English only; Bangla versions exist for a substantial subset but not universally, and which Acts have them is not something to assume.

Write `scripts/01_survey_acts.py` that, for each candidate Act in `configs/domains.yaml`:
- fetches the Act's landing page,
- detects whether a Bangla version is available,
- counts sections,
- records the result to `results/tables/act_survey.csv`.

Output columns: `act_id, act_name_en, act_name_bn, bangla_available, n_sections_bn, n_sections_en, notes`.

**The decision this drives:**

| Survey outcome | Action |
|---|---|
| ≥ 800 sections with Bangla text across ≥ 4 domains | Proceed as planned. |
| 400–800 sections | Narrow to 3 domains, keep Bangla, reduce gold set to 150. |
| < 400 sections | **Pivot to cross-lingual retrieval**: Bangla queries against English legal text. |

That third row is not a failure mode — take it seriously as a legitimate outcome. Bangla-query → English-passage retrieval is a *harder and more interesting* problem, multilingual encoders (LaBSE, mE5, BGE-M3) are built for exactly it, and the lexical gap becomes total rather than partial, which sharpens your thesis rather than weakening it. The only real cost is that "self-trained Word2Vec on Bangla legal text" becomes less central, so you would train it on the Bangla *query* side and analyze it there instead.

**Record the outcome in `DECISIONS.md` on Day 2.** Everything downstream depends on it.

### 3.2 The toy pipeline (Day 3–5)

Three Acts. Fifty sections. Thirty questions written by your own team in ten minutes — they do not need to be good, they need to exist. BM25 only. Print results to the terminal.

```
scripts/02_scrape.py --acts 3 --out data/raw/
scripts/03_build_corpus.py --in data/raw/ --out data/processed/corpus_toy.jsonl
# hand-write 30 questions into data/processed/gold_toy.jsonl
scripts/09_run_eval.py --corpus corpus_toy.jsonl --gold gold_toy.jsonl --retriever bm25
```

**Definition of done for Week 1:** that last command prints a Recall@5 number. Any number. It can be terrible. The point is that scraping → parsing → normalization → indexing → retrieval → scoring is a connected path with no gaps.

If this does not work by Friday of Week 1, you have an environment or integration problem, and you should spend the weekend on it rather than proceeding. Everything after this compounds.

### 3.3 Parallel Week 1 work

While A does the survey and scraping:
- **B** writes the annotation guideline draft (§4.4) and hand-writes the 30 toy questions.
- **C** writes `metrics.py` (§6) — the eval harness is the first real code, not the last.
- **D** sets up the repo, the `Retriever` interface (§12.2), and a Gradio skeleton that returns hardcoded output.

By Friday, every person has touched the codebase and the interfaces exist.

---

## §4 · Phase 1 — Corpus curation

*Owner: A. Weeks 1–3. Frozen end of Week 3.*

### 4.1 Scraping

Design principles, in priority order:

**Archive raw, always.** Save the unmodified HTML to `data/raw/{act_id}/{section_id}.html` before parsing anything. Your section splitter will be wrong on at least one Act — probably several — and you will discover this in Week 4. Re-parsing from a local archive takes minutes; re-crawling takes hours and is rude.

**Be polite.** 1.5–2 seconds between requests, a real User-Agent that identifies your project and gives a contact address, and respect `robots.txt`. Check the site's terms of use before you start and note what they say in `DECISIONS.md`.

**Be resumable.** Check whether the file exists before fetching. A crawl that cannot resume will be run from scratch three times.

```python
# src/dhara/scrape.py — sketch
import time, pathlib, requests

HEADERS = {"User-Agent": "Dhara-Research-Crawler/1.0 (student NLP project; contact@example.edu)"}
DELAY = 1.5

def fetch(url: str, dest: pathlib.Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"          # do not trust the header; bdlaws serves Bangla
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    time.sleep(DELAY)
    return resp.text
```

**Record the crawl date per document.** It goes in the chunk schema and is displayed in the UI. Laws are amended; a dated snapshot is an honest claim, an undated one is not.

### 4.2 Section splitting

The unit of retrieval is the **section (ধারা)** — or **article (অনুচ্ছেদ)** for the Constitution. One section = one embedding = one citable answer.

The parser must extract, per section: the section number (in whatever numeral form the page uses), the section heading, the body text, and the enclosing chapter (অধ্যায়).

**Expect irregularity.** Some Acts have sub-sections `(১)`, `(২)`; some have provisos (শর্তাংশ); some have schedules (তফসিল) that are tables, not prose; some have repealed sections that are empty or marked `[বিলুপ্ত]`. Handle each explicitly:

- **Sub-sections stay with their parent section.** A citizen's question maps to a section, not a clause.
- **Repealed/omitted sections are dropped**, and the count of dropped sections is recorded. Retrieving a repealed provision would be an actual harm.
- **Schedules and tables are excluded from v1** unless one of your domains depends on them. Note the exclusion.
- **Per-Act overrides are fine.** A `configs/split_overrides.yaml` with special handling for two or three awkward Acts is better than a universal parser that quietly mangles them.

### 4.3 Long sections

Audit your section lengths first (§4.6). If a section exceeds roughly 400 words:

Split into overlapping sub-chunks (~250 words, ~50 word overlap), keeping **identical citation metadata** and adding a `sub_idx` field. Retrieval scores at sub-chunk level; the display layer deduplicates to section level and shows the best-matching passage with the full section available on expand.

Do not skip this. A 1,200-word section embedded as a single vector produces a mushy average that matches nothing well — it is one of the more common quiet causes of bad retrieval numbers.

### 4.4 The chunk schema

This is a **frozen contract** (§12.1). B, C, and D all build against it.

```python
# src/dhara/schema.py
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Chunk:
    chunk_id: str            # "labour_2006_s103" or "labour_2006_s103_p1"
    section_id: str          # "labour_2006_s103" — parent, for dedup
    act_id: str              # "labour_2006"
    act_name_bn: str
    act_name_en: str
    act_year: int
    chapter_bn: Optional[str]
    section_no_bn: str       # "১০৩" — as printed
    section_no_ascii: str    # "103" — normalized, for lookup
    section_title_bn: Optional[str]
    text_bn: str             # normalized body (light normalization)
    text_raw: str            # pre-normalization, for display
    domain: str              # family|land|labour|consumer|constitutional
    sub_idx: int             # 0 if not split
    n_sub: int               # 1 if not split
    source_url: str
    crawl_date: str          # ISO-8601
    n_words: int
```

**Two text fields, deliberately.** `text_raw` is what the UI shows the user — the law as printed, untouched. `text_bn` is what models consume. Never show a user normalized text; the ZWNJ you stripped might have been load-bearing for a conjunct, and displaying mangled Bangla to a Bangla speaker destroys trust in the whole system instantly.

**`section_no_ascii` is not optional.** Users write "শ্রম আইনের ১০৩ ধারা" and also "labour act section 103". Both must find the same chunk.

### 4.5 Bangla normalization

*This is a substantive methodology section in your report, not a utility function. Document every rule and why.*

The critical insight: **you need two normalization levels, not one.**

| Level | Used by | Rationale |
|---|---|---|
| **Light** | Transformers (bi-encoder, cross-encoder, BanglaBERT) | Their tokenizers were trained on natural text. Aggressive normalization moves your input off the distribution the model learned. |
| **Aggressive** | BM25, Word2Vec, TF-IDF | These match on surface forms. Every unnormalized variant is a missed match. |

Applying aggressive normalization to transformer inputs is a real and commonly-made mistake that costs measurable performance.

```python
# src/dhara/normalize.py
import re, unicodedata

ZWNJ, ZWJ = "\u200c", "\u200d"
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
DIGIT_MAP = {d: str(i) for i, d in enumerate(BN_DIGITS)}

def light(text: str) -> str:
    """For transformer inputs. Fix encoding inconsistencies only."""
    text = unicodedata.normalize("NFC", text)   # rule 1
    text = text.replace(ZWNJ, "")               # rule 2 — see note
    text = text.replace("\xa0", " ")            # rule 6
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def aggressive(text: str) -> str:
    """For lexical models. Collapse every variant that means the same thing."""
    text = light(text)
    text = text.replace(ZWJ, "")                # rule 2b
    text = "".join(DIGIT_MAP.get(c, c) for c in text)   # rule 4
    text = text.replace("।", " । ")             # rule 5
    text = re.sub(r"[^\u0980-\u09FF0-9a-zA-Z।\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
```

**The seven documented rules:**

1. **NFC normalization.** Bangla vowel signs and conjuncts have multiple valid byte encodings. Without this, visually identical strings compare as different — the single most consequential rule here.
2. **ZWNJ / ZWJ (U+200C / U+200D).** Inserted inconsistently in web text, completely invisible, and they silently break tokenization. ZWNJ is safe to strip. **ZWJ is not always safe** — it is load-bearing in some conjuncts such as র‍্য. Strip it only at the aggressive level, and verify on a sample. Document that you checked.
3. **নুক্তা (nukta) forms.** ড় ঢ় য় exist as precomposed characters (U+09DC, U+09DD, U+09DF) and as base + nukta (U+09BC). These three are Unicode *composition exclusions*, so NFC leaves them decomposed. That is fine — what matters is that you are **consistent** and that you state which form your pipeline produces. Verify that your model's tokenizer agrees.
4. **Digit normalization.** ১২৩ ↔ 123. **Critical for this project specifically**, because section numbers appear in both forms in both queries and text.
5. **দাঁড়ি (।, U+0964).** Bangla sentence terminator. Pad it as a separate token for lexical models so it does not glue to the preceding word.
6. **Scraping artifacts.** Non-breaking spaces, HTML entity leftovers, repeated newlines.
7. **Stemming — document the limitation honestly.** Available Bangla stemmers are weak and lossy on inflected legal vocabulary. Evaluate what exists, pick one or none, and **report what you found**. An honest negative result about tooling availability in a low-resource language is a legitimate contribution to your report, not an admission of failure.

**Write unit tests for this module.** Twenty hand-picked strings with expected outputs, run in CI. Normalization bugs are invisible — text looks fine and retrieval is quietly 5 points worse — and they are near-impossible to find in Week 7.

### 4.6 Corpus QA and freeze

*Blocking gate, end of Week 3. Nothing downstream is trustworthy without this.*

Run `scripts/03_build_corpus.py --qa` and produce `results/tables/corpus_stats.csv`:

- Total chunks, total sections, chunks per domain, chunks per Act
- Word-count distribution: min, median, p90, p95, max
- Count of chunks below 20 words (suspiciously short — probably parse errors)
- Count of chunks above 400 words (should be zero after sub-chunking)
- Duplicate detection: exact and near-duplicate text across chunks
- Count of chunks with missing `section_no`, missing `chapter`, or empty text
- Language sanity: proportion of Bangla codepoints per chunk (catches English blocks that slipped in)

**Then hand-verify 50 randomly sampled chunks.** Two people, independently, against the source website. Check: is the text complete? Is the section number right? Does the citation resolve to the correct page?

This takes an afternoon and it is the highest-value afternoon in the project. A systematic parse error found in Week 3 costs an hour to fix. Found in Week 7, it invalidates every number you have.

**Freeze criteria:**
- [ ] ≥ 800 chunks (or the reduced target set by §3.1)
- [ ] All four domains represented, none below 10% of the corpus
- [ ] Zero chunks with empty text or missing citation fields
- [ ] Zero chunks above 400 words
- [ ] 50-chunk manual audit passed with ≤ 2 errors
- [ ] `corpus_stats.csv` committed
- [ ] Tagged in git as `corpus-v1`

Once frozen: **`corpus_v1.jsonl` does not change.** If Week 5 reveals a problem, you create `corpus_v2.jsonl`, and you re-run every experiment or clearly label which corpus version produced which number.

---

## §5 · Phase 2 — Dataset construction

*Owner: B. Weeks 2–5. This is the critical path of the entire project.*

You have documents and no questions. Manufacturing the question–section pairs determines whether every model downstream looks good or bad. If your gold set is weak, the best fine-tuning in the world will produce numbers nobody can interpret.

Three sources, used together. Reporting how each contributes is itself a methodological contribution.

| Source | Volume | Used for | Owner |
|---|---|---|---|
| Mined real questions | 250–400 | Training + gold candidates | B |
| Synthetic (LLM-generated) | 2,000–2,500 | Training only | B |
| Hand-annotated | 200 | **Gold test only** | B leads, all 4 annotate |

### 5.1 Mining real questions

Sources: legal-aid organization FAQ pages, government service portal FAQ sections, published citizen-help material, and public discussion where people ask legal questions in their own words.

**Privacy handling is mandatory and non-negotiable:**
- Strip names, phone numbers, NID numbers, addresses, and any identifying detail before the text enters your dataset.
- Rewrite lightly to remove distinguishing specifics while preserving the *register* — the way people actually phrase things, which is the entire reason you want real questions.
- Do not republish verbatim with attribution to a person.
- Record source *type* (e.g. "legal aid FAQ"), not source *identity of the asker*.

State this handling explicitly in the report's ethics section. It is the kind of thing examiners notice when it is absent.

**What you are harvesting is register, not content.** A real question tells you that people say "জমি আমার নামে করব কীভাবে?" rather than "নামজারির পদ্ধতি কী?" — that distinction is your entire research problem, and no LLM will invent it reliably for you.

### 5.2 Synthetic generation

For each chunk, prompt an LLM to write 2–3 plausible citizen questions the section answers.

```
তুমি একজন সাধারণ বাংলাদেশি নাগরিক, যার আইন সম্পর্কে কোনো প্রাতিষ্ঠানিক
জ্ঞান নেই। নিচের আইনের ধারাটি পড়ো এবং এমন ৩টি প্রশ্ন লেখো যা এই ধারাটি
উত্তর দেয়।

নিয়ম:
১. প্রশ্নগুলো মুখের ভাষায় লেখো — যেভাবে গ্রামের একজন মানুষ বা একজন
   শ্রমিক কথা বলে।
২. ধারার আইনি শব্দ ব্যবহার করবে না। নিজের ভাষায় সমস্যাটা বর্ণনা করো।
   (যেমন: "গ্রেপ্তার" না লিখে "পুলিশ ধরে নিয়ে গেছে" লেখো)
৩. প্রতিটি প্রশ্ন একটি বাস্তব সমস্যা থেকে আসবে, সংজ্ঞা জিজ্ঞাসা নয়।
৪. প্রশ্নগুলো ১৫-৩০ শব্দের মধ্যে রাখো।
৫. শুধু প্রশ্ন তিনটি লেখো, প্রতিটি নতুন লাইনে। অন্য কিছু লিখবে না।

আইনের ধারা:
{section_title}
{section_text}
```

**Generate a variety split.** Roughly 70% colloquial (the prompt above) and 30% semi-formal (a variant instructing the model to write as an educated user who knows some legal vocabulary). Real users are not uniformly colloquial, and training only on one register produces a model that fails on the other.

Batch, cache by `chunk_id`, and checkpoint — you will re-run this and you should not pay for it twice.

### 5.3 The lexical-overlap audit

**Read this section twice. It is the single biggest threat to the validity of your results.**

An LLM asked to write a question about a section will tend to reuse that section's vocabulary, no matter what the prompt says. If you train and evaluate on such questions, you have accidentally built a lexical-overlap task. Your fine-tuned model will post excellent numbers. You will have measured nothing, and the entire premise of the project — that the lexical gap matters — will be untested.

```python
# src/dhara/synth.py
def jaccard(q: str, p: str) -> float:
    qs = set(aggressive(q).split())
    ps = set(aggressive(p).split())
    return len(qs & ps) / max(len(qs | ps), 1)

def content_overlap(q: str, p: str, stopwords: set) -> float:
    """Content-word overlap: what fraction of the question's content words
    appear in the passage. More diagnostic than raw Jaccard."""
    qs = set(aggressive(q).split()) - stopwords
    ps = set(aggressive(p).split()) - stopwords
    return len(qs & ps) / max(len(qs), 1)
```

**The audit procedure:**

1. Compute overlap for all synthetic pairs and for all mined real pairs.
2. Plot both distributions on the same axes. Put this figure in your report.
3. If synthetic overlap is much higher than real overlap, you have a problem. Fixes, in order of preference:
   - **Regenerate** with a stronger colloquial constraint and explicit banned terms drawn from the section itself.
   - **Filter**: discard synthetic pairs above an overlap threshold, e.g. content overlap > 0.5. Report how many you discarded and the threshold used.
   - **Report the discrepancy honestly** if you cannot fix it, and rely entirely on the gold set for headline claims.
4. Report the final overlap statistics for both sets in the report regardless of outcome.

This audit is genuinely a contribution — most course projects using synthetic data never check this, and being able to say you did, with a figure, materially raises the credibility of everything else you report.

### 5.4 Gold test set annotation

200 real questions, hand-mapped to correct sections. This is your only trustworthy measurement instrument.

**Write the guideline before annotating.** `docs/annotation_guideline.md`, covering:

- **What counts as correct.** The section that directly answers the question. Not a section that merely mentions the topic.
- **Multiple correct answers.** Allowed, as a ranked list of up to 3. Genuinely common in law — a procedural question may be answered by both a substantive provision and a procedural rule. Your metrics must handle multi-relevance (§6.1).
- **No answer exists.** Mark `unanswerable`. Keep roughly 10–15% of the gold set as unanswerable questions. These are essential for calibrating the abstention threshold (§8.3), and a system that cannot say "I don't know" is dangerous in this domain.
- **Ambiguous questions.** Escalate to group discussion; do not guess silently.
- **Worked examples.** Five annotated examples with reasoning, including two hard ones.

**Annotation procedure:**

1. B annotates 20 questions and circulates them as calibration examples.
2. All four annotate the same 50 questions independently. **Measure agreement** — Cohen's/Fleiss' κ on the top-1 section, and exact-match rate.
3. Discuss disagreements as a group. Update the guideline. This step is where the guideline actually gets written.
4. Split the remaining 150 across four people, ~38 each.
5. B does a final consistency pass over everything.

**Report the agreement number.** It costs one afternoon and it is what separates "we labelled some data" from "we constructed an evaluation resource." If κ is low, that is itself a finding worth discussing — it means the task is genuinely hard, which contextualizes your model's ceiling.

**Tag every gold question along three axes:**

```json
{
  "qid": "gold_0042",
  "question_bn": "পুলিশ আমাকে ধরে নিয়ে গেছে, কিছু বলছে না — আমার কী অধিকার?",
  "relevant_chunk_ids": ["constitution_a33"],
  "register": "colloquial",
  "domain": "constitutional",
  "source": "mined_faq",
  "answerable": true,
  "annotator": "B",
  "notes": ""
}
```

`register` ∈ {colloquial, formal} is the tag that produces your headline result. Get it right — assign it based on vocabulary used, not on who wrote it.

### 5.5 Intent labels

The `domain` field doubles as the intent classification target. Five classes: family, land, labour, consumer, constitutional. No extra annotation effort — it comes free with §5.4 and with the chunk metadata for training data.

For the classifier's training set, use the synthetic questions with their source chunk's `domain` as the label. This gives you ~2,500 labelled examples at zero cost. **Check class balance** and report it; if one domain dominates, use class weights rather than pretending balance exists.

### 5.6 Splits and freeze

```
train_pairs_v1.jsonl   ~2,500 pairs   (synthetic + 70% of mined)
dev_pairs_v1.jsonl     ~300 pairs     (synthetic + 30% of mined)
gold_test_v1.jsonl     200 questions  (hand-annotated, real only)
```

**Split by chunk, not by question.** If chunk X generated three synthetic questions, all three go to the same split. Otherwise a near-duplicate question appears in both train and dev, and your dev numbers become optimistic in a way that will mislead your model selection.

**Verify gold-set isolation.** Run a check that no gold question — and no near-duplicate of one — appears in train or dev. Fuzzy-match on normalized text. Assert it in code, in `scripts/09_run_eval.py`, so it fails loudly rather than silently.

**Freeze at end of Week 5**, tagged `dataset-v1`.

### 5.7 Hard negative mining

*Week 5, after the zero-shot dense baseline exists. B owns, D consumes.*

For each training question, retrieve the top 30 chunks using BM25 and the zero-shot dense model. Any retrieved chunk that is *not* the gold chunk is a candidate negative.

**Take negatives from ranks 5–30, not 1–4.** The top few are often genuinely relevant but unlabelled — training against them teaches the model that correct answers are wrong, which actively damages performance. This is a well-known failure mode with an unglamorous name (false negatives) and a large impact.

Prefer negatives that are **hard but clearly wrong**: sections from the same Act, or the same domain, that discuss adjacent matters. Those are what teach fine distinctions. A section about consumer rights is a useless negative for a question about inheritance — the model already gets that one right.

```json
{"qid": "syn_01823", "positive": "labour_2006_s103", 
 "negatives": ["labour_2006_s104", "labour_2006_s100", "labour_2006_s118"]}
```

Target 4–8 negatives per question. This single technique typically buys more than any hyperparameter tuning you will do.

---

## §6 · Phase 3 — Evaluation harness

*Owner: C. Built Week 1, before any model exists.*

**Build the measuring instrument before the thing being measured.** This is not pedantry — if D builds a bi-encoder and then C builds a metric, the metric will be unconsciously shaped by what the bi-encoder does, and comparisons across rungs stop being fair.

### 6.1 Metrics

Multi-relevance is required — gold questions may have up to 3 correct chunks (§5.4).

```python
# src/dhara/metrics.py
import numpy as np

def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant: return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Did we get at least one right in the top k? Often the more
    user-meaningful number when there is one obvious answer."""
    return float(bool(set(ranked[:k]) & relevant))

def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 1)
              for i, cid in enumerate(ranked[:k], start=1) if cid in relevant)
    ideal = sum(1.0 / np.log2(i + 1)
                for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
```

**Report both Recall@k and Hit@k.** With multi-relevance they diverge, and Hit@k ("did the user see a right answer on screen?") is closer to what the system is actually for.

### 6.2 The Retriever interface

Every rung implements the same interface. This is a **frozen contract** (§12.2) — C defines it in Week 1, D builds against it in Week 6.

```python
# src/dhara/retrievers/base.py
from abc import ABC, abstractmethod

class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted descending. Length <= k."""
```

Because every model conforms, `evaluate.py` never changes, and adding a rung is one file plus one config line.

### 6.3 The evaluation runner

`scripts/09_run_eval.py` takes a retriever name and a gold file, and emits one JSON per run to `results/runs/`:

```json
{
  "run_id": "biencoder_ft_seed42_20260930T1412",
  "retriever": "biencoder_ft",
  "corpus_version": "corpus_v1",
  "gold_version": "gold_test_v1",
  "seed": 42,
  "config": {"checkpoint": "...", "epochs": 3, "lr": 2e-5},
  "overall": {"recall@1": 0.41, "recall@5": 0.68, "hit@5": 0.72,
              "mrr@10": 0.52, "ndcg@10": 0.58},
  "by_register": {"colloquial": {...}, "formal": {...}},
  "by_domain": {"family": {...}, "land": {...}, ...},
  "per_query": [{"qid": "gold_0042", "ranked": [...], "rr": 0.5}, ...]
}
```

**`per_query` is not optional.** Error analysis (§10.2) needs it, significance testing needs it, and regenerating it in Week 8 means re-running everything.

The results table is then assembled by a script that reads `results/runs/*.json`. **No hand-typed numbers, ever.** Hand-copied numbers go stale the moment you re-run an experiment, and in Week 8 you will not know which table reflects which run.

### 6.4 Assert gold isolation

At the top of the evaluation runner:

```python
assert_no_leakage(gold_path, train_path, dev_path)
```

Fail loudly. This runs on every evaluation, forever.

---

## §7 · Phase 4 — Models

Every rung is a lab topic, and the ladder is the narrative arc of your results chapter.

### 7.1 Rung 1 — BM25

*Owner: C. Week 3.*

```python
from rank_bm25 import BM25Okapi

class BM25Retriever(Retriever):
    name = "bm25"
    def index(self, chunks):
        self.ids = [c.chunk_id for c in chunks]
        tokenized = [aggressive(c.text_bn).split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    def search(self, query, k=10):
        scores = self.bm25.get_scores(aggressive(query).split())
        top = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top]
```

**Tune `k1` and `b` on dev.** An untuned BM25 that loses to your neural model proves nothing; reviewers and examiners both know this. Also try indexing `section_title + text` versus `text` alone — titles are dense with legal terminology and often help.

**Expected finding:** competitive on formal-register queries, poor on colloquial ones. That split is your motivating result — lead the results chapter with it.

### 7.2 Rung 2 — Self-trained Word2Vec

*Owner: C. Week 4.*

Train skip-gram on the aggressive-normalized corpus.

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=corpus_tokens, vector_size=200, window=5,
                 min_count=3, sg=1, negative=10, epochs=30, workers=4)
```

`min_count=3` and `epochs=30` because your corpus is small — the defaults are tuned for far larger data.

**Two uses:**

**(a) Retrieval baseline.** Mean-pool word vectors, cosine similarity. Try IDF-weighted pooling as a variant; it usually helps and is one line.

**(b) The interpretability evidence — this is where the marks are.** Build a nearest-neighbour table for legal terms in *your* legal-trained model versus a general-Bangla-trained model (train one on Bangla Wikipedia, or use published Bangla vectors):

| Term | Neighbours (legal corpus) | Neighbours (general Bangla) |
|---|---|---|
| আটক | | |
| নামজারি | | |
| খারিজ | | |
| রিট | | |
| জামিন | | |

The neighbourhoods differ dramatically, and this table is the most direct, concrete, human-readable answer to "explain how you did your embeddings." It is a self-trained model whose learned semantics you can display and interpret line by line.

**Expect weak retrieval numbers** — 200k–400k words is small for Word2Vec. **Report this honestly as a corpus-size finding, with analysis.** Understanding why a method underperforms earns more than the method performing well.

### 7.3 Rung 3 — BiLSTM encoder

*Owner: C. Week 5.*

BiLSTM over frozen Word2Vec embeddings, trained with contrastive loss on your training pairs. Dual-encoder: same weights encode both question and section (shared encoder), mean-pool or max-pool the hidden states, cosine similarity, InfoNCE loss with in-batch negatives.

```
Embedding(frozen W2V, 200d) → BiLSTM(hidden=256, bidirectional) 
→ mean-pool → Linear(512→256) → L2 normalize
```

Batch size 32+ matters here: in-batch negatives mean batch size *is* the number of negatives.

This is the first rung that actually **learns the mapping** from citizen language to legal language rather than relying on static vectors. Expect a clear jump over Rung 2 and a clear gap below Rung 4. That gap is your empirical argument for contextual representations.

Also train the BiLSTM **intent classifier** here (§7.8).

### 7.4 Rung 4a — Zero-shot dense (the control)

*Owner: D. Week 5. Locked before fine-tuning begins.*

**This baseline is what makes your contribution claimable.** Without it, your result says "transformers are good," which everyone knows. With it, your result says "*our domain fine-tuning* is good," which is your actual thesis.

Benchmark three checkpoints zero-shot on dev, pick on evidence:

| Checkpoint | Notes |
|---|---|
| `intfloat/multilingual-e5-base` | Strong multilingual retriever. **Requires prefixes** — see below. |
| `sentence-transformers/LaBSE` | Built for cross-lingual sentence alignment; strong if you pivoted to cross-lingual (§3.1). |
| `BAAI/bge-m3` | Large, multilingual, strong; heavier to fine-tune on a T4. |

**The E5 prefix footgun.** The E5 family requires `"query: "` before questions and `"passage: "` before documents. Omitting them silently degrades performance substantially, and it is a genuinely common mistake. If you use E5, apply the prefixes at both index time and query time, and in *both* the zero-shot and fine-tuned runs — otherwise your comparison is between a crippled baseline and a working model, which is not a comparison.

Record the choice and the evidence in `DECISIONS.md`.

### 7.5 Rung 4b — Fine-tuned bi-encoder (the headline)

*Owner: D. Week 6.*

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer(CHECKPOINT)
model.max_seq_length = 256

# with hard negatives: (anchor, positive, negative) triplets
examples = [InputExample(texts=[q, pos, neg]) for q, pos, neg in triplets]
loader = DataLoader(examples, shuffle=True, batch_size=16)
loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=3,
    warmup_steps=int(0.1 * len(loader) * 3),
    optimizer_params={"lr": 2e-5},
    use_amp=True,
    evaluator=ir_evaluator,          # on dev, not gold
    evaluation_steps=200,
    save_best_model=True,
    output_path="models/biencoder_ft",
)
```

**Why `MultipleNegativesRankingLoss`:** it treats every other item in the batch as a negative, so effective negatives = batch_size − 1, plus your explicit hard negatives. It suits your data shape (question ↔ correct section) exactly and needs no negative sampling logic of its own.

**Starting hyperparameters** (tune on dev only):

| Parameter | Value | Note |
|---|---|---|
| max_seq_length | 256 | Check against your p95 section length |
| batch_size | 16 (grad accum to effective 32–64) | Bigger is genuinely better for MNRL |
| epochs | 2–4 | Small dataset overfits fast; watch dev |
| lr | 2e-5 | 1e-5 if unstable |
| warmup | 10% of steps | |
| fp16 | on | |

**Run three seeds and report mean ± std.** A single-seed improvement of 2 points is indistinguishable from noise, and reporting it as a result is the easiest thing for an examiner to challenge. This costs you one extra hour of compute.

**Save the pre-fine-tuning embeddings** of gold questions and their correct chunks before you start. You need them for Figure B (§10.3) and regenerating them later means reloading the base checkpoint and redoing the work.

### 7.6 Rung 5 — Cross-encoder reranker

*Owner: D. Week 7.*

Retrieve top-50 via hybrid fusion, then score each (query, chunk) pair *jointly* — the query and passage attend to each other, which a bi-encoder cannot do. Far more accurate, far slower per pair, but you only apply it to 50 candidates.

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=384)
scores = ce.predict([(query, chunk_text) for chunk_text in candidates])
```

Try it zero-shot first — multilingual rerankers often work respectably out of the box. Then fine-tune on your pairs with hard negatives and binary cross-entropy if time allows.

**Expected finding:** the largest single accuracy jump in the whole ladder. Report latency alongside accuracy (§8.4) — the accuracy/speed tradeoff is a real engineering finding, not a footnote.

### 7.7 Hybrid fusion

*Owner: D. Week 7.*

Reciprocal Rank Fusion — no score normalization needed, which is exactly why it is the right choice when combining BM25 scores with cosine similarities that live on incomparable scales.

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

Report BM25-only, dense-only, and hybrid separately. **Hybrid does not always win**, and if it does not, say so — an honest negative ablation is worth more than a fudged positive one.

### 7.8 Intent classification (parallel track)

*Owner: C. Weeks 5–6.*

Same task, four representations, one table:

| Model | Features | Type |
|---|---|---|
| Multinomial Naive Bayes | TF-IDF | **Generative** |
| Logistic Regression | TF-IDF | Discriminative |
| Logistic Regression | Word2Vec mean-pooled | Discriminative |
| BiLSTM | Word2Vec sequence | Discriminative |
| Fine-tuned BanglaBERT | contextual | Discriminative |

Metrics: accuracy, macro-F1, per-class F1, confusion matrix. Report macro-F1 as the headline — accuracy hides poor performance on minority classes.

Naive Bayes gives you the generative-vs-discriminative comparison at essentially zero cost, satisfying that requirement in about twenty lines.

**Use BanglaBERT (`csebuetnlp/banglabert`) here rather than for retrieval** — it is an ELECTRA discriminator, well-suited to classification, and it is the strongest Bangla-specific option. Remember its required normalizer (§2.2).

**Keep intent routing out of the default retrieval path.** Report "retrieval + intent filtering" as an *ablation*. A wrong intent prediction silently destroys retrieval for that query, and you want to measure that cost rather than absorb it invisibly.

### 7.9 Unsupervised track

*Owner: C. Week 6.*

K-Means over section embeddings (from the fine-tuned bi-encoder), k chosen by silhouette score across a sweep.

Then quantify against your `domain` labels:
- **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**
- A cluster × domain contingency table

A number answering "did the model discover the actual structure of the law?" is much stronger than a picture of coloured dots. Include the picture too, but lead with the number.

**Interesting things to look for and discuss:** clusters that cut *across* Acts (procedural provisions from different laws grouping together) are more interesting than clusters that recover Act boundaries, because the latter can be explained by shared vocabulary alone.

---

## §8 · Phase 5 — Inference service

*Owner: D. Week 7.*

The training code and the serving code are different programs with different requirements. Do not let the demo import from training scripts — build a clean service layer.

### 8.1 Index building

`scripts/08_build_index.py` runs once after training and produces everything the app needs:

```
models/index_v1/
├── chunk_ids.json          # row index → chunk_id
├── embeddings.npy          # (n_chunks, dim) float32, L2-normalized
├── bm25.pkl                # pickled BM25 index
├── chunks.jsonl            # display metadata, keyed by chunk_id
└── manifest.json           # model checkpoint, corpus version, build date
```

**Precompute and cache.** With ~1,000 chunks and a 768-dim model this is roughly 3 MB. Brute-force cosine over 1,000 vectors is sub-millisecond — **you do not need FAISS or any ANN index at this scale**, and adding one is complexity you will have to explain and debug for no measurable benefit. Say this explicitly in the report; knowing when *not* to reach for infrastructure is a real engineering judgment.

`manifest.json` records which model checkpoint and which corpus version produced the index. When the demo behaves oddly in Week 8, this is the first thing you check.

### 8.2 The query path

```python
# src/dhara/service.py
class DharaService:
    def __init__(self, index_dir, use_reranker=True):
        # load once at startup, never per-request
        ...

    def answer(self, query: str, top_k: int = 3) -> Result:
        q = light(query)                            # §4.5
        if self.model_family == "e5":
            q_embed_text = f"query: {q}"            # §7.4 — do not forget

        dense_ranking = self._dense_search(q_embed_text, k=50)
        sparse_ranking = self._bm25_search(aggressive(q), k=50)
        fused = rrf([dense_ranking, sparse_ranking])[:50]

        if self.use_reranker:
            fused = self._rerank(q, fused)

        deduped = self._dedupe_to_section(fused)    # §4.3 sub-chunks
        top = deduped[:top_k]

        if top[0].score < self.abstain_threshold:
            return Result(status="low_confidence", results=top)
        return Result(status="ok", results=top)
```

**Deduplicate to section level before display.** If a long section was split into three sub-chunks and all three rank highly, showing the user the same law three times looks broken. Keep the best-scoring sub-chunk as the highlighted passage, and offer the full section on expand.

### 8.3 The abstention threshold

**A legal-adjacent system that cannot say "I don't know" is dangerous.** This is not a nice-to-have.

Calibrate it properly using the unanswerable questions you deliberately included in the gold set (§5.4):

1. Score every gold question with the final pipeline.
2. Plot the score distribution for answerable vs. unanswerable questions.
3. Pick the threshold that gives an acceptable tradeoff — for this application, prefer **false abstentions over false confidence**. Wrongly telling someone "I don't have a confident match for this" is a minor annoyance; confidently returning the wrong law is the actual harm you are guarding against.
4. Report the threshold, the resulting precision/recall on the unanswerable subset, and your reasoning.

When abstaining, the UI must be specific about what happened and what to do next, not vague (§9.4).

### 8.4 Latency budget

Measure and report, on the hardware you actually demo on:

| Stage | Target | Typical |
|---|---|---|
| Query embedding | < 50 ms | |
| Dense search (1k chunks) | < 5 ms | |
| BM25 search | < 20 ms | |
| RRF fusion | < 1 ms | |
| Cross-encoder rerank (50 candidates) | 300–800 ms | dominant cost |
| **Total** | **< 1 s** | |

The reranker is the whole latency budget. Report accuracy *and* latency for pipelines with and without it — the tradeoff is a genuine engineering finding, and "our best model is 15× slower for 8 points of Recall@5" is exactly the kind of honest analysis that reads well.

If total latency exceeds ~1.5s, rerank the top 20 instead of 50 and report the cost.

---

## §9 · Phase 6 — The interface

*Owner: D. Week 8 (skeleton exists from Week 1).*

Sir asked only for a clear input→output system. But this is the artifact people will remember, and it costs one day to make it good rather than default. It is also the only part of the project a non-technical examiner can evaluate directly.

### 9.1 Design direction

**Pin the subject first.** This is not a chatbot and should not look like one. It is a *legal reference instrument* — the visual world it belongs to is the printed gazette, the certified copy, the margin annotation, the section marker. The user's emotional state is usually anxiety about a real problem. The interface's job is to look **calm, authoritative, and honest about its own limits.**

That rules out the two obvious defaults: a chat bubble UI (implies conversation and advice, which is exactly what this system does not offer) and a search-engine results page (implies ten blue links of equal weight, when you want to present three citations with visible confidence).

**Token system:**

| Role | Value | Reasoning |
|---|---|---|
| Ink | `#16233A` | Deep official-document navy. Primary text and headings. |
| Paper | `#FAFAF8` | Near-white with warmth, not cream. Document, not notepad. |
| Rule | `#D8D6D0` | Hairlines and dividers. |
| Seal | `#A63D40` | Muted brick red. **Citations only.** Never a button, never decoration. |
| Jade | `#2F6F5E` | Confidence indicator, high match. |
| Amber | `#B57F2A` | Low confidence, abstention. |

Spend all your boldness on the seal red, used in exactly one place, and keep everything else quiet.

**Typography:**
- **Display / section numbers:** Noto Serif Bengali — the serif carries document authority, and it renders Bangla conjuncts cleanly at large sizes.
- **Body:** Hind Siliguri or Anek Bangla — high legibility for long legal passages, which is what users spend their time reading.
- **Utility (scores, metadata):** any clean monospace with Latin coverage.

**Bangla web typography is a real requirement, not a detail.** Test your font stack on actual legal text with dense conjuncts (কর্তৃপক্ষ, দ্রষ্টব্য, রাষ্ট্রপতি) before committing. A font that mangles conjuncts destroys credibility with your actual users instantly, and Bangla is unforgiving here in a way Latin scripts are not. Set `line-height` around 1.9 — Bangla needs more leading than Latin because of the মাত্রা and the descender-heavy conjuncts.

**Signature element — the citation seal.** Each result carries its citation set as a bordered block in the left margin: Act name small above, **ধারা number set large in Bangla numerals**, chapter below. It reads as a stamp on a certified document. This is the one memorable thing in the interface, and it happens to encode the project's whole thesis: the output is not an answer, it is *a pointer to an authority you can verify*.

### 9.2 Layout

```
┌──────────────────────────────────────────────────────────┐
│  ধারা                                    crawl: 2026-08-20│
│  আপনার প্রশ্নের সাথে সম্পর্কিত আইনের ধারা খুঁজুন            │
│  ──────────────────────────────────────────────────────  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ আপনার প্রশ্ন লিখুন...                              │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                       [ খুঁজুন ]           │
│                                                          │
│  উদাহরণ:  [পুলিশ ধরে নিয়ে গেছে]  [জমি নামজারি]  [ছুটি]    │
│  ──────────────────────────────────────────────────────  │
│                                                          │
│  ┌────────┬─────────────────────────────────────────┐    │
│  │শ্রম আইন│ সাপ্তাহিক ছুটি               ●  উচ্চ মিল  │    │
│  │        │                                          │    │
│  │  ১০৩   │ কোনো শ্রমিককে... [আইনের মূল পাঠ]        │    │
│  │        │                                          │    │
│  │নবম     │ ▸ সম্পূর্ণ ধারা দেখুন    ↗ মূল উৎস       │    │
│  │অধ্যায়  │                                          │    │
│  └────────┴─────────────────────────────────────────┘    │
│                                                          │
│  [ two more results, same structure ]                    │
│  ──────────────────────────────────────────────────────  │
│  ⚠ এটি আইনি পরামর্শ নয়। এটি শুধুমাত্র আইনের ধারা খুঁজে   │
│    দেখায়। আইন সংশোধিত হতে পারে।                          │
└──────────────────────────────────────────────────────────┘
```

**Never truncate the legal text into an ellipsis with no way to see the rest.** Show a meaningful excerpt with the matched region emphasized, and make the full section one click away. A user who cannot read the whole provision cannot verify anything, which defeats the purpose.

**Always show `text_raw`, never `text_bn`** (§4.4). Displaying normalized text to a Bangla reader shows them mangled script.

### 9.3 Gradio implementation

```python
# src/app/app.py
import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;700&family=Hind+Siliguri:wght@400;600&display=swap');

:root {
  --ink:#16233A; --paper:#FAFAF8; --rule:#D8D6D0;
  --seal:#A63D40; --jade:#2F6F5E; --amber:#B57F2A;
}
.gradio-container { background: var(--paper) !important; max-width: 860px !important; }
body, .prose { font-family: 'Hind Siliguri', system-ui, sans-serif; color: var(--ink); }
h1, .section-no { font-family: 'Noto Serif Bengali', serif; }

.result { display:flex; border:1px solid var(--rule); background:#fff;
          border-radius:2px; margin-bottom:14px; overflow:hidden; }
.seal   { flex:0 0 108px; border-right:2px solid var(--seal);
          padding:14px 10px; text-align:center; background:#FDFBFA; }
.seal .act    { font-size:.72rem; color:var(--ink); opacity:.7; line-height:1.5; }
.seal .num    { font-family:'Noto Serif Bengali',serif; font-size:2rem;
                color:var(--seal); font-weight:700; margin:6px 0; }
.seal .chap   { font-size:.68rem; opacity:.6; line-height:1.5; }
.body   { padding:14px 18px; flex:1; }
.body .title  { font-weight:600; margin-bottom:8px; }
.body .text   { line-height:1.9; font-size:.95rem; }
.conf   { float:right; font-size:.75rem; }
.conf.high { color:var(--jade); } .conf.low { color:var(--amber); }
.disclaimer { border-top:1px solid var(--rule); margin-top:24px; padding-top:12px;
              font-size:.82rem; color:var(--amber); line-height:1.8; }
"""

def render(results, status):
    if status == "low_confidence":
        return ("<div class='disclaimer'>আপনার প্রশ্নের সাথে নিশ্চিতভাবে মেলে "
                "এমন কোনো ধারা পাওয়া যায়নি। প্রশ্নটি অন্যভাবে লিখে দেখুন, "
                "অথবা নিচের কাছাকাছি ফলাফলগুলো দেখুন।</div>") + render(results, "ok")
    html = ""
    for r in results:
        cls = "high" if r.score > 0.6 else "low"
        label = "উচ্চ মিল" if cls == "high" else "সম্ভাব্য মিল"
        html += f"""
        <div class='result'>
          <div class='seal'>
            <div class='act'>{r.act_name_bn}</div>
            <div class='num'>{r.section_no_bn}</div>
            <div class='chap'>{r.chapter_bn or ''}</div>
          </div>
          <div class='body'>
            <span class='conf {cls}'>● {label}</span>
            <div class='title'>{r.section_title_bn or ''}</div>
            <div class='text'>{r.excerpt_html}</div>
            <div style='margin-top:10px;font-size:.8rem'>
              <a href='{r.source_url}' target='_blank'>↗ মূল উৎস</a>
            </div>
          </div>
        </div>"""
    return html

with gr.Blocks(css=CSS, title="ধারা") as demo:
    gr.Markdown("# ধারা\nআপনার প্রশ্নের সাথে সম্পর্কিত আইনের ধারা খুঁজুন")
    q = gr.Textbox(label="", placeholder="আপনার প্রশ্ন লিখুন...", lines=3)
    btn = gr.Button("খুঁজুন", variant="primary")
    gr.Examples(
        examples=["পুলিশ আমাকে ধরে নিয়ে গেছে, আমার কী অধিকার?",
                  "বাবার জমি ভাইদের মধ্যে ভাগ করব কীভাবে?",
                  "মালিক ছুটি দিচ্ছে না, কী করতে পারি?"],
        inputs=q)
    out = gr.HTML()
    gr.Markdown("⚠ **এটি আইনি পরামর্শ নয়।** এটি শুধুমাত্র প্রাসঙ্গিক আইনের ধারা "
                "খুঁজে দেখায়। আইন সংশোধিত হতে পারে — মূল উৎস যাচাই করুন।",
                elem_classes="disclaimer")
    btn.click(fn=handle, inputs=q, outputs=out)
    q.submit(fn=handle, inputs=q, outputs=out)
```

**Load the model once at module level, not inside the handler.** Reloading a transformer per request makes the demo feel broken, and it is the single most common Gradio mistake.

**`q.submit` matters** — people press Enter. If Enter does nothing, the interface feels dead regardless of how it looks.

### 9.4 Interface copy

Words are design material here. Three rules:

- **Errors and abstentions give direction, not apology.** Not "দুঃখিত, কিছু পাওয়া যায়নি" but "নিশ্চিতভাবে মেলে এমন ধারা পাওয়া যায়নি। প্রশ্নটি অন্যভাবে লিখে দেখুন।" Say what happened and what to do next.
- **The empty state is an invitation.** The three example questions are the most important copy in the interface — they teach users, in one glance, that they can write in ordinary language. Choose examples that are visibly colloquial. This is your thesis, demonstrated in the UI.
- **The disclaimer is permanent and visible without scrolling to it.** Not a modal the user dismisses, not a footnote. It is part of the product.

### 9.5 Quality floor

Not optional, and cheap:
- Responsive down to a phone width — the realistic user for this system is on a phone.
- Visible keyboard focus rings.
- Sufficient contrast (the seal red on white passes; check anything else you add).
- Loading state on the button so a 1-second rerank does not look like a hang.
- `lang="bn"` on the document, so screen readers and font fallback behave.

---

## §10 · Phase 7 — Figures, error analysis, report

### 10.1 Results tables

All auto-generated from `results/runs/*.json`. Four tables:

1. **Main ladder** — five configurations × {Recall@1, Recall@5, Hit@5, MRR@10, nDCG@10}, with mean ± std over 3 seeds for the fine-tuned rows.
2. **Register split** — the same, split colloquial vs. formal. **This is the paper's central table.** The prediction: BM25's colloquial–formal gap is large, the fine-tuned bi-encoder's gap is small. That contraction is the measured contribution.
3. **Per-domain** — which areas of law are hardest, with a hypothesis for why.
4. **Classification** — five models × {accuracy, macro-F1}, plus the confusion matrix for the best.

### 10.2 Error analysis

*Owner: A and C, Week 7. Do not skip — this is one of the most-rewarded and least-often-done sections in a project report.*

Take the 40 lowest-MRR gold questions under your best model. Read every one. Categorise the failure:

| Category | Meaning | What it implies |
|---|---|---|
| Annotation error | The "wrong" answer was actually acceptable | Your ceiling is lower than reported — quantify this |
| Genuine ambiguity | Several sections plausibly apply | A multi-answer UI is the right response, not a better model |
| Lexical gap unclosed | Colloquial phrasing still missed | More/better training pairs for that domain |
| Section too long | The correct answer was buried in a large chunk | Chunking strategy issue (§4.3) |
| Near-duplicate provisions | Similar sections across Acts, wrong one retrieved | Needs metadata or reranking signal |
| Out of scope | No correct answer exists in the corpus | Abstention should have fired — check the threshold |

Report counts per category. **If annotation errors turn out to be a meaningful fraction, say so and adjust your reported ceiling.** That honesty is worth more marks than the points it costs.

### 10.3 The two figures

**Figure A — the ladder.** Grouped bar chart, Recall@5 across all five rungs, split colloquial vs. formal, error bars from the seed runs. Your whole project in one image.

**Figure B — embedding geometry before and after fine-tuning.** Two panels, side by side. Embed the 200 gold questions and their correct chunks with the base checkpoint (left) and the fine-tuned model (right); project with t-SNE or UMAP; draw a faint line connecting each question to its correct chunk.

Left panel: long lines everywhere, question–answer pairs scattered apart. Right panel: lines collapsed short, pairs sitting together.

**Quantify it so the figure is not purely visual:** report the mean cosine similarity between gold questions and their correct chunks, before and after. A single number — "mean question–answer similarity rose from 0.31 to 0.68" — makes the picture into evidence.

Use the same projection parameters and random seed for both panels, or the comparison is not valid. State the seed in the caption.

This figure is the most persuasive answer to "explain how you did your embeddings" that you can put in front of an examiner, and it is your closing slide.

### 10.4 Report structure

| Section | Pages | Owner |
|---|---|---|
| Abstract, introduction, motivation | 2 | D |
| Related work | 1.5 | B |
| Data collection: sources, scraping, crawl date, splitting, schema | 3 | **A** |
| Preprocessing: the seven Bangla rules, two-level design, tooling limits | 2 | **A** |
| Dataset construction: three sources, synthetic prompt, overlap audit, annotation protocol, κ | 3 | **B** |
| Methodology: the five rungs, architecture, training setup | 4 | C + D |
| Evaluation protocol: metrics, splits, seeds, leakage checks | 1.5 | **C** |
| Results: four tables, two figures | 4 | C + D |
| Error analysis | 2 | A + C |
| Ethics, limitations, disclaimer | 1.5 | B |
| Conclusion, future work | 1 | D |

Everyone writes their own section **as they finish that phase**, not in Week 8. `DECISIONS.md` (§2.5) is the raw material — a decision recorded in Week 3 with its reasoning becomes two sentences of methodology in Week 8; the same decision unrecorded becomes a guess.

---

## §11 · Week-by-week timeline

| Wk | Milestone | **A** Data | **B** Dataset | **C** Classical + Eval | **D** Transformers + System |
|---|---|---|---|---|---|
| **1** | **Feasibility gate + toy pipeline runs** | Act survey (§3.1); scrape 3 Acts; splitter v0 | Annotation guideline draft; 30 toy questions | `metrics.py`; `Retriever` interface; eval runner | Repo, configs, CI; Gradio skeleton with hardcoded output |
| **2** | Corpus draft; generation pipeline works | Full scrape 4 domains + Constitution; normalization module + unit tests | Synthetic prompt built + tested on 50 chunks; mining begins | BM25 implemented; tuned on toy set | Benchmark 3 zero-shot checkpoints on toy questions |
| **3** | **`corpus_v1` FROZEN** | Sub-chunking; corpus QA; 50-chunk manual audit; freeze + tag | Synthetic generation at scale (~2,500); **overlap audit + figure** | **BM25 baseline on real corpus — first real numbers** | Fine-tuning script running end-to-end on dev |
| **4** | **Gold set complete** | Begin data-collection chapter | Lead annotation; 50-question agreement round; κ reported | Word2Vec trained; NN tables; W2V retrieval baseline | Hard-negative mining infrastructure |
| **5** | **`dataset_v1` FROZEN**; neural baselines | Preprocessing chapter; corpus QA pass | Splits + leakage check; freeze; intent labels; hard negatives mined | BiLSTM encoder + BiLSTM/NB/LR intent classifiers | **Zero-shot dense baseline LOCKED** (the control) |
| **6** | **Headline result** | Methodology review; begin error analysis prep | Dataset chapter written | Clustering + ARI/NMI; classification table | **Fine-tuned bi-encoder, 3 seeds** |
| **7** | Full system assembled | **Error analysis, 40 cases** | Ethics + limitations chapter | Results tables finalised; significance check | Cross-encoder; RRF fusion; index build; service layer; **Figures A & B** |
| **8** | **Ship** | Report assembly + proofread | Related work; abstract review | Results chapter; slides | UI polish; abstention calibration; latency table; demo rehearsal |

### Hard gates

Do not pass these. If you are behind, cut scope — never cut evaluation quality.

- **End Week 1:** `09_run_eval.py` prints a Recall@5 number on the toy set. If not, you have an integration problem, and it will only get more expensive.
- **End Week 3:** `corpus_v1` frozen and tagged. Slipping here cascades into everything.
- **End Week 4:** gold set complete with agreement reported. **If this slips, cut to 3 domains and 150 gold questions.** Do not cut the agreement round.
- **End Week 6:** fine-tuned bi-encoder beats the zero-shot control on dev. If it does not, you have two weeks to debug rather than two days.

### What to cut, in order

If Week 5 arrives and you are behind, drop in this order and record each cut in `DECISIONS.md`:

1. Generation layer — already excluded, keep it excluded
2. Cross-encoder fine-tuning (use it zero-shot)
3. BiLSTM *retrieval* rung (keep BiLSTM classification — syllabus coverage survives)
4. Third and second training seeds (report single-seed, say so explicitly)
5. One domain, dropping to three
6. Clustering track

**Never cut:** the gold set, the agreement round, the zero-shot control, the register split, the overlap audit. Those five are what make the numbers mean anything.

---

## §12 · Integration contracts

Four people working in parallel need frozen interfaces and mock data. Nobody should ever be blocked waiting for someone else's artifact.

### 12.1 File contracts

Every file below has a fixed schema, agreed in Week 1, versioned in the filename.

| File | Producer | Consumers | Frozen |
|---|---|---|---|
| `corpus_v1.jsonl` | A | B, C, D | End Wk 3 |
| `train_pairs_v1.jsonl` | B | C, D | End Wk 5 |
| `dev_pairs_v1.jsonl` | B | C, D | End Wk 5 |
| `gold_test_v1.jsonl` | B | C | End Wk 5 |
| `hard_negatives_v1.jsonl` | B | D | End Wk 5 |
| `results/runs/*.json` | C, D | C (tables) | Append-only |
| `models/index_v1/` | D | D (app) | Wk 7 |

**Mock-first rule.** In Week 1, A commits `corpus_mock.jsonl` with 20 hand-written chunks in the exact final schema, and B commits `gold_mock.jsonl` with 10 questions. C and D build entirely against the mocks. When the real files land in Week 3 and Week 5, swapping a path in `configs/paths.yaml` is the entire integration cost.

This one practice is worth more than any other coordination technique on a four-person project.

### 12.2 Code contracts

- **`Retriever`** (§6.2) — C defines it Week 1. Every rung implements it. `evaluate.py` never learns about specific models.
- **`Chunk`** (§4.4) — A defines it Week 1. Everyone imports it from `schema.py`; nobody redefines fields locally.
- **`normalize.light` / `normalize.aggressive`** (§4.5) — A owns. Anyone who needs different behaviour asks A rather than writing their own, or you will end up with three subtly divergent normalizations and irreproducible numbers.

### 12.3 Weekly rhythm

- **Monday, 30 min:** each person states their week's deliverable and what they need from whom.
- **Thursday, 20 min:** blockers only. Anyone blocked more than 24 hours escalates here.
- **Friday:** merge to `main`. Every week. No exceptions.
- **Anything merged to `main` must run.** A broken `main` blocks three people.

---

## §13 · Definition of done, and what usually goes wrong

### 13.1 Phase checklists

**Corpus (A)** — ≥ 800 chunks · four domains, none under 10% · zero empty or uncited chunks · zero chunks over 400 words · 50-chunk manual audit ≤ 2 errors · `corpus_stats.csv` committed · normalization unit tests pass · tagged `corpus-v1`

**Dataset (B)** — ≥ 2,000 training pairs · 200 gold questions (10–15% unanswerable) · κ reported on a 50-question overlap · overlap audit figure produced · leakage assertion passes · split by chunk not question · tagged `dataset-v1`

**Evaluation (C)** — all five metrics implemented and unit-tested · every rung behind the `Retriever` interface · `per_query` recorded in every run · tables auto-generated, zero hand-typed numbers · leakage assertion in the runner

**Models (C, D)** — five rungs evaluated on identical gold data · zero-shot control locked before fine-tuning · 3 seeds on the headline result · E5 prefixes applied consistently if using E5 · hyperparameters tuned on dev only

**System (D)** — index builds from a script and is reproducible · model loaded once at startup · sub-chunks deduped to sections · abstention threshold calibrated on unanswerable questions · latency measured on demo hardware · raw text displayed, never normalized text

**Report (all)** — every number traceable to a run JSON · four tables, two figures · error taxonomy with counts · disclaimer in abstract, UI, and limitations · crawl date stated

### 13.2 Failure modes, ranked by how often they actually happen

| Symptom | Likely cause | Fix |
|---|---|---|
| Fine-tuned model barely beats zero-shot | Synthetic questions lexically overlap sources — the model learned nothing new | Run the overlap audit (§5.3). This is the #1 cause. |
| Retrieval numbers implausibly high | Gold leaked into training, or split by question instead of chunk | Run the leakage assertion (§5.6) |
| Dense model much worse than expected | E5 prefixes missing, or aggressive normalization applied to transformer input | §7.4 and §4.5 |
| Bangla renders as boxes or broken conjuncts | Font stack lacks Bangla coverage, or normalization ran on display text | §9.1; display `text_raw` only |
| Results not reproducible between runs | Seed unset, or corpus changed silently after some runs | Freeze discipline (§0 rule 2) |
| Training loss goes to zero immediately | In-batch negatives too easy — batch too small, or duplicate texts in batch | Raise batch size, dedupe, add hard negatives |
| Word2Vec neighbours are nonsense | Corpus too small, `min_count` too high, too few epochs | Expected at this scale — report it as a finding (§7.2) |
| Demo feels frozen for seconds | Model reloading per request | Load at module level (§9.3) |
| Merge conflicts in the eval harness | Multiple people editing `metrics.py` | C owns it exclusively (§12.2) |
| Methodology chapter is vague in Week 8 | `DECISIONS.md` was never maintained | Unfixable retroactively — maintain it from Day 1 |

### 13.3 The one thing to get right

**Build the whole pipeline end-to-end in Week 1 at toy scale, then scale each part.**

Teams fail this kind of project by perfecting a scraper for five weeks and discovering in Week 6 that the eval harness has a bug. A working ugly pipeline in Week 1 turns every later problem from a crisis into an incremental fix.

And know where the real bottleneck is. It is not the corpus, and it is not the model. It is the **question–section pairs**. If those are weak, the best fine-tuning in the world produces numbers nobody can interpret — and interpretable numbers are the entire point.
