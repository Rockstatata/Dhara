# Dhara — technical deep dive: every NLP task, the code that does it, and why

Written 2026-09-19, for defending implementation-level questions (exact algorithms, tensor
shapes, hyperparameters, why a specific design was chosen over the obvious alternative). Every
code excerpt below is quoted from the actual file at the cited path/line, not reconstructed from
memory. `docs/PROJECT_REPORT.md` is the results-and-narrative companion to this document; this
one is the "how does it actually work" reference. Where the two disagree on a number, this
document and `results/runs/*.json` are authoritative — this one was written after the latest
classification and demo fixes.

---

## 1. Corpus construction and question collection

### 1.1 Scraping the law (`src/dhara/scrape.py`)

The bdlaws government portal (`http://bdlaws.minlaw.gov.bd` — HTTP only, port 443 refuses
connections) is scraped for provisions not already covered by BLAD, a published dataset the
project reuses for the bulk of the corpus. Every fetch is **archive-first**:

```python
# scrape.py:63-73 (paraphrased structure)
if dest_path.exists():
    html = dest_path.read_text(...)   # never re-fetch what's already on disk
else:
    html = _fetch(url); dest_path.write_text(html, ...)
```

This is what makes a multi-hour crawl **resumable** after a network failure — re-running the
script only fetches what's missing. `DELAY = 1.5` seconds between requests, a real
`User-Agent` identifying the crawler and contact email (`Dhara-Research-Crawler/1.0 (student
NLP project; sarwad038@gmail.com)`), and `crawl_date` auto-stamped per page (`date.today()`) —
this is what lets the UI later say "law as of this date," a real ethics requirement (laws
change; an undated citation is misleading).

**A real bug this caught, worth knowing the story of**: the original retry logic caught only
`requests.HTTPError` (e.g. 404) and let `ConnectionResetError` (wrapped in
`requests.RequestException`) escape uncaught. Because `acts()` accumulates provisions into a
list *inside* its `try` block, a dropped connection mid-Act silently discarded every provision
fetched so far for that Act with no error — this zeroed out whole Acts (শিশু আইন, কপিরাইট আইন)
on 2026-08-25 before being found and fixed. The fix: catch `RequestException` broadly, retry 3×
with linear backoff (`5.0 * attempt` seconds), and only treat `HTTPError` as terminal (a 404
really is "this page doesn't exist," not "try again").

Provision pages are parsed with BeautifulSoup + lxml, pulling two CSS-selected fragments:

```python
# scrape.py:86-103
soup = BeautifulSoup(html, "lxml")
head = soup.select_one(".txt-head")      # -> provision title (BLAD doesn't have one)
body = soup.select_one(".txt-details")   # -> text_raw
```

### 1.2 Turning HTML into `Chunk`s — provision parsing, chunking, repeal detection (`src/dhara/blad.py`)

Two builders (`build()` from BLAD's JSON, `build_from_bdlaws()` from the scraped JSONL), unioned
in `scripts/03_build_corpus.py`, both emitting `Chunk` objects.

**Provision-number extraction** uses a regex family that has to handle three different notations
found in the source text simultaneously — a footnote-marker prefix, the number itself (Bangla or
ASCII digits, optionally with a trailing letter like "১০৩ক"), and the sentence-ending danda:

```python
# blad.py:46-56
PROVISION_HEAD = re.compile(
    r"^\s*\.?\s*(?:(\d+)\[)?\s*([০-৯0-9]+\s*[-–]?\s*[ক-হA-Za-z]{0,3})\s*[।৷.]"
)
CONTINUATION = re.compile(r"^\s*(?:\d+\[)?\s*\(([০-৯0-9]+[ক-হa-z]?)\)")
```

A record that opens with a bare `(1)`/`(2)` marker (`CONTINUATION`) is not a new provision — it's
a sub-clause continuing the previous one, and is merged back into it. Missing this merge silently
dropped 153 provisions of the Code of Criminal Procedure in an earlier version, found by checking
the parsed provision count against a manual spot-count.

**Chunking — why 400/250/50, not an arbitrary "split at 512 tokens"**:

```python
# blad.py:64-66, 128-135
MAX_WORDS = 400
SUB_WORDS = 250
SUB_OVERLAP = 50

def _sub_chunk(words: list[str]) -> list[list[str]]:
    if len(words) <= MAX_WORDS:
        return [words]
    out, start = [], 0
    while start < len(words):
        out.append(words[start : start + SUB_WORDS])
        start += SUB_WORDS - SUB_OVERLAP      # advances 200 words per window
    return out
```

A provision ≤400 words stays one chunk (the unit a citizen would actually be pointed to — "ধারা
৪২০" is one thing, not three fragments). Longer provisions (some sections of the Penal Code, Code
of Civil Procedure run to 1000+ words) are split into 250-word windows advancing by 200 words, so
consecutive windows share a 50-word overlap — this exists so a sentence that happens to fall
exactly on a naive split boundary is still whole inside at least one window, instead of being cut
in half and made unembeddable by either half alone. `chunk_id` becomes `{provision_id}_p{i}` when
split; `provision_id` (never split) is what `schema.py`'s dedup and every eval script scores
against — a hit on *any* sub-chunk of the right provision counts as a correct retrieval, so a
long provision isn't unfairly penalized by existing as multiple candidate vectors.

**Repealed-law detection runs at two independent levels**, because a single check missed real
cases:

- **Act level** — BLAD's own title field silently omits the repeal marker for many Acts; the
  *portal's* title is checked instead:
  ```python
  # blad.py:347-363
  EXCLUDE_TITLE = re.compile(r"সংশোধন|ইনষ্টিটিউট|\(Amendment\)|\[রহিত\]|\[Repealed\]", re.I)
  ```
  This one regex caught 215 repealed Acts (7,849 chunks — two-thirds of everything initially
  tagged `cybercrime`) that BLAD's title alone would have let straight into the corpus.
- **Provision level**: `EXCLUDE_PROVISION = re.compile(r"\[বিলুপ্ত\]|\[Omitted\]")` drops
  individually-omitted sections inside an otherwise-live Act.

Retrieving a repealed provision as if it were current law is the one failure mode in this project
classified as **actual harm**, not just a lower score — this is why the check is duplicated
rather than trusted to one source.

**Deduplication runs last** (`blad.py:479-514`), after BLAD and bdlaws are merged, because
`chunk_id` is the literal key the gold set's `relevant_chunk_ids` points at — an identical-text
collision (the same provision appearing in both sources) is dropped; a *different-text* collision
(e.g. an amendment clause parsed to the same provision number as the substantive section it
amends) is kept and suffixed `_d2`, `_d3`, never silently overwritten.

### 1.3 `Chunk` — the frozen schema every downstream consumer imports (`src/dhara/schema.py:26-56`)

```python
@dataclass
class Chunk:
    chunk_id: str; provision_id: str; act_id: str
    act_title_bn: str; act_title_en: str; act_year: Optional[int]; act_no: str
    provision_kind: str        # "section" (ধারা) | "article" (অনুচ্ছেদ)
    provision_no_bn: str; provision_no_ascii: str
    provision_title_bn: Optional[str]
    text_bn: str                 # light-normalized — what models consume
    text_raw: str                 # untouched — the only thing ever shown to a reader
    domain: str; risk_tier: str
    sub_idx: int; n_sub: int
    source_url: str; source_dataset: str; crawl_date: str
    ...
    def citation(self) -> str:
        kind = "অনুচ্ছেদ" if self.provision_kind == "article" else "ধারা"
        return f"{self.act_title_bn}, {kind} {self.provision_no_bn}"
```

Two fields carry the same information in two forms on purpose: `provision_no_bn` ("১০৩", as
printed, for display) and `provision_no_ascii` ("103", via `str.translate` over a
Bangla-digit-to-ASCII map, `schema.py:19-23`) — so a citizen typing "section 103" and one typing
"১০৩ ধারা" resolve to the same lookup key. `text_bn` vs `text_raw` is the same
display-vs-model-input split carried through every stage of the project (§2.1).

### 1.4 Mining real citizen questions (`src/dhara/mine.py`, `src/dhara/questions.py`)

Four sources, each cleared individually against robots.txt / ToS / AI-training-signal headers
**before** any scraping, and three sources explicitly rejected with the reason on record
(`mine.py`'s own header): Jagonews24 (`Content-Signal: ai-train=no`), Quora (robots.txt forbids
AI training), Reddit (`Disallow: /`) — Facebook is excluded entirely (ToS + API removal, manual
collection only, never automated).

- **Prothom Alo** (পাঠকের উকিল / পাঠকের প্রশ্ন) — via its public Quintype JSON API, no HTML
  scraping needed at all.
- **Ajker Patrika** — server-rendered HTML, one question per article, literally followed by
  `উত্তর:` in the source markup (used as the split boundary).
- **Lawyers Club Bangladesh** — WordPress REST API (open), editorial explainers rather than
  reader letters, so only usable when the headline itself is phrased as a question.
- **The Daily Star** (Your Advocate, English) — the only source with labour/tax/consumer
  coverage, kept despite being English because it fills a real coverage gap.

**Splitting raw article text into individual questions** uses per-source boundary heuristics
with a confidence label attached to every result (`questions.py:57-70`): `split_confidence ∈
{high, medium, low}` — a noisy fallback regex (`_by_question_runs`) is explicitly marked `low` so
a human annotator sees the shakiest splits first rather than trusting them silently.
`MIN_WORDS=8, MAX_WORDS=220` filters out fragments and whole-article dumps.

**PII scrubbing happens before the raw text ever leaves the git-ignored `data/interim/`
directory** — four regexes chained in a fixed order (email → NID → phone → signature):

```python
# questions.py:26-49
PHONE = re.compile(r"[০-৯0-9]{11}|[০-৯0-9]{4}[- ][০-৯0-9]{6,7}")
NID = re.compile(r"\b[০-৯0-9]{13}\b|\b[০-৯0-9]{17}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SIGNATURE = re.compile(r"\s*(?:নাম প্রকাশে অনিচ্ছুক|[^।?*\n]{2,40})\s*,\s*[^।?*\n]{2,30}?(?:\s*থেকে)?\s*[।.]?\s*$")
```

**`text_verbatim` vs `text_bn`**: the mined text stays local and git-ignored forever (newspaper
copyright); the *released, public* dataset carries only a paraphrase that preserves register
while changing wording, produced through a human annotation pass. The lawyer's published *reply*
is kept separately as evidence of register and likely relevant Act for annotators — "advocates
name a remedy or an Act, rarely a section" — but is never entered as a gold label itself, and
never released.

### 1.5 Synthetic question generation — the part designed around a specific validity threat (`src/dhara/synth.py`)

**The core methodological choice, stated directly in the module's own docstring**: rather than
prompting an LLM "write a citizen question for this provision" (which leaks the provision's own
vocabulary into the question no matter how the prompt is worded — the model has the answer in
context while writing the question), the pipeline inverts the order. A hand-authored topic
inventory (`configs/synth_topics.yaml`) is written with **zero visibility into any provision
text**, and questions generated from those topics are only afterward *matched* to provisions by a
topic-signature score:

```python
# synth.py:386-411, match_score_pre
def match_score_pre(hay, signature, require):
    if require and not any(r in hay.body for r in require):
        return 0.0
    score = 0.0
    for term in signature:
        if term in hay.title: score += 2.0     # title hits count double
        elif term in hay.body: score += 1.0
    return score / max(len(signature), 1)
```

`MIN_MATCH = 0.12` gates which matches are kept. A further **register-leak gate**
(`Topic.violates_register`) discards any rendered question that contains a statute term drawn
from its own matched provisions (checked post-`aggressive()`-normalization) — the count of
discards is reported, not hidden.

**The metric that keeps the whole synthetic pipeline honest — `content_overlap`**:

```python
# synth.py:97-148 (structure)
def content_overlap(q, p, stopwords=STOPWORDS):
    qs = set(aggressive(q).split()) - stopwords - PUNCT - NO_GAP_LOANWORDS
    ps = set(aggressive(p).split()) - stopwords - PUNCT - NO_GAP_LOANWORDS
    return len(qs & ps) / max(len(qs), 1)
```

`MAX_CONTENT_OVERLAP = 0.5` — any generated question with more than half its content words
shared with the provision it's paired against is dropped, at every quality tier, with the discard
count reported per tier. `NO_GAP_LOANWORDS` is a narrow, deliberately *not*-extended allowlist of
English loanwords (সাইবার, রেজিস্ট্রেশন, টোকেন) that citizens and statutes genuinely say
identically — the list is not extended to common process nouns like তথ্য/সরকার/আদালত/মামলা
because **real human-mined questions fail the same overlap gate on exactly those words 7–11% of
the time** — i.e. the gate's strictness is calibrated against ground truth, not tuned to make the
synthetic pipeline's own output look cleaner than reality.

**Narrative scaffolding**: bare topic templates measured ~7× shorter than real mined questions
(median 11 vs 74 words, std 3.4 vs 56.3) — a severe train/eval distribution gap if used as-is. A
`Narrative` composer wraps each bare "core" template in `opener + background + core + closing`
sentences, with gendered cue-word matching (`FEMALE_CUES`/`MALE_CUES`) enforcing that the
scaffolding never contradicts the persona the core implies (a "my husband..." core would never
get a "my wife is at her father's house" background stitched onto it).

**Per-question positive matching is intent-based, not a topic-anchor cross-product** — a measured
defect this fixed: the earlier cross-product approach gave a median of 6 "positive" provisions
per question, of which typically only 1–2 actually answered it; under a contrastive loss that
treats every listed positive as correct, this taught the model that wrong provisions were right
about two-thirds of the time it saw them. `INTENT_CUES` (a hand-written dict of 10 intents —
penalty, procedure, time_limit, compensation, registration, entitlement, prohibition, validity,
guardianship, protection, notice), each carrying both question-side and provision-title-side cue
words, matches a question's detected intent against candidate provision titles before accepting a
pair as a positive.

### 1.6 Two more, distinct question-authoring pathways, kept honestly labelled

- **Authored questions** (`scripts/65_build_authored_questions.py`) are hand-written
  `(chunk_id, question)` pairs, not LLM- or template-generated at all, scored against the *same*
  quality gates real mined questions pass (body-overlap ≤0.20, zero title overlap, word length
  inside the 5th–95th percentile of real human question lengths). A narrow, explicitly-scoped
  waiver (`NO_REGISTER_GAP_DOMAINS`: civil_registration, cybercrime, road_transport,
  constitutional, local_government, consumer, money_recovery) relaxes only the two overlap gates,
  and only in domains where the citizen/legal register split was *empirically measured* to be
  nearly absent — not assumed. Every row is stamped `annotation_mode: "authored"`,
  `review_status: "needs_human_check"`.
- **"Approved title-pairs"** (`scripts/49_promote_approved_retrieval_pairs.py`) is a
  silver→gold promotion step: machine-generated candidates sit in a review queue
  (hard-asserted exactly 3,000 unique-qid rows), and an explicit `--approve-all` flag — chosen
  to be explicit rather than a silent default — "records the project team's confirmation," never
  silently upgrading unreviewed rows to `human_approved_title_pair` status. The output files are
  treated as immutable once written; the script refuses to run a second time over them.

---

## 2. Data preprocessing

### 2.1 Two normalization levels, never mixed (`src/dhara/normalize.py`)

```python
def light(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(ZWNJ, "")                # ZWNJ stripped
    text = text.replace(DANDA_VARIANTS, DANDA)    # unify the two danda characters
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()

def aggressive(text: str) -> str:
    text = light(text)
    text = text.replace(ZWJ, "")                  # ZWJ ALSO stripped here (not in light())
    text = "".join(DIGIT_MAP.get(c, c) for c in text)   # Bangla digits -> ASCII
    text = text.replace(DANDA, f" {DANDA} ")       # danda isolated as its own token
    text = _NON_TEXT.sub(" ", text)                # strip everything outside Bangla/ASCII/danda
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
```

`light()` feeds every transformer input (BGE-m3, the classifier/tagger/LM's own tokenization is
built on top of it too) and deliberately preserves ZWJ — load-bearing inside conjuncts like র‍্য,
a real character a transformer's subword tokenizer was trained to see. `aggressive()` feeds
BM25/Word2Vec/TF-IDF, where surface-form matching is the entire mechanism, so ZWJ variance,
digit-script variance, and case are all collapsed. **The danda is isolated as its own token in
`aggressive()`** specifically because Word2Vec's skip-gram training wants it as a sentence-boundary
context signal — but it's then explicitly excluded again downstream, in `content_tokens()`, from
counting as a lexical match for BM25/TF-IDF, because the danda occurring in the majority of
corpus chunks was found to give every long provision a free "term match" that had nothing to do
with topical relevance and was skewing BM25 ranking toward document length.

```python
# normalize.py:103-113
def content_tokens(text: str) -> list[str]:
    return [t for t in aggressive(text).split()
            if t not in STOPWORDS and t not in _PUNCT_TOKENS]
```

`STOPWORDS` (~100 hand-curated Bangla function words) deliberately **excludes** negation
(না/নেই/নয়) and domain words (ধারা/আইন/আদালত): "shall not" vs "shall" is a legally opposite
rule a bag-of-words representation cannot otherwise distinguish, and citizens genuinely type
ধারা/আইন when searching — stripping them would remove real signal to chase a marginal
noise-reduction.

### 2.2 Leakage-safe splitting — connected components over a provision-sharing graph (`scripts/68_build_human_aware_retrieval_splits_v4.py`)

**Why a naive random split is wrong here**: if question A (train) and question B (dev) both cite
the same provision, a model can get B "right" by memorizing that provision from A's training
signal rather than by generalizing — the eval number would overstate real performance. The fix
needs to guarantee that every provision touched by *any* row is confined to exactly one split,
even when rows are linked only *transitively* (A shares a provision with B, B shares a different
provision with C ⇒ A, B, and C must all land in the same split).

This is exactly a **connected-components** problem over a graph where provisions are nodes and a
question naming ≥2 provisions draws an edge between them, solved with a **disjoint-set
(Union-Find)** data structure:

```python
# 68_build_human_aware_retrieval_splits_v4.py:127-131 (structure)
uf = UnionFind()
for row in gold:
    provisions = [chunk_to_row[cid]["provision_id"] for cid in row["relevant_chunk_ids"]]
    for p in provisions[1:]:
        uf.union(provisions[0], p)
```

`UnionFind.find()` uses iterative parent-pointer path compression (every node visited during a
`find` gets re-pointed directly at the root), and `union()` attaches one root under the other —
together these give amortized near-constant-time operations even as the structure grows, which
matters because this runs over every gold row's provision set. Once every provision has been
unioned, each gold *row* is assigned to the connected component (root) of its first provision —
guaranteeing the leakage invariant by construction, not by a post-hoc check (though the script
still asserts it: `train_provisions & eval_provisions == ∅`, and separately, no question-string
overlap across any pair of splits).

**Component-size policy, and the real instability it fixes**: a single 386-row mega-component
existed (a handful of very popular provisions that huge numbers of real citizen questions happen
to cite). Letting a component that large land in a 35–36-row dev/test split would make the
"effective independent sample size" of the eval set much smaller than its row count suggests —
one lucky/unlucky draw of a mega-component could swing the whole split's score. The fix:
components larger than `MAX_EVAL_COMPONENT_SIZE = 4` are routed to train unconditionally (more
repetition of a popular provision only helps train), and only the remaining small, largely
independent components compete for dev/test, filled by round-robining largest-first across
domains so no single thin domain's small-component supply caps the whole split.

### 2.3 Hard-negative mining — rank window and *global* positive exclusion (`scripts/71_mine_negatives_v6.py`)

```python
RANK_LO, RANK_HI = 5, 30
N_NEGATIVES = 8
DEPTH = 60
```

For each training question: encode with the **zero-shot** (never fine-tuned — mining against a
model you're about to train on would be circular) BGE-m3 checkpoint, score against every corpus
embedding, mask out repealed chunks, take the top 60 by rank. Positions 1–4 are always skipped —
CLAUDE.md's own documented trap: the very top candidates are often relevant-but-unlabelled, and
training a contrastive loss against them teaches the model that a correct answer is wrong.
Positions 5–30 are then scanned, and a candidate is accepted as a negative unless its
`provision_id` is in a positive set accumulated **globally across the entire training file** —
not just that row's own labelled positive. Without this, a provision that happens to be the
*correct* answer for a *different* question elsewhere in the same file could be sampled as a
*negative* for this question, actively teaching the model a contradiction. This was the
mechanism behind an early fine-tune's outright collapse (R@10 0.32→0.06) before being diagnosed
and fixed.

---

## 3. Retrieval experiments: BM25 → Word2Vec → BGE-m3 zero-shot → fine-tuned

### 3.1 BM25 — the lexical floor, not a fused input (`src/dhara/retrievers/bm25.py`)

`rank_bm25.BM25Okapi` over `content_tokens()`-tokenized text, `k1=1.5, b=0.75` (treated as
hyperparameters to tune on dev per the module's own docstring, not fixed constants). A real
ablation is built into the indexable-field logic itself:

```python
# bm25.py:33-36
def _document(self, chunk):
    if self.use_title and chunk.provision_title_bn:
        return f"{chunk.provision_title_bn} {chunk.text_bn}"
    return chunk.text_bn
```

Titles carry the densest formal-legal vocabulary a lexical matcher can exploit, so title+text vs
text-only is reported as its own results-table row, not assumed.

**BM25 is a TF-IDF-family ranking function measuring term-frequency-weighted lexical overlap —
theoretically, it cannot bridge a Bangla question to English statutory text at all**, since the
two share almost no surface tokens by construction. This is why BM25's measured 0.067 Recall@10
(0% on English-only-Act gold) is treated as the expected, informative floor rather than a bug:
it's the clean empirical demonstration of exactly the cross-lingual gap the whole project is
built to measure.

### 3.2 Word2Vec — hand-written skip-gram with negative sampling, not gensim (`src/dhara/word2vec.py`)

Written from scratch in PyTorch — partly a practical necessity (no gensim wheel for the Python
version in use), partly deliberate, because the project's own report needs to explain the learned
geometry at the level of "why does this specific vector arithmetic work," which is easier to do
honestly for code you wrote yourself.

**The model and its loss, in full**:

```python
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, dim=200):
        self.centre = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.centre.weight, -0.5/dim, 0.5/dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, centre, positive, negative):
        # centre: (B,)   positive: (B,)   negative: (B, k)
        c = self.centre(centre)                                # (B, dim)
        p = self.context(positive)                              # (B, dim)
        n = self.context(negative)                               # (B, k, dim)
        pos_score = (c * p).sum(dim=1)                            # (B,)
        neg_score = torch.bmm(n, c.unsqueeze(2)).squeeze(2)        # (B, k)
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()
```

**The theory**: full skip-gram maximizes `p(context | centre)` via softmax over the *entire*
vocabulary, which is computationally intractable at real vocabulary sizes (a normalizing sum
over every word for every training pair). Negative sampling replaces that softmax with a set of
binary logistic-regression sub-problems: push the dot product of `(centre, true context)` up via
`log σ(c·p)`, and push the dot product of `(centre, k sampled non-context words)` down via
`log σ(-c·n)` — an efficient Monte-Carlo approximation of the true softmax gradient, from
Mikolov et al.'s original word2vec paper. Only the `centre` embedding table is kept after
training (the `context` table is a training scaffold, discarded — matching the original paper's
own practice).

**Hyperparameters** (`word2vec.py`'s `train()` defaults): `dim=200, window=5, epochs=5,
batch_size=1024, negatives=10, lr=2.5e-3`. The context window is **dynamic** — a fresh
`randint(1, window)` is drawn per centre word, per epoch, weighting nearby context words more
without an explicit distance-weight term (also from the original paper). **Subsampling** of
frequent words uses keep-probability `sqrt(t / f)` with `t = 1e-3`, re-drawn every epoch — so the
training set literally looks different each epoch, discouraging the model from over-learning
function-word co-occurrences at the expense of content words. **Negative-sampling distribution**
is unigram frequency raised to the 0.75 power (also the original paper's choice — flattens the
distribution so rare words get sampled as negatives more often than raw frequency would, without
making it uniform).

**Query/document vectorization is mean pooling of the centre vectors**, using `aggressive()`
tokenization specifically (**not** `content_tokens()` — stopwords and the danda are meaningful
skip-gram co-occurrence signal that was part of what the vectors were trained on; stripping them
only at retrieval time would silently mismatch train-time and query-time tokenization):

```python
# retrievers/word2vec.py:55-63
vec = self.centre[ids].mean(axis=0)
return vec / norm if norm > 1e-9 else vec       # normalized, then plain cosine at search time
```

**Why R@10=0.029 is expected, not broken**: mean-pooling discards word order and relative
importance entirely, and skip-gram vectors trained on a ~39k-chunk, low-hundred-thousand-token
corpus (CLAUDE.md's own documented trap threshold is 200k–400k tokens for stable neighbourhoods)
simply haven't seen enough co-occurrence evidence to place rare legal terms accurately in the
embedding space. This is the direct, measured argument for why the retrieval task needs a large
*pretrained* transformer rather than a from-scratch distributional model — the from-scratch
BiLSTM retriever ablation (§ removed from current scope, see `DECISIONS.md`) made the identical
point even more starkly for a supervised encoder trained from scratch.

### 3.3 BGE-m3 zero-shot dense bi-encoder (`src/dhara/retrievers/biencoder.py`)

A **bi-encoder** embeds the query and every document **independently**, then compares by a single
dot product — the query never sees the document during encoding, and vice versa. This is what
makes it fast enough to search 39,484 vectors online (a query costs one forward pass + one
matrix multiply, "milliseconds," per the module's own docstring — explicitly why no ANN index is
built at this scale; the complexity would buy nothing).

**Checkpoint choice, with a falsifiable reason recorded, not asserted**: `multilingual-e5-base`
(278M params, 768-dim) was tried first and reached only 8% Recall@10 on the English-gold slice —
close enough to BM25's 0% that "any multilingual encoder crosses the language wall" looked false.
`BAAI/bge-m3` (568M params, 1024-dim) reached 22% on the same slice (32% overall on the full
corpus), which is what moved the checkpoint default. This decision is *evidence-driven*: the
alternative was tested, measured, and found insufficient before being replaced, not assumed
inferior from the start.

**The E5-prefix footgun, handled by inferring the prefix from the checkpoint name rather than a
hand-set flag** (so the zero-shot and fine-tuned runs, which must use the *same* checkpoint
family, can never accidentally drift onto different prefix conventions):

```python
# biencoder.py:46-53
def prefixes_for(checkpoint: str) -> tuple[str, str]:
    name = checkpoint.lower()
    if "e5" in name: return "query: ", "passage: "
    if "bge" in name and "m3" not in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""      # BGE-m3 takes no prefix at all
```

Document encoding uses **light** normalization only (`_document()`), because aggressive
normalization (lowercasing, digit conversion, punctuation stripping) would move the input off the
distribution the transformer's subword tokenizer was actually trained on — a documented,
measurable performance loss if done wrong (CLAUDE.md's own known-traps table).

### 3.4 LoRA fine-tuning — the core supervised deliverable (`notebooks/ssh-retrieval-finetune.ipynb`)

**The document template used for training, indexing, and evaluation — kept identical everywhere
on purpose, since a template mismatch between training and serving would silently degrade
retrieval no matter how good the fine-tune was**:

```python
def document_text(row):
    title = row.get('act_title_bn') or row.get('act_title_en') or ''
    section = row.get('provision_no_ascii') or row.get('provision_no_bn') or ''
    return f"Act: {title} | Section: {section} | {row.get('text_raw') or row.get('text_bn') or ''}"
```

Queries are the raw question string, unprefixed (BGE-m3's own convention).

**LoRA — the theory and the exact config used**:

```python
lora = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias='none', target_modules=['query', 'value'],
)
```

Full fine-tuning would update the entire weight matrix `W` of every targeted linear layer. LoRA
instead **freezes** `W` and learns a low-rank update `ΔW = B·A`, where `A ∈ ℝ^{r×d_in}` and
`B ∈ ℝ^{d_out×r}` are the only trainable parameters, with `r=16` far smaller than the backbone's
hidden dimension (1024) — the forward pass becomes `h = Wx + (α/r)·B·A·x`, where `α=32` is a
fixed scaling constant so the update's effective magnitude doesn't shrink as `r` changes. Only
the `query` and `value` projection matrices inside self-attention are targeted (not `key` or the
feed-forward layers) — the standard finding from the original LoRA paper that attention's
query/value projections carry most of the adaptable signal for downstream tasks, and this keeps
trainable-parameter count tiny (0.28% of the full model here) — the entire reason fine-tuning a
568M-parameter model is tractable on a single T4 GPU's memory at all. `bias='none'` and
`lora_dropout=0.05` are standard regularization choices; `gradient_checkpointing_enable()` and
fp16 training are what makes the memory budget work on that hardware specifically.

**The loss — `MultipleNegativesRankingLoss` (in-batch + explicit hard negatives)**: for a batch
of `(anchor, positive, negative_1, ..., negative_n)` tuples, the loss computes the similarity of
every anchor to **every** positive and explicit negative present anywhere in the batch (not just
its own), then applies cross-entropy treating the anchor's own positive as the single correct
class among all candidates in the batch. For batch size `B` with up to `N` explicit hard
negatives per row, the softmax denominator per anchor sums over up to `B + B·N` candidates — the
`B−1` other rows' positives serve as free "in-batch negatives" at zero extra compute, and the
mined hard negatives add harder, deliberately-chosen distractors on top. This is why
`BatchSamplers.NO_DUPLICATES` matters: a duplicate positive appearing twice in one batch would
silently count as its own in-batch negative, which is a false signal (it's not actually wrong,
it's the answer, just to a different anchor in the batch).

**Per-stage curriculum** (`run_stage()`, shared function, all stages `warmup_ratio=0.1, fp16=True,
max_grad_norm=1.0`, seed=42):

| Stage | Data | Epochs | LR | Batch × grad-accum |
|---|---|---|---|---|
| `bge_m3_approved_title_v4` | 2,685 human-approved title-pairs | 2 | 5e-6 | 8 × 4 |
| `bge_m3_diverse_augmentation_v1` (if present) | LLM augmentation | 1 | 3e-6 | 8 × 4 |
| `bge_m3_authored_and_human_v1` | 942 authored + 408×2 (oversampled) human | 3 | 4e-6 | 8 × 2 |

**Human-row oversampling** (`HUMAN_OVERSAMPLE_FACTOR = 2`) exists because raw real-citizen rows
(408, after the split/merge policy fix that recovered them from an earlier bug's 52) would
otherwise be heavily outnumbered by the 942 authored rows in the final stage — 2× repetition
(shuffled with a fixed seed, not naively appended in a block) keeps genuine citizen phrasing
proportionally represented in the gradient without letting oversampling dominate it entirely.

**A hard promotion gate, not a soft warning**:
```python
assert finetuned_dev['R@10'] >= baseline_dev, 'Do not promote this checkpoint. Retrain.'
```
A checkpoint that regresses dev performance relative to the locked zero-shot baseline cannot even
be saved — this is enforced in code, not left to a human remembering to check.

### 3.5 Cross-encoder reranking — a fundamentally different architecture, not a bigger bi-encoder

A **bi-encoder** computes `u(Q)` and `v(D)` independently and compares `u(Q)ᵀv(D)` — the two
never interact until the final dot product. A **cross-encoder** concatenates `[CLS] Query [SEP]
Document [SEP]` into a single sequence and runs it through the transformer's full self-attention
**jointly** — every query token can attend to every document token and vice versa throughout all
layers, producing a much richer relevance signal at the cost of needing one full forward pass
*per candidate document*, which is why it's only ever run over a shortlist (top-50 here), never
the whole corpus — the complexity that makes a bi-encoder searchable at scale is exactly what a
cross-encoder gives up for accuracy.

`BAAI/bge-reranker-v2-m3`, trained on 1 positive + up to 4 mined hard negatives per question
(`InputExample(texts=[q, doc], label=1.0/0.0)`), 1 epoch, `lr=2e-5`, batch size 4. A real,
documented regression happened here: the reranker was trained against each question's specific
labelled chunk (`positive_chunk_ids[0]`) but evaluated against an arbitrary *different* chunk
sharing the same provision (a train/eval mismatch from how stage-1 candidates were deduped) — a
genuine bug, found, root-caused, and fixed in `retrieval_metrics()`'s candidate tracking, though
the regression persisted at the same magnitude after the fix, meaning a second, still-unidentified
cause remains (see `docs/PROJECT_REPORT.md` §5.4 for the current, honest status).

---

## 4. Classification, sequence tagging, language modeling

### 4.1 Classification — generative vs. discriminative on identical TF-IDF features (`src/dhara/classify.py`)

**Naive Bayes forced to a uniform class prior**, rather than trusting sklearn's default (the
*learned* frequency, which with `other` at ~22–66% of the data across versions would do most of
the "just predict other" work for NB by itself):
```python
nb = MultinomialNB(class_prior=[1.0 / n_classes] * n_classes)
```
This isolates NB's actual generative mechanism — Bayes' rule combining a prior with a
likelihood — to just the likelihood term, the fairest apples-to-apples generative analogue of
LogReg's `class_weight='balanced'` (which reweights the loss, not a prior).

**LogReg's regularization strength `C` chosen by cross-validation, not assumed**:
```python
kfold = KFold(n_splits=3, shuffle=True, random_state=42)   # plain KFold, not stratified —
                                                              # several domains have single-digit
                                                              # counts, which breaks stratification
for c in (0.3, 1.0, 3.0):
    scores = cross_val_score(LogisticRegression(C=c, class_weight='balanced', max_iter=1000),
                              X, train_labels, cv=kfold, scoring='f1_macro')
```
Selected on the training fold only, never touching the 200-row held-out test set. On the
best-performing (human-only, 352-row) configuration, CV independently reproduces `C=1.0`, the
same value the earlier unexamined default happened to use — real evidence the default wasn't
lucky, now that it's been checked rather than assumed. A second lever, adding word bigrams to the
TF-IDF features (`ngram_range=(1,2)`), was tested and **reverted** after it regressed the same
configuration's macro-F1 from 0.343 to 0.203: 1.4× the feature count (3,587→4,998 features even
at `min_df=2`) overfits a training set this small rather than capturing the extra Bangla
case/postposition signal it was meant to.

**`VanillaRNNClassifier` — a single-layer, unidirectional `nn.RNN`, shape-traced**:
```python
self.rnn = nn.RNN(dim, hidden=128, batch_first=True)
def forward(self, ids, lengths):                        # ids: (B, T)
    emb = self.embedding(ids)                             # (B, T, dim)
    packed = pack_padded_sequence(emb, lengths, ...)
    _, h_n = self.rnn(packed)                              # h_n: (1, B, 128) — 1 layer × 1 direction
    return self.head(h_n.squeeze(0))                        # (B, 128) -> (B, n_classes)
```
Only the final hidden state is used — the whole question is compressed into one 128-d vector
before classification, which is the "vanilla RNN, `h_n.squeeze(0)`" spec the syllabus asks for.

**`StackedBiLSTMClassifier` — 2-layer bidirectional LSTM, shape-traced**:
```python
self.lstm = nn.LSTM(dim, hidden=128, num_layers=2, bidirectional=True, batch_first=True)
def forward(self, ids, lengths):
    _, (h_n, _) = self.lstm(packed)                         # h_n: (4, B, 128) — 2 layers × 2 directions
    h_fwd, h_bwd = h_n[-2], h_n[-1]                            # LAST layer's fwd/bwd rows specifically
    return self.head(torch.cat((h_fwd, h_bwd), dim=1))          # (B, 256) -> (B, n_classes)
```
`h_n`'s first axis stacks `[layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd]` in that order for a
2-layer bidirectional LSTM — `h_n[-2]`/`h_n[-1]` picks out the *last* layer's forward and
backward final states specifically (not layer 0's), concatenated to a 256-d vector that carries
context from both directions of the final, most-abstract layer.

**Class-weighted cross-entropy**, hand-reimplementing sklearn's `class_weight='balanced'` formula
for a PyTorch loss (`n_samples / (n_classes × count_c)`), applied because the RNN classifiers had
*no* imbalance correction at all in an earlier version (unlike NB's uniform prior and LogReg's
built-in balancing) — a real, disclosed finding is that this correction, tried in isolation,
initially made the RNNs *much worse* (rare domains with 1–2 examples get ~25× loss weight, which
8 epochs of Adam at `lr=1e-3` on 352 rows is not numerically stable against) until combined with
the rare-domain-merge intervention that made it net positive.

### 4.2 Sequence tagging — `BiRNNSequenceLabeler`, weak labels, per-token output (`src/dhara/models/tagger.py`)

```python
self.rnn = nn.RNN(dim, hidden=128, bidirectional=True, batch_first=True)
self.head = nn.Linear(hidden * 2, n_tags=5)
def forward(self, ids):                # ids: (B, T)
    out, _ = self.rnn(emb)               # out: (B, T, 256) — EVERY timestep's fwd/bwd output
    return self.head(out)                 # -> (B, T, 5) — per-token logits
```

The critical architectural difference from the classifier above: the classifier reduces a
sequence to one final hidden state and classifies the whole question; the tagger instead keeps
**every timestep's** hidden output and classifies **each token independently** — the output
tensor has an extra time axis (`B, T, n_tags` vs `B, n_classes`), because the task is "label each
word," not "label the sentence." `PAD_TAG_ID = -1` is passed to `CrossEntropyLoss(ignore_index=
-1)` so padded positions in a batch contribute zero gradient — a separate concern from the input
embedding's own `padding_idx=0`, which is about the vocabulary, not the tag space.

Labels are bootstrapped by regex (section-number proximity to a marker word) plus gazetteer
matching (`data/lexicon/register_map.json`) against raw questions, never hand-corrected — stated
as an explicit limitation in the module's own docstring and in every run's output JSON, not
hidden. The real finding from replacing the misleading raw-accuracy metric with per-entity F1:
`ACT` (0.857) and `PARTY` (0.627) are learnable because citizens genuinely name institutions and
family/legal roles in plain language; `SECTION_NO` (0.000, only 2 weak-label occurrences across
352 training questions) and `LEGAL_TERM` (0.000, formal-register gazetteer barely fires on
colloquial text) are not learnable from this data — the same register gap Topic 3 measures
independently, surfacing a second time in an unrelated architecture.

### 4.3 Language modeling — perplexity as a register-gap measuring instrument (`src/dhara/models/language_model.py`)

```python
self.lstm = nn.LSTM(dim, hidden=256, num_layers=2, batch_first=True)   # UNIDIRECTIONAL
def forward(self, ids):    # ids: (B, T) -> logits: (B, T, vocab_size)
```
**Deliberately unidirectional** (causal), unlike the bidirectional tagger/classifier RNNs — a
language model predicts the *next* token from only what came before it; a bidirectional model
would let the network peek at the answer through its backward pass, which isn't language
modeling, it's cheating.

**Windowing**: each document is tokenized, wrapped in `<bos>`/`<eos>`, cut into non-overlapping
65-token windows (`seq_len=64` + 1 for the next-token target); input is `window[:-1]`, target is
`window[1:]`.

**Perplexity, exactly**:
```python
loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=0)
ppl = float(torch.exp(loss))
```
i.e. `PPL = exp( −(1/N) Σ log p(token_t | token_<t)) )` — the standard information-theoretic
definition, the exponential of the average per-token negative log-likelihood, no smoothing or
backoff. Perplexity has a direct interpretation as "the model's effective branching factor" — a
PPL of 100 means the model is, on average, as uncertain as if it were choosing uniformly among
100 equally likely next tokens; a *lower* PPL on formal legal text than colloquial citizen text
would mean formal Bangla is more predictable (less surprising) to a model that has seen the
corpus — exactly the quantitative form of "the lexical/register gap" the whole project's thesis
rests on, measured here **without needing a retriever in the loop at all**, as an independent
line of evidence. The actual measured result is nuanced, not a clean story: median colloquial PPL
(89.5) is slightly *lower* than median formal PPL (100.3), but the colloquial *mean* (138.9,
pulled up by a heavy tail of very-hard questions, max PPL 1575 vs formal's 358) is higher — the
register gap this project cares about is driven by outlier phrasing, not a uniform shift across
all colloquial text, and that nuance is only visible because both mean and median were reported,
not just one.

### 4.4 `build_pretrained_embedding_matrix()` — Topic 0's shared artifact (`src/dhara/vocab.py:72-104`)

```python
def build_pretrained_embedding_matrix(vocab, vectors, dim, seed=42):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.normal(mean=0.0, std=0.1, size=(len(vocab), dim), generator=generator)
    matrix[vocab.pad_id] = 0.0
    hits = 0
    for token, index in vocab.stoi.items():
        if token in SPECIALS: continue
        if token in vectors:
            matrix[index] = torch.tensor(vectors[token], dtype=torch.float32)
            hits += 1
    matrix.coverage = hits / max(len(vocab) - len(SPECIALS), 1)
    return matrix
```
`vectors` is the project's own trained Word2Vec `centre` table (§3.2) — reused across the
classifier, tagger, and language model specifically so **one token id means the same underlying
distributional meaning everywhere**, rather than each model learning its own private embedding
space from a cold start on limited task-specific data. **OOV rows are not left at zero** — a zero
vector would make every unseen word literally identical to the padding token and to every other
unseen word, collapsing part of the model's input space into one point; instead they're drawn
from `Normal(0, 0.1)` under a seeded generator (reproducible), which gives every OOV token a
distinct, small-magnitude starting point the model can still learn to differentiate during
fine-tuning. `coverage` (hits / non-special vocab size) is attached to the returned tensor and
reported, so a downstream model with poor embedding coverage is visible rather than silently
underperforming for an unexplained reason.

---

## 5. Evaluation infrastructure — why every number in this project is allowed to be trusted (`src/dhara/metrics.py`, `scripts/18_compare_runs.py`)

**Everything scores on `provision_id`, never `chunk_id`** — retrieving *any* chunk of the correct
provision counts as correct, so a long provision split into several sub-chunks isn't unfairly
penalized relative to a short one-chunk provision. The shared primitive:

```python
def first_hit(ranked_provision_ids, gold):
    seen, rank = set(), 0
    for provision_id in ranked_provision_ids:
        if provision_id in seen: continue          # dedupe on the fly
        seen.add(provision_id); rank += 1
        if provision_id in gold: return rank
    return None
```

Recall@k, MRR@10 are built directly on the list of per-query ranks this returns — deliberately
never pre-aggregated at the source, because both error analysis and the significance test below
need the raw per-query numbers, not a summary that's already thrown them away. (nDCG@10 is
**not** in `metrics.py` — it's computed ad hoc inline inside the fine-tuning notebook's own eval
cells with the standard `1/log2(rank+1)` formula; worth knowing this duplication exists rather
than assuming one canonical implementation.)

**The paired bootstrap significance test — the mechanism that keeps every claimed improvement
honest**:

```python
def paired_bootstrap(ranks_a, ranks_b, k=5, n_resamples=10_000, seed=0, alpha=0.05):
    hit_a = [1.0 if r is not None and r <= k else 0.0 for r in ranks_a]
    hit_b = [1.0 if r is not None and r <= k else 0.0 for r in ranks_b]
    observed = (sum(hit_a) - sum(hit_b)) / n
    deltas = []
    for _ in range(n_resamples):
        total = sum(hit_a[i] - hit_b[i] for i in (rng.randrange(n) for _ in range(n)))
        deltas.append(total / n)
    deltas.sort()
    lo, hi = deltas[int(0.025*n_resamples)], deltas[int(0.975*n_resamples)]
    return {"delta": observed, "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
```

**Why *paired*, specifically**: the same random index `i` is drawn into both `hit_a[i]` and
`hit_b[i]` on every resample — because it's the *same* underlying query difficulty contributing
to both systems' scores, that shared variance **cancels** in the resampled *difference*, giving
far more statistical power at small sample sizes than treating the two runs as independent
samples would. **Why bootstrap resampling at all, rather than a normal-approximation
confidence interval**: with a binary hit/miss statistic and n as small as 36, a normal
approximation is a poor fit; resampling the actual paired differences 10,000 times and reading
off the 2.5th/97.5th percentiles directly makes no distributional assumption about the underlying
data. **Significance is CI-excludes-zero**, not a p-value threshold — equivalent under a
percentile bootstrap, but stated in the form that's directly interpretable ("the true difference
is between X and Y with 95% confidence, and that range doesn't include 'no difference'").

`scripts/18_compare_runs.py` refuses to compare two runs whose qid sets don't match exactly
(`SystemExit` — "a paired test on unaligned questions is meaningless"), loops over language
slices (`all/english/bengali/mixed`) and every recall cutoff, and prints an explicit
self-check when nothing survives significance: *"No difference survives a paired bootstrap.
Report that plainly — a null result honestly stated is worth more here than a narrated bar
chart."* That sentence, present in the script's own source, is the project's actual evaluation
philosophy made executable rather than just stated in a document.

---

## 6. Why this is not "just another RAG demo"

A minimal RAG tutorial is: chunk some documents arbitrarily, embed them with an off-the-shelf
model, embed the query, cosine-similarity, return top-k, done — usually an afternoon's work with
no measurement of whether any of it actually helps. Line up what this project did against that
baseline, stage by stage:

1. **Corpus construction is not "chunk at N tokens."** Provisions are parsed from two different
   source formats with format-specific regex families that handle real edge cases (footnote
   markers, multi-part continuations, Bangla/ASCII digit variance), chunked at a
   linguistically-motivated 400-word/250-word-window/50-word-overlap policy rather than an
   arbitrary token count, deduplicated with different handling for identical-vs-different-text
   collisions, and checked for legal validity (repealed-Act and repealed-provision detection) at
   two independent levels because one level alone was measured to miss real cases — a domain
   where retrieving stale law is not a UX inconvenience but active harm.

2. **Three separate, methodologically distinct question-collection pipelines**, not one. Real
   mined questions go through PII scrubbing, source-specific boundary detection with confidence
   labelling, and copyright-safe paraphrasing before release. Synthetic questions are generated
   by a pipeline specifically *inverted* to avoid a subtle validity threat (LLM writing a
   question from its answer leaks vocabulary regardless of prompt wording) — topic templates are
   written blind to provision text, then matched afterward by a scored, leak-gated,
   overlap-audited process with a discard rate reported at every stage. Hand-authored questions
   pass the *same* quality gates as mined ones, with a narrow, empirically-justified (not
   assumed) domain-scoped waiver. None of this exists in a standard RAG pipeline, which typically
   has no synthetic-question generation step at all, let alone one engineered against a specific
   known failure mode.

3. **A four-representation retrieval ladder, each rung actually measured, not skipped.** BM25
   (lexical floor, its 0% English-gold performance used as *evidence* for the thesis, not hidden
   as an embarrassment), a from-scratch skip-gram Word2Vec model (with the actual negative-sampling
   math implemented and explained, not imported as a black box), a locked zero-shot pretrained
   dense baseline (chosen over a plausible alternative checkpoint only after that alternative was
   tested and found insufficient), and a LoRA-fine-tuned version of the same checkpoint with a
   real contrastive loss, globally-exclusion-checked hard-negative mining, and a multi-stage
   training curriculum. A standard RAG demo stops at rung three and calls it done.

4. **A second retrieval stage — cross-encoder reranking** — architecturally distinct from the
   bi-encoder (joint attention vs. independent encoding), with its own training run, its own real
   regression that was found and partially root-caused rather than swept under the rug, and an
   explicit statement of what remains unresolved. Most RAG tutorials never rerank at all.

5. **A tested, honestly-reported hybrid retrieval ablation** — BGE-m3's sparse lexical and
   ColBERT multi-vector heads, plus bn→en dual-query translation, specifically targeting the
   cross-lingual wall — implemented via a completely different API (`FlagEmbedding.BGEM3FlagModel`
   vs `sentence_transformers`), tested, found not to help, and the *specific mechanistic reason
   why* (a translation-truncation bug feeding a question-less English query into a max-fusion
   rule) traced and documented rather than left as an unexplained null result.

6. **Three additional, independently-evaluated NLP subsystems that have nothing to do with
   retrieval directly**: a generative-vs-discriminative classification comparison on identical
   TF-IDF features (with a real, disclosed methodological failure — bigram features overfitting
   a small training set — tested and reverted, not silently avoided); a sequence tagger whose
   headline metric was itself upgraded mid-project from a misleading O-dominated accuracy number
   to honest per-entity F1, revealing a genuine, cross-topic-consistent linguistic finding about
   which entity types survive contact with real colloquial phrasing; and a from-scratch causal
   language model used purely as a measuring instrument (perplexity) to quantify the register gap
   *without* a retriever in the loop at all, as independent corroborating evidence for the
   project's central claim.

7. **Leakage prevention implemented as a graph algorithm, not a random split.** Provision-sharing
   is modeled explicitly as a graph and split by connected components via Union-Find, specifically
   because a random split was measured to let popular-provision clusters destabilize the eval
   set — this is real algorithmic engineering applied to a data-integrity problem, not a
   `train_test_split()` call.

8. **A statistically rigorous evaluation layer.** Every retrieval delta claimed in this project is
   gated behind a paired bootstrap significance test with 10,000 resamples and a stated
   confidence interval — the evaluation code itself refuses to compare misaligned runs and prints
   an explicit "report this as noise" message when nothing clears significance. A typical RAG
   demo reports a single point-estimate recall number and stops.

9. **Ethics and safety infrastructure that has no equivalent in a retrieval demo**: risk-tier-aware
   abstention thresholds calibrated against a deliberately-included unanswerable-question subset,
   a legal-aid-referral framing layer that is provably decoupled from ranking (a documented,
   fixed bug specifically ensured the risk framing can't depend on an arbitrary UI slider), and a
   repealed-law exclusion list checked at serving time as well as at corpus-build time.

10. **A documented trail of real negative results, bugs found, and reverted attempts** — cross-topic
    negative contamination collapsing an early fine-tune, a stale-file bug and a split/merge
    policy mismatch each independently flattening a later attempt, a reranker regression still
    only partially explained, a leaked-answer evaluation bug caught and formally retracted in the
    project's own decision log, a bigram classification feature that looked reasonable and
    measurably wasn't. None of these are things a from-scratch RAG tutorial would ever produce,
    because none of them would be caught without the measurement discipline (the paired
    bootstrap, the leakage assertions, the overlap audits) built into every stage above it.

The throughline across all ten points is the same: every stage was **measured**, every
alternative that was rejected was rejected *because* it was tested and found worse (not assumed
worse), and every negative or null result is recorded with its specific mechanistic cause rather
than smoothed over. That discipline, applied consistently across corpus construction, three
distinct data-generation methodologies, a four-rung retrieval ladder plus reranking plus a
hybrid-retrieval ablation, three independent NLP subsystems, and a statistically rigorous
evaluation layer, is the actual scope difference between this project and an afternoon's RAG
demo — not a longer feature list, a longer chain of verified decisions.
