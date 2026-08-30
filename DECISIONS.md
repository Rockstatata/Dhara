# Decisions

Every non-obvious choice: date, decision, reason, accepted cost. This file is the raw material for the report's methodology chapter and cannot be reconstructed later.

## 2026-08-23 — Language balance pursued by adding Bangla acts, never by translating

Goal was an even Bangla/English corpus. Two routes were rejected before the third
was taken.

**Fetching Bangla versions of the English acts is impossible**, and this was
verified rather than assumed: bdlaws was fetched at `?lang=bn` for acts 11, 75,
138, 90, 241 and 46. All six return 1,770–2,207 Bangla characters — an identical
figure across acts, because it is the site navigation — against 8,670–30,788
English characters of body text. Those laws have never been published in Bangla.

**Machine-translating them was rejected outright.** Serving a citizen a
machine-translated provision as if it were the law is the exact harm the ethics
section exists to prevent, and no disclaimer makes it acceptable. If translation
is ever wanted it is an experimental condition with its own labelling, never
corpus content.

**Route taken: add more Bangla acts that qualify on the everyday-frequency rule.**
BLAD holds 684 unused Bangla acts (21,816 provisions), so the additions were
chosen on merit and the ratio followed. Added: road transport (act-1262 — it was
never missing, BLAD spells it পরিবহণ, not পরিবহন), the ICT Act, food safety, drugs
and cosmetics, copyright, trademark, the Children Act, four metropolitan police
acts as the Bangla-language side of police powers, co-operative societies, and a
new `local_government` domain (municipality, city corporation, union parishad,
upazila parishad) covering trade licences, holding tax and birth certificates.

Result: 5,501 chunks, **58.9% Bangla / 41.1% English**, from 40/60 before.

**But the aggregate ratio is cosmetic and must not be reported alone.** Per
domain it is unchanged where it matters: family 7% Bangla, land 10%, criminal
procedure 29%. A citizen asking a family-law question still faces a 93%-English
target set. Balance was achieved by adding Bangla law *elsewhere*, which helps
overall coverage and does nothing for the domains people ask about most. Results
are therefore reported per domain **and** per language; the corpus-level split is
a descriptive statistic, not a claim.

## 2026-08-23 — Two parser bugs found while rebalancing, both silent data loss

`provisions_dropped_unnumbered` jumping from 6 to 374 exposed two faults that
would otherwise have shipped invisibly:

  - **BLAD splits some sub-sections into separate records.** `(2) It extends to…`
    arrives as its own entry with its parent `(1)` in a different one. These were
    being discarded — 153 provisions of the Code of Criminal Procedure and 43 of
    the Negotiable Instruments Act alone. They are now merged into the preceding
    provision, per the rule that sub-sections stay with their parent.
  - **Letter-suffixed provision numbers were unmatched.** `2A.`, `17B.`, `33A.`
    — the pattern allowed a Bangla letter suffix but not a Latin one, so
    amendment-inserted provisions in the English acts were dropped.

Preambles ("WHEREAS it is expedient…") and asterisk-marked omissions (`3[***]`)
are now counted in their own buckets rather than as parser misses. Drops fell
from 374 to 79.

## 2026-08-23 — Supervisor guidance received; several assumed constraints were not real

Written answers from Shawon Sir, and what each one changes here.

**Embeddings — either is acceptable, and a comparison is best.** "If your corpus
is small, pretrained embeddings often perform better. However, if you train your
own, you can provide a more concrete explanation... A balanced approach could be
to start with pretrained embeddings and, if feasible, supplement with your own
training for comparison." We already train our own (PyTorch skip-gram with
negative sampling), which is the concrete-explanation path and answers the
"explain how you did embedding" requirement directly. **Added:** a pretrained
Bangla vector comparison column, which Sir explicitly recommends and which the
neighbour table was already built to hold. Note his premise about corpus size no
longer applies — the corpus is ~45,000 chunks, not small.

**Classification is not required.** "You are not bound to do classification works.
It's an open ended project." This was being treated as mandatory and is not. The
supervised core of this project is the retrieval fine-tuning itself — contrastive
learning on labelled question→provision pairs — which satisfies "primarily
supervised" without any classifier. Domain classification stays as **syllabus
demonstration and a cheap generative-vs-discriminative comparison**, not as a
requirement.

**Generative vs discriminative — one is enough.** "You may focus on one type and
clearly justify your choice. If time permits, comparing at least one generative
and one discriminative approach can strengthen your report." Naive Bayes on
TF-IDF is about twenty lines, so we keep both. It is a bonus, not a gate.

**No GUI required.** "A fancy GUI is not required. However, there should be a
clear input-output system." The Gradio demo stays deliberately plain. No further
design investment.

**Unsupervised work adds value.** Clustering with ARI/NMI against domain labels
stays in scope as a secondary contribution.

**Data collection must be documented.** "Include sources, methods, and any
preprocessing you performed." Already covered by `DECISIONS.md`,
`docs/research/question-sources.md` and `docs/PIPELINE.md`; this confirms that
effort was correctly placed.

## 2026-08-23 — No transformer written from scratch

Dropped on the user's instruction. Pretrained transformers are used directly:
BanglaBERT for classification, a multilingual sentence encoder for retrieval.

The "explain how you did embedding" requirement is met by the Word2Vec
implementation instead — skip-gram with negative sampling written in PyTorch,
where the objective, the frequency subsampling and the unigram^0.75 negative
distribution are all ours and can be explained line by line. That is a better fit
for the requirement than a scratch transformer would have been, because the
learned geometry is directly inspectable as a nearest-neighbour table.

## 2026-08-23 — Restructured around the course syllabus, not around a retrieval ladder

The plan had drifted toward what makes a good retrieval paper and away from what
the course actually requires. Checked against the syllabus, three of the five
PyTorch topics had **no artifact at all**: sequence tagging, language modelling,
and sequence-to-sequence. BM25 — a TF-IDF-family ranking function, not a language
model — had grown to occupy far more of the plan than one baseline deserves.

Every syllabus topic now maps to something the system genuinely needs, rather
than to an exercise bolted on beside it. The full map is `docs/PIPELINE.md`.

  - **Topic 0** — `build_pretrained_embedding_matrix()`, self-trained Word2Vec
    versus general Bangla vectors, with the nearest-neighbour comparison as a
    result in its own right.
  - **Topic 1** — domain classification: Naive Bayes (generative) and Logistic
    Regression on TF-IDF, then `VanillaRNNClassifier` and
    `StackedBiLSTMClassifier` on Topic 0 embeddings.
  - **Topic 2** — `BiRNNSequenceLabeler` tagging ACT / SECTION_NO / LEGAL_TERM /
    PARTY in citizen questions. Does real work: extracts an explicit "১০৩ ধারা"
    so retrieval can boost that provision directly.
  - **Topic 3** — `StackedLSTMLanguageModel` on formal legal Bangla, used as a
    **measuring instrument**: perplexity on colloquial questions versus formal
    ones quantifies the register gap independently of any retriever. A second,
    orthogonal piece of evidence for the central claim.
  - **Topic 4** — `Seq2SeqTranslation` doing **register** translation, colloquial
    Bangla to formal legal Bangla, trained on the 100 `PAIR` annotations. Used as
    query reformulation before retrieval. Language translation is impossible here
    — no Act exists in both languages, so no parallel legal corpus exists — but
    parallel *register* data is something we are already collecting.
  - **Topic 5** — a transformer encoder written from scratch in PyTorch, trained
    as a dual encoder, **and** a fine-tuned pretrained model. The scratch model
    will lose on this much data; that gap is the finding, because it shows what
    pretraining buys in our own numbers.

Cut to make room: the Word2Vec and BiLSTM *retrieval* rungs. They become Topic 0
and Topic 1 classification instead, which is where the syllabus wants them. Two
table rows lost, three PyTorch artifacts gained.

Checkpoint split, because one model cannot do both jobs: `csebuetnlp/banglabert`
for classification (ELECTRA, strongest Bangla-specific), and a multilingual
sentence encoder for retrieval, which must reach the English-only Acts that
BanglaBERT cannot represent.

## 2026-08-23 — No generation layer. Dhara is retrieval, and that is now permanent

The supervisor has ruled out RAG for this project, on the grounds that feeding a
dataset to a retrieval-augmented generator demonstrates no NLP work. That is a
fair reading of what RAG usually is, and it does not conflict with this design —
Dhara has no generation step. The output is the retrieved provision and its
citation, shown as `text_raw`, the law as printed. Nothing writes an answer.

Consequently the proposal's stretch objective 8 — "a grounded generation layer
producing a plain-Bangla explanation" — is **cut permanently**, not deferred. It
was already stretch-only and cuttable; it is now removed from scope.

Everything on the ladder is trained here: BM25 tuned on dev, Word2Vec trained
from scratch on our corpus, a BiLSTM dual-encoder trained from scratch with
InfoNCE, a bi-encoder fine-tuned on mined hard negatives, a cross-encoder
reranker, and four intent classifiers. The contribution is a *measurement* — the
register gap and the language gap — not a pipeline assembled from an API.

Worth stating in the report: MINA (ACL Findings 2026) and LegalRAG are both RAG
systems, and both evaluate only on Bar Council exam questions. Not generating is
part of what separates this work from them.

Two LLM uses survive and both are disclosed rather than hidden:
  - **Synthetic training questions** (doc2query/InPars). Training split only,
    never at inference, never in evaluation. Less load-bearing than originally
    planned now that 584 real mined questions exist.
  - **Model-assisted annotation.** An agent proposes BM25 candidates; a human
    decides every gold label. See the gold-set entry above.

## 2026-08-23 — No act on bdlaws exists in both languages; corpus is single-language per act

Tested in both directions rather than assumed, because the whole corpus design
turned on it. `?lang=` on the bdlaws portal switches the **interface chrome only**
and never the text of the law:

| Act | `?lang=` | Bangla chars | English chars |
|---|---|---|---|
| Penal Code 1860 (act-11) | `bn` | 1,893 (menus) | 30,788 |
| Labour Act 2006 (act-952) | `en` | 14,456 | 1,405 (menus) |
| Consumer Act 2009 (act-1014) | `en` | 4,206 | 1,423 (menus) |

Checked at provision level too: Penal Code section 1 under `?lang=bn` returns
"Title and extent of operation of the Code" with **zero** Bangla characters.

So there is no parallel bilingual corpus to collect, in either direction. Each act
has exactly one authoritative language, set by when it was enacted: colonial-era
law is English and has never been translated, post-1987 law is Bangla. A
"100% Bangla and 100% English" corpus of the same laws is not something that
exists to be fetched. Reported here because it is the single fact that determines
what the retrieval task actually is.

## 2026-08-23 — Full corpus: every act, not only the everyday-life domains

`scripts/03_build_corpus.py --all` ingests all 1,484 acts — 37,678 chunks over
34,911 provisions, 61% Bangla / 39% English. Acts outside `configs/domains.yaml`
carry `domain: other` and `risk_tier: medium`; unknown risk must not present as
low, and must not fire a legal-aid referral on every result either, which would
make the referral meaningless where it actually matters.

The domain-scoped build stays the default (`--all` is opt-in): evaluation is
still deep on the everyday-life domains, and a 35k-provision index makes
retrieval measurably harder by adding confusable candidates. Both files are kept.

## 2026-08-23 — 203 acts are empty in BLAD and are fetched from bdlaws instead

BLAD carries 203 acts with `language: "unknown"` and zero provisions, and they
are not obscure: সাইবার নিরাপত্তা আইন ২০২৩ (the main cybercrime act),
ডিজিটাল নিরাপত্তা আইন ২০১৮, মানিলণ্ডারিং প্রতিরোধ আইন ২০০৯, পেটেন্ট আইন ২০২২.

`scripts/02_scrape.py --gaps` fetches exactly those from the portal. The pass
closes a second gap at no extra cost: bdlaws puts the provision title in
`.txt-head`, and BLAD has no provision titles at all. A smoke test on two acts
returned titles for 100% of provisions. Titles are dense with legal terminology,
so this should help retrieval more than the raw provision count suggests.

## 2026-08-22 — ICCIT 2026 declined; target the following cycle

ICCIT 2026 closes 31 August, nine days out. Meeting it would mean a ~150-question
gold set and no agreement round, and those two things are the contribution.
Accepting there would also block the same work from NLLP/BLP, since dual
submission is not allowed. Declined on those grounds, not on timing alone.

## 2026-08-22 — Corpus is built on BLAD, and BLAD is thinner than its card claims

BLAD is used instead of scraping bdlaws: re-deriving a published dataset by hand
is exactly what a reviewer would object to. Verified against the file rather than
the dataset card, four advertised things are absent:

  - **No provision titles.** The card lists `section_title`; records carry
    `section_content` only. Titles are dense with legal terminology and help
    retrieval, so this is a real loss.
  - **No provision numbers as a field.** The number sits at the head of the text
    in either numeral system, terminated by either danda (U+0964 and U+09F7 both
    occur, sometimes in one Act), sometimes behind a footnote marker like `55[`.
  - **`is_repealed` is null for all 1,484 acts.** Repeal status is derived from
    the title here instead.
  - **No chapter (অধ্যায়).**

Result: 3,352 chunks over 3,017 provisions from 26 acts across 11 domains, no
chunk above 400 words, no chunk missing a citation. Two acts still need a
targeted bdlaws pass: সাইবার নিরাপত্তা আইন ২০২৩ (act-1457 exists in BLAD with
zero provisions) and সড়ক পরিবহন আইন ২০১৮ (absent entirely).

## 2026-08-22 — The corpus is 60% English, and that changes the research claim

Measured, not assumed: 2,010 of 3,352 chunks are English. It splits by the age of
the Act, because older Bangladeshi law has never been published in Bangla.

| Bangla-only | English-dominant |
|---|---|
| labour, constitutional, consumer, women_children, cybercrime, tenancy, civil_registration | criminal_procedure (100%), family (93%), land (89%), money_recovery (65%) |

Family and land are two of the highest-volume citizen question domains. So in the
areas where citizens ask most, the law they need is in English while the question
they ask is in Bangla.

This is Risk 1 from the proposal arriving exactly as forecast, and it selects the
proposal's own fallback (c): a mixed corpus with language as a metadata field and
a reported per-language result split. Dhara therefore measures **two** gaps, not
one — the colloquial/formal **register** gap, and a Bangla-query/English-provision
**language** gap that is unavoidable for the busiest domains. Reporting both,
split by language and register, is a stronger and more clearly novel contribution
than register alone, and it separates this work further from MINA and LegalRAG,
which evaluate only on Bar Council exam questions.

Consequence for the models: the checkpoint choice now has to weigh cross-lingual
alignment, which moves LaBSE and BGE-M3 up relative to a Bangla-only encoder.

## 2026-08-22 — Question mining: 612 articles, 585 candidate questions

Five sources cleared and collected: Prothom Alo পাঠকের উকিল (92 articles) and
পাঠকের প্রশ্ন: আইন (24), Ajker Patrika আইনি পরামর্শ (71), Lawyers Club Bangladesh
দৈনন্দিন জীবনে আইন (382), Daily Star Your Advocate (43). Every per-source count
matches the independent enumeration in `docs/research/question-sources.md`
exactly, which is the only cheap check available that the parsers are right.

Four sources were excluded on their own published terms, and the reasons live in
the `src/dhara/mine.py` docstring so nobody re-adds them: Jagonews24 sets
`Content-Signal: ai-train=no`; Quora's robots.txt forbids using content to train
models; Reddit's is `Disallow: /`; Meta's ToS forbids automated collection and
the Groups Graph API permissions were withdrawn in April 2024. Facebook questions
can only be collected by a person reading public posts and rewriting them, which
the paraphrase policy requires anyway.

Splitting was harder than expected and the finding is worth recording. Prothom
Alo's column has been migrated between CMSes repeatedly, and 81 of 92 articles
separate reader letters from lawyer replies using **Private Use Area codepoints**
(mostly U+F032 and U+F06C) — Wingdings bullets that lost their font. Until those
were handled the articles looked like single undifferentiated 600-word blobs and
the splitter recovered 36 questions instead of 313.

## 2026-08-22 — Six further everyday domains added, each carrying a risk tier

Added: criminal procedure (FIR, GD, arrest, bail), women & children protection
(নারী ও শিশু নির্যাতন দমন আইন ২০০০, dowry, domestic violence), urban tenancy
(Premises Rent Control Act 1991), civil registration (birth/death registration,
NID correction), road transport (Road Transport Act 2018), and money recovery
(cheque dishonour, Artha Rin Adalat).

The uncomfortable part, recorded deliberately: everyday frequency and potential
harm rank in the same order. The law people meet most is the law where a wrong
answer costs most, and the original guide dropped the Penal Code for exactly this
reason. Excluding these domains does not stop citizens asking about them — it
means they get worse answers elsewhere. Shipping them with the same neutral
presentation as a weekly-holiday entitlement is the version that actually causes
harm.

So every domain carries a `risk_tier` in `configs/domains.yaml`. For high-tier
domains the interface leads with a referral to legal aid **above** the retrieved
provisions, not as a footnote. Retrieval behaviour is unchanged; only the framing
differs. The abstention threshold is tuned separately per tier, biased further
toward abstaining on high-tier queries.

## 2026-08-22 — The paper is a resource paper; BLP first, LREC fallback

The defensible novelty is the dataset, not the method: the first Bangla
citizen-question → statutory-provision retrieval resource with a measured
colloquial–formal register split. "We fine-tuned a bi-encoder" surprises no
reviewer. Framing it as a resource paper makes the register split and the
lexical-overlap audit the core tables rather than supporting material.

Venue: BLP (Bangla Language Processing) is both the nearest fit and the strongest
realistic target — its 2nd edition ran at IJCNLP-AACL 2025, a 2026 edition is not
yet announced. LREC is the fallback and is arguably the better home for a
resource paper. Note for honesty: NLP venues are not ranked by impact factor;
workshops have none at all. Prestige here runs ACL/EMNLP main > Findings >
LREC/COLING > topical workshops, and a well-executed BLP resource paper is worth
more than a rejected EMNLP submission.

## 2026-08-22 — Mined questions: verbatim kept local, paraphrase released

A resource paper means publishing the data, and the mined questions come from
copyrighted newspaper columns. Redistributing 200 verbatim reader questions is
infringement, not a technicality.

Every mined question stores three things: `source_url`, `text_verbatim` (kept
locally, git-ignored, never released), and `text_bn` — a paraphrase written to
preserve register while changing wording. The public release carries the
paraphrase, the URL, and the provision label. A `paraphrased` boolean makes the
ratio reportable. Decided now rather than at release time because retrofitting it
means re-reading every source.

## 2026-08-22 — Coverage is wide, evaluation is deep

The corpus covers every domain (~2,500–3,500 chunks; one overnight crawl, and it
is the resource contribution). The gold questions concentrate on five headline
domains — family, land, labour, consumer, cybercrime — with the remaining domains
retrievable but reported only in aggregate. Spreading 200 gold questions across
eleven domains leaves ~18 each, and the per-domain × register table stops meaning
anything. Stated explicitly in the paper as a coverage-versus-evaluation split.

## 2026-08-22 — Cybercrime added as a domain; scope rule changed to everyday frequency

The original four-domain split was chosen by legal taxonomy and it omitted the
second most common thing Bangladeshi citizens actually ask lawyers about.
Prothom Alo's round-up of the legal questions its readers sent in 2023 reports
family law and **cybercrime** as the two largest categories, so a corpus of
"citizen-facing law" without cybercrime was mis-specified, not merely narrow.

Decision: add `cybercrime` (Cyber Security Act 2023 and related — online
harassment, defamation, account takeover, mobile financial-service fraud), and
replace the selection rule itself. Domains are now chosen by **how often an
ordinary person collides with them**, not by taxonomy. `configs/domains.yaml`
becomes the authoritative list and the docs point at it rather than enumerating
domains inline, so future additions do not require editing six files.

Accepted cost: corpus size, synthetic-generation volume, and gold-set coverage
all scale with the domain count, and gold questions spread thinner across the
per-domain × register table. Further additions each need their own entry here,
because the highest-frequency domains are also the highest-harm ones.

## 2026-08-22 — Build for a conference paper, not only the course submission

The course submission lands late September / October 2026. A conference paper is
also intended. Where the two disagree, the paper's bar governs: a locked
zero-shot control, significance testing rather than point estimates, a released
dataset with a stated licence, a measured agreement number, and a real ethics
statement. Every one of those is already required by the course plan, so this
costs nothing extra now and avoids re-running experiments later to satisfy a
reviewer. Venue and deadline are not yet chosen.

## 2026-08-22 — Gold set: 200 questions, BM25-sourced candidates, human adjudication

The user chose the maximum gold-set size for accuracy. An agent may propose
candidates but never decides the answer: the same model family writes the
synthetic training questions, so model-decided gold answers would make the
headline claim circular.

Candidates are drawn from **BM25**, not from a dense model. This biases the gold
set toward provisions a lexical retriever can find, which makes the project's own
thesis harder to prove — the conservative direction. 15 questions are annotated
with no candidate list at all to measure how often the BM25 list missed the true
answer; that miss rate is reported. Each gold question carries
`annotation_mode` ∈ {assisted, unaided} and `source` ∈ {mined, authored}.

## 2026-08-22 — Intra-annotator agreement now, inter-annotator κ deferred

With a single annotator, inter-annotator κ is not a number that exists. Rather
than skip agreement entirely, 30 gold questions are re-annotated blind after a
gap and reported as intra-annotator (test–retest) reliability, stated plainly as
the weaker measure it is. When a second annotator joins, they double-annotate 50
and both numbers are reported. Recorded here so the gap reads as a plan rather
than an omission.

## 2026-08-22 — No difference is claimed without a confidence interval

At 200 gold questions the 95% CI on Recall@5 is roughly ±7 points, so adjacent
rungs of the ladder will sometimes be indistinguishable. `metrics.py` implements
a paired bootstrap over the `per_query` records every run already stores, and
every rung-to-rung comparison reports the CI on the difference. Costs about
twenty lines and no runtime. A rung pair honestly reported as within noise reads
as more credible than a bar chart narrated as if every gap were real.

## 2026-08-22 — Canonical vocabulary fixed before any schema field is named

Three words carried multiple meanings across the proposal and the implementation
guide, and each was also a JSON field name. Settled in `CONTEXT.md`: `domain`
means the area of law only (the machine-learning sense is always written out as
"domain-specific fine-tuning"); `intent` is dropped as a separate concept because
§5.5 already makes it the same label as `domain`; **provision** becomes the
umbrella term for both ধারা and অনুচ্ছেদ, with **section** and **article**
reserved for their exact senses; **chunk** is the unit that is embedded and
**provision** is the unit that is cited. Cost: the guide's own prose uses
"section" loosely, so quoting it verbatim in the report needs a light edit.

## 2026-08-22 — Standalone amendment Acts and repealed Acts are excluded from the corpus

The bdlaws portal publishes amendment Acts as separate Acts with their own pages
— e.g. `act-1672` (বাংলাদেশ শ্রম (সংশোধন) আইন, ২০২৬) amends the Labour Act 2006,
and `act-1578` is a 2025 ordinance already marked `[রহিত]`. Their provisions read
"in section 4, for the words X substitute Y", which is meaningless and
potentially harmful as an answer to a citizen question.

Decision: index only consolidated principal Acts. Skip any act whose portal title
contains `সংশোধন`, `(Amendment)`, `[রহিত]`, or `[Repealed]`, and skip provisions
marked `[বিলুপ্ত]` inside a retained Act. Record the excluded count and the
exclusion rule in `corpus_stats.csv` and state both in the report.

The honest ideal — indexing principal Acts and carrying per-provision amendment
status in metadata — was rejected on cost, not on merit. The consequence is that
Dhara returns the portal's consolidated text as of `crawl_date` and cannot tell a
user that a provision was amended after that date. This is a stated limitation,
and it is why `crawl_date` is displayed in the interface.

## 2026-08-22 — bdlaws portal facts confirmed by probe, not assumption

No `robots.txt` exists (a soft 404). The site is HTTP-only; port 443 refuses
connections, so the crawler cannot use HTTPS. Language is a query parameter
(`?lang=bn` / `?lang=en`), not a separate URL. The chronological index carries
~5,025 act links. Provisions have their own pages (`/act-952/section-28053.html`)
and the id in that URL is an opaque global integer, not the provision number —
so `section_no_bn` must be parsed from page content and the URL id is stored
separately as a stable key.

## 2026-08-25 — The danda was a BM25 term, and it was driving the ranking

`aggressive()` pads the danda `।` into a standalone token because Word2Vec's
context window needs to see the sentence boundary. `content_tokens()` then passed
it straight through to BM25, where it occurred in 3,980 of 6,359 corpus chunks.
Every long provision therefore collected a free term match, and document length
rather than topic drove the ranking: an inheritance question returned
ভূমি সংস্কার আইন, উপজেলা পরিষদ আইন and ট্রেডমার্ক আইন as its top three.

Fixed by dropping punctuation tokens in `content_tokens()` rather than in
`aggressive()`, so the Word2Vec contract is unchanged. Top-5 for the same query
now matches on বোন / ভাই / সম্পত্তি / খালা / নানা.

Caught while auditing the first generated annotation sheets. It is exactly the
failure mode the guide warns about — normalization bugs are invisible, the text
looks fine and retrieval is quietly several points worse — and it would have been
baked into 794 human labels had the sheets gone out unread.

## 2026-08-25 — The exclusion rule in configs/domains.yaml was not what the code enforced

`domains.yaml` lists ইনষ্টিটিউট among `exclusions.title_contains`, with a comment
explaining that institute-establishment acts are not citizen-facing. The regex in
`blad.py` never included it, so 22 chunks of জাতীয় স্থানীয় সরকার ইনষ্টিটিউট আইন,
১৯৯২ sat in the corpus and surfaced in BM25 candidate lists. The config was
right and the code disagreed with it silently.

Regex corrected. The wider lesson recorded here: the exclusion rule is stated in
two places and nothing checks that they agree. Corpus rebuild is deferred until
the provision-title crawl finishes, so both fixes land in one rebuild.

## 2026-08-25 — Two crawlers were writing to one output file

Two `02_scrape.py` processes over the same act list were running concurrently
against `data/interim/bdlaws_titles.jsonl`, launched about an hour apart. The
later one opened the file in write mode and truncated it while the earlier one
held a handle at a stale offset, so the JSONL interleaved and the observed line
count went *down* between two reads. Both were also fetching the same pages, so
the effective request rate on bdlaws was double the 1.5–2s the scraping-conduct
rule specifies.

Both killed and one relaunched. Nothing was lost: the crawl resumes from the
13,073 archived HTML pages under `data/raw/bdlaws/`, and the JSONL is rewritten
from that archive on every run.

Fixed the same day rather than left as a note: `02_scrape.py` now writes
`<out>.lock` before crawling and removes it in a `finally`, and refuses to start
when the lock already exists unless `--force` is passed. Roughly fifteen lines
against a failure that is silent, corrupts the output, and doubles the load we
put on someone else's server.

## 2026-08-25 — The annotation sheet gets three columns it did not have

`annotator`, stamped into every row rather than left implicit in the filename, so
authorship survives concatenation and each person's contribution is separately
countable in the report. Four people are labelling; the methodology section
should be able to say who did what as a number.

`answer_source` ∈ {candidate_list, own_search, none}. About a third of the corpus
is English-only, and BM25 cannot match a Bangla question to an English provision,
so a Bangla inheritance question has no reachable answer in a candidate list
built by BM25. Without this column an answer found by the annotator's own search
is indistinguishable from one the list supplied, and the BM25 miss rate — a
headline number for the lexical-gap claim — cannot be computed at all.

`annotation_mode` ∈ {assisted, unaided}, promoted from a flag inside `block` to
its own column so it can be read without string-splitting.

Candidate lines now carry `<chunk_id>` explicitly, so an answer found by the
annotator's own search is addressable in the same field as a candidate rank.

## 2026-08-25 — Answer agreement is reported as raw percentage, not as kappa

Cohen's κ corrects for agreement expected by chance. With 6,359 candidate chunks,
chance agreement is effectively zero, so κ on the *answer* field reduces to raw
percentage agreement and reporting it as κ dresses a plain percentage in borrowed
authority. `08_merge_gold.py` therefore reports the answer at two levels of raw
agreement — exact chunk-set match, and same parent provision, because two
annotators choosing different sub-chunks of one long section have not disagreed
about the law — and reserves Cohen's κ for `domain` and `register`, which have a
handful of categories each and are what κ is for.

## 2026-08-25 — One annotation manual, and the code follows it rather than the reverse

A second manual (`docs/ANNOTATION.md`) was written before noticing that
`docs/annotation_guideline.md` already covered the same ground and covered the
legal judgement considerably better — the two-pass structure, the English-act
table, the register examples, the ranked list of common mistakes. Two overlapping
manuals is worse than either alone, because annotators follow whichever they
opened. The second was deleted and the parts of it that were genuinely new — the
search tool, the merge command, the Excel encoding warning — folded into the
existing guide.

Comparing them surfaced six places where the guideline and `08_merge_gold.py`
disagreed about the accepted vocabulary: `candidate` vs `candidate_list`,
`3/2/1/?` vs `high/medium/low` for confidence, a two-valued vs three-valued
`register`, whether `other` is a legal domain, whether `unanswerable` is distinct
from `none`, and the three-answer cap. Every one would have failed validation in
bulk on the first real merge.

Resolved in favour of the guideline in all six cases, because it is the contract
the humans are handed and the code is the cheaper thing to change. Two of its
choices are better than what the code had, not merely different:

  - **`register` is two-valued, not three.** The guide gives an explicit tie-break
    for the mixed case — one borrowed legal word does not make someone a lawyer,
    tag it colloquial. A two-way split with a stated rule agrees better than a
    three-way split with a fuzzy middle.
  - **`unanswerable` and `none` are different labels.** `unanswerable` means no
    law answers this; `none` means the annotator could not find it. Only the
    first calibrates the abstention threshold. Only the second is a retrieval
    miss. The merge script had collapsed them, which would have mixed a property
    of the corpus with a property of the annotators inside one number.

This is the third config/code divergence found today, after the exclusion rule
and the two crawlers. Nothing in the repo checks that a document and the code
implementing it still agree.

## 2026-08-25 — Portal coverage audited against the chronological volume index

Checked the 1,675 acts in the portal's chronological index (59 volumes, Regulation
V of 1799 through the 2026 acts and ordinances) against the 1,444 in
`corpus_full_v1.jsonl`. The 231 absent break down as 194 amendment acts, 35
repealed acts, and 2 genuine gaps. The recent-year gaps that look alarming at
first — 67 of 79 acts absent for 2025, 96 of 139 for 2026 — are almost entirely
`(সংশোধন) অধ্যাদেশ … [রহিত]`: amendment ordinances that were subsequently
repealed, excluded twice over by rule. Coverage against the portal is therefore
complete to within two acts.

The two genuine gaps:

  - **act-1, The Districts Act, 1836.** In BLAD with a single section whose text
    carries no section number at all, so the chunk was dropped. Administrative
    rather than citizen-facing; not in any configured domain.
  - **act-1733, ইনভেস্ট বাংলাদেশ আইন, ২০২৬.** In the portal index, absent from
    BLAD, never fetched — it post-dates every crawl. Also outside the configured
    domains.

Neither is worth a crawl on its own. Both are recorded so the coverage claim in
the paper can be stated as "1,444 of 1,675, with 229 excluded by rule and 2 known
gaps" rather than as "complete".

## 2026-08-25 — 505 section records were being dropped for an unparseable number

The coverage audit turned up something the act-level count hides: 505 of 35,633
section records in retained acts (1.42%) parse to an empty provision number and
are dropped by `parse_provision`'s caller. 139 of those are in acts that are in
the evaluation corpus — 109 criminal_procedure, 14 money_recovery, 11 land, 2
family, 2 constitutional, 1 women_children.

Classified, they are four different problems wearing one symptom:

  - **482 unnumbered fragments**, overwhelmingly the `First.- / Second.- /
    Thirdly.-` enumerations of nineteenth-century English acts. These are
    continuation text belonging to the numbered provision above them, so the
    parent provision is present in the corpus but **truncated**.
  - **12 provisions that are numbered**, in shapes the regex did not cover.
  - **9 continuation clauses** (`shall, if the offence be committed…`) that the
    existing `CONTINUATION` rule does not catch because they open with a word
    rather than a `(n)` marker.
  - **2 preambles**, correctly dropped.

Fixed now: `PROVISION_HEAD` widened to accept a stray leading dot or tab
(`.301.`, `.\t72.`), a hyphen before the letter suffix (`171-I.`, `70-B.`,
`265-I.`, `19-I.`) and three-character suffixes (`44CCC.`, `১৫ককক।`). Recovers 13
provisions against zero changed parses, measured over all 35,596 candidate
records before applying. Among them are Registration Act 70-B to 70-F, which is a
headline `land` domain.

Deliberately **not** fixed now: reattaching the 482 orphan fragments to their
parent provisions. That changes `text_bn`, therefore word counts, therefore
whether a provision splits into sub-chunks, therefore `chunk_id` — and a
`chunk_id` that moves after annotation has begun silently invalidates gold
answers with no error raised. The window for that change closes when the first
annotator starts. Either it happens before the sheets go out or it becomes
`corpus_v2` with every affected result re-run.

Stated as a limitation regardless: the truncation is concentrated in old English
acts, and criminal_procedure is a `coverage` domain rather than a headline one,
so the effect on the reported comparison is small but not zero.

## 2026-08-25 — A dropped connection was discarding a whole act's crawl

The title crawl finished reporting `acts with content 51/57`. Six acts —
শিশু আইন (1119), স্থাবর সম্পত্তি অধিগ্রহণ (1220), রংপুর মহানগরী পুলিশ (1239),
বাংলাদেশ ইপিজেড শ্রম আইন (1285), কপিরাইট আইন (1452) and act-1207 — produced zero
title rows despite having their index pages parsed correctly and 9 to 97 section
pages already sitting in the archive.

Two defects, one symptom:

  - `act()` caught only `requests.HTTPError`. bdlaws drops connections under
    sustained crawling, and both failure modes seen in the log —
    `ConnectionResetError(10054)` and a refused connect (`WinError 10061`) — are
    `requests.ConnectionError`, a sibling of `HTTPError` rather than a subclass.
    So a transient blip escaped the per-section handler entirely.
  - `acts()` bound `found = list(act(act_id))` *inside* its try/except. When the
    escaped exception arrived, `continue` discarded every provision already
    fetched for that act. Partial work was thrown away in full.

Fixed: transient `requests.RequestException` now retries three times with linear
backoff before giving up on a single section, while `HTTPError` still fails fast
because a 404 will not become a 200. `acts()` accumulates into a list as it goes
and yields whatever it collected even when interrupted.

The failure was invisible in the obvious place. The output file looked healthy —
5,145 rows, 99% carrying titles — and only the `51/57` line and a per-act count
of zero gave it away. Worth remembering that a crawl's summary line is a better
integrity check than the size of its output.

## 2026-08-25 — Repealed acts were in the corpus, including two thirds of cybercrime

The repeal marker lives in the **portal's** title, not in BLAD's. The portal says
কপিরাইট আইন, ২০০০ [রহিত]; BLAD stores কপিরাইট আইন, ২০০০. `EXCLUDE_TITLE` was
being run over BLAD's title, so it could not see the marker, and every repealed
act that BLAD happened to carry walked into the corpus unchallenged.

Measured: **4 repealed acts / 316 chunks in the evaluation corpus** (5% of it),
and **215 acts / 7,849 chunks in the full corpus** (16.6%). The worst of the full
set are The Income-tax Ordinance 1984, The Customs Act 1969 and The Insurance Act
1938 — all long superseded, all indexed as if current.

The evaluation-corpus four, and what each needs:

| In corpus | Successor | Action |
|---|---|---|
| কপিরাইট আইন, ২০০০ [রহিত] | act-1452, ২০২৩ — indexed | drop the old one |
| মাদকদ্রব্য নিয়ন্ত্রণ আইন, ১৯৯০ [রহিত] | act-1276, ২০১৮ — indexed | drop the old one |
| সাইবার নিরাপত্তা আইন, ২০২৩ [রহিত] | act-1710, ২০২৬ — **absent** | drop, and crawl 1710 |
| সাইবার সুরক্ষা অধ্যাদেশ, ২০২৫ [রহিত] | act-1710, ২০২৬ — **absent** | drop, and crawl 1710 |

The cybercrime finding is the serious one. That domain held 220 chunks, of which
129 — act-1457 and act-1538 — are repealed law, while the act that actually
governs today, সাইবার সুরক্ষা আইন, ২০২৬ (act-1710), was never indexed at all. So
a citizen asking a cybercrime question would have been answered from a repealed
statute, in a `risk_tier: high` domain whose own note reads "this is the law
people seek protection under AND the law people are prosecuted under". That is
precisely the harm the ethics constraint names, and it was live.

Fixed: `repealed_on_portal()` checks the portal index regardless of which source
supplied the text, and is called on both the BLAD and the bdlaws build paths.
`configs/domains.yaml` now lists act-1710 and names the three repealed cyber acts
explicitly as excluded, so the next person does not re-add them.

Cybercrime has been legislated four times in eight years — DSA 2018, CSA 2023,
the 2025 ordinance, the 2026 act. Every predecessor is repealed. The currency of
that list matters more than anywhere else in the config, and it deserves a
re-check before submission rather than trust in this snapshot.

## 2026-08-25 — act-1710 was already crawled; the gap was the config, not the data

Correcting the entry above. সাইবার সুরক্ষা আইন, ২০২৬ was not missing from the
crawl at all — 52 provisions of it were already sitting in
`data/interim/bdlaws_missing.jsonl`. What was missing was its line in
`configs/domains.yaml`, so it never matched the cybercrime domain and never
reached the corpus. Re-crawling it into a second file duplicated every one of its
provisions and put 68 chunks into the corpus sharing a `chunk_id` with another
chunk.

That is worse than it sounds. `chunk_id` is the answer key an annotator writes
down; two chunks sharing one makes the gold answer ambiguous with nothing raising
an error. Caught only because the search tool printed each cyber provision twice
in a listing.

The redundant crawl file was deleted and `_bdlaws_rows` now dedupes on
`(act_id, section_id)` across all source files, first occurrence winning. The
crawl passes overlap by design — an act fetched to fill one gap can be re-fetched
by a later pass aimed at another — so the guard belongs there permanently rather
than in a one-off cleanup.

Lesson worth keeping: the fix for a missing act is to check the config before
reaching for the crawler. The data was already on disk.

## 2026-08-25 — Candidate lists are deduplicated by provision and show a body snippet

Two changes to `06_make_annotation_sheets.py` after reading the generated sheets
as an annotator would.

A long provision is split into overlapping sub-chunks that share one citation,
and BM25 returns several of them happily. Three consecutive lines reading `ধারা ২`
spend a third of the annotator's ten slots re-offering one provision they have
already rejected. Candidates are now collapsed to the best-scoring sub-chunk per
provision — over-fetching 4x to still fill ten — so ten lines mean ten genuinely
different answers.

And the line now carries the provision title *and* a 110-character body snippet.
Titles were worth crawling because they are the densest formal legal vocabulary
in the corpus, but the most frequent of them are `সংজ্ঞা` and `প্রয়োগ`, which
tell an annotator nothing. The title identifies the provision; the snippet is
what says whether it answers the question.

## 2026-08-27 — Pass 1 changed from authoring to verification, and the guide had to change with it

`09_autolabel_pass1.py` filled every row's `answer`/`answer_source`/`register`/
`domain`/`confidence` before any human touched the sheets. `docs/annotation_guideline.md`
still described Pass 1 as picking an answer from a blank cell, which is no longer
what the task is — and a guide describing the wrong task is worse than no guide,
because it tells annotators to trust the blank-authoring workflow's pace and
skip the checking the new workflow actually needs.

Rewrote the top of Pass 1 rather than add a second document: same lesson as the
ANNOTATION.md/annotation_guideline.md duplication earlier this week. States the
measured failure rate up front (15% Act-level accuracy against advocate-named
Acts; 0/5 on English-source answers in a 7-question probe) so annotators know
*why* they cannot skim a pre-filled row, not just that they shouldn't.

Added a `verified` column to all four sheets, blank by default. Without it there
was no way to tell "a human checked this" from "the machine filled this in," and
every row currently satisfies the second without the first. `08_merge_gold.py`
now gates agreement, the candidate-list miss rate, and the unanswerable count on
`verified` rows once any exist, and prints a loud warning instead of numbers
when none do — confirmed necessary by testing: at zero verified rows the old
code reported 1.000 agreement and a 0.000 miss rate, which is the auto-labeller
agreeing with itself, not a measurement. Tested the gate with a simulated 30%
verification pass; the agreement pool dropped from 270 pairs to 33 as expected.

Also added a `## Working with Claude while you verify` section. The load-bearing
line in it: "a machine may propose, a human decides" applies to Claude exactly as
it applies to the auto-labeller — an AI-verified gold set is the same circularity
problem as an AI-labelled one, just one layer removed. The section is written to
make Claude fast at the slow parts (searching, quoting provision text, comparing
candidates) without making it the one who decides. The worked example in that
section was checked against real corpus text before being written down — an
earlier draft of it misdescribed a real chunk_id's content, which is exactly the
failure mode the section warns against and would have been a bad thing to ship
inside it uncaught.

## 2026-08-27 — Annotators should never have to remember or type a chunk_id

A real annotator, verifying a real row, found the correct law by their own
research (Muslim Marriages and Divorces (Registration) Act, 1974, sections 3-5,
plus Muslim Personal Law (Shariat) Application Act, 1937, section 2) and had no
way to enter it without opening a terminal, running the search tool, and
hand-copying an internal `chunk_id` — a real usability failure, not a training
gap. Chunk_ids are an implementation detail; nobody doing legal research thinks
in them.

Added `ActIndex` to `08_merge_gold.py`: `resolve()` now accepts a plain citation
— "Muslim Marriages and Divorces (Registration) Act, 1974, section 3" or
"সালিস আইন ধারা ৩৪" — and matches the Act name against every title in the
corpus, scored as the fraction of the *typed* words found in the full title (so
an abbreviation like "Shariat Act section 2" still resolves). Tried only after
the existing `<chunk_id>` marker and bare-rank/chunk_id paths fail, so nothing
already working changes.

**Found and fixed a real collision risk before it could bite anyone.** Bangladesh
re-titles the same generic name every year: 26 groups of acts in this corpus
share an identical name once the year is dropped — 14 separate "Finance Act"s,
5 Chittagong Hill-Tracts regulations. An early version of the matcher picked
whichever act scored highest with no tie-check, which meant "Finance Act
section 5" without a year would have silently resolved to an arbitrary one of
fourteen unrelated Acts — a wrong answer with no warning, which is worse than
refusing to guess. Fixed: a tie at the top score now returns no match at all,
same as finding nothing, and the caller's error tells the annotator to add the
year or paste a `<chunk_id>`. Verified this also catches a subtler case: "Finance
Act 1974" alone still ties against "The Refugees Rehabilitation Finance
Corporation (Repeal) Act, 1974", which happens to contain both words in an
unrelated title — the short-fragment scoring is inherently loose, and the
tie-detector is what keeps that looseness from becoming a silent wrong answer.

Also documented rather than solved: the matcher only matches the language the
corpus actually stored the title in. Most English colonial-era Acts have no
Bangla title in this corpus at all, so a colloquial Bangla name (দণ্ডবিধি for
the Penal Code) will not match — the annotator has to use the English name for
those. Building a Bangla-name alias table for every English-only Act was judged
not worth the effort against how rarely it will actually come up; the failure
mode is a clear "could not identify" error, not a wrong answer, so it is safe
to leave as a documented limitation.

## 2026-08-27 — Claude generates a Pass-1 suggestion column, in its own lane, never `answer`

The `09_autolabel_pass1.py` proposals in `answer` are lexical-retriever output and
measured near-useless on this task (15% Act-level accuracy against advocate-named
Acts; the CALIBRATION rows show police Acts and bankruptcy Acts proposed for
marriage questions). Verifying from that starting point is barely faster than
starting blank, because the annotator has to discard the proposal first.

Decision: Claude does a full second proposal pass over all 794 rows — read the
question, the recovered advocate reply, and a dense+BM25 candidate pool, then
name the provision(s) with a one-line rationale. This is a stronger proposer than
BM25 (semantic, cross-language, reads the advocate's reasoning), so the proposals
are worth more as a starting point.

The hard constraint, and why the column is separate: an AI-verified gold set is
the same circularity problem as an AI-labelled one (already recorded 2026-08-27,
"a machine may propose, a human decides"). If Claude's picks land in `answer`,
the annotator's job silently becomes confirm-or-edit, anchoring bias does the
rest, and the gold set measures agreement-with-Claude — which is fatal for a
paper whose claim is that a semantic model closes the colloquial→formal gap.
So the suggestions land in four NEW columns — `llm_suggest`, `llm_rationale`,
`llm_confidence`, `llm_flag` — and `answer` stays whatever it was. The annotator
forms their own answer with the suggestion as one input among the BM25
candidates and the advocate reply, and moves a `chunk_id` into `answer` only
after checking the provision text themselves.

UNAIDED rows get no suggestion at all (`llm_flag = UNAIDED-no-assist`): those
15-odd rows exist to measure how often every automated route missed the true
provision, and seeding them with Claude's guess destroys that instrument.

Disclosure: the paper's methodology section says the gold set is LLM-assisted,
human-adjudicated — Pass 1 proposals from a lexical retriever and from Claude,
every gold label verified against corpus text by a human, `pass1_auto` and the
`llm_suggest`→`answer` override rate both reported. This is an honest, publishable
process; hiding the Claude pass would not be. The override rate is the number
that shows the verification was real and not a rubber-stamp — if it is near zero,
the write-up says so.

Mechanics: CPU-only box, so dense retrieval over all 39,484 chunks with
`multilingual-e5-base` is a ~60-90 min one-time embed, cached to disk; the
per-question search is then a matrix multiply. BM25 runs alongside for the
lexical signal. Where the pool misses the true provision (the cross-language
case this project measures), Claude falls back to targeted `07_search_corpus`
queries — same escape hatch the human annotator uses.

## 2026-08-28 — Dense retriever changed from multilingual-e5-base to BGE-m3

The architecture assumed a multilingual sentence encoder would reach the
English-only Acts that a Bangla-only model cannot. A zero-shot probe over the 552
answerable gold questions (`data/processed/probe_questions.jsonl`, provision-level
recall) showed `multilingual-e5-base` does not deliver on that assumption:

| recall@10        | BM25 | e5-base zero-shot |
|------------------|------|-------------------|
| overall (552)    | 0.07 | 0.15              |
| English-gold (333, 60%) | 0.00 | 0.05       |
| Bengali-gold (108)      | 0.27 | 0.36       |
| mixed (111)             | 0.12 | 0.27       |

60% of the gold answers live in English-only Acts (MFLO 1961, DMMA 1939,
Succession Act, Penal Code, CrPC, Contract Act, TPA, Evidence Act — the
un-replaced colonial and early-independence private law), and e5-base reaches
them from a Bangla question 5% of the time. That caps the whole system.

`BAAI/bge-m3`, zero-shot, on an 8.3k-chunk probe pool (all gold-provision chunks
+ 8,000 distractors), same 552 questions:

| recall@10 (8.3k pool) | e5-base | BGE-m3 |
|-----------------------|---------|--------|
| overall               | 0.22    | 0.48   |
| English-gold          | 0.08    | 0.35   |
| Bengali-gold          | 0.53    | 0.72   |
| mixed                 | 0.34    | 0.61   |

BGE-m3 quadruples English-gold recall (0.08 → 0.35) and roughly doubles overall.
The cross-language gap is bridgeable; e5-base was simply too weak a model for it.
This removes the need for corpus translation as a prerequisite (translation stays
an optional experimental condition, per the 2026-08-23 entry).

Decision: `BAAI/bge-m3` is the dense retriever. Update `configs/models.yaml`,
`scripts/12_zeroshot_probe.py`, `src/dhara/retrievers/biencoder.py`. The
zero-shot BGE-m3 control is locked from a full-corpus run before any fine-tuning
(`notebooks/colab_bge_m3_eval.py`), so the headline comparison is fine-tuned vs
zero-shot BGE-m3, not neural vs lexical.

Accepted cost: BGE-m3 is 568M params (vs e5-base 278M) and 1024-dim (vs 768), so
the corpus embed must run on a T4 (~10 min) rather than the CPU box (~5 hours),
the dense index is ~33% larger, and fine-tuning needs LoRA or a small batch on
the T4. All acceptable. Pool-based recall figures above overestimate the
full-corpus number (8.3k haystack vs 39,484); the full-corpus run replaces them.

Also confirmed the BM25 ~93% candidate-list miss rate is real, not a bug:
normalization is symmetric, the retriever works. The miss decomposes into the
cross-language wall (English-gold: BM25 0% recall) and a genuine Bangla-to-Bangla
register gap (Bengali-gold: BM25 misses 73% at rank 20). Both are the thesis,
now measured separately. `k1`/`b` are still on the old 6k-chunk defaults and need
re-tuning on dev against the 39,484-chunk corpus, but tuning does not change the
decomposition.

---

## 2026-08-31 — Full-corpus zero-shot BGE-m3 control is locked

The T4 run of `notebooks/colab_bge_m3_eval.ipynb` completed: all 39,484 chunks
embedded with `BAAI/bge-m3` (fp16, 1024-dim, `max_seq_length=512`, document =
`"{provision_title_bn} {text_bn}"`, no prefix), all 552 probe questions embedded
the same way. Artifacts stored as `models/index_bge_m3_zeroshot_v1/`
(`embeddings.npy`, `chunk_ids.json`, `probe_query.npy`, `manifest.json`, plus the
notebook's own `colab_eval_results.json` as a provenance record).

`scripts/13_eval_dense_index.py` re-derives every number from those vectors on
CPU and writes `results/runs/bge_m3_zeroshot.json` **with `per_query`**, which
the notebook did not emit. The notebook's aggregate figures reproduce to within
0.01 (fp16 GPU top-k against fp32 CPU, different tie-breaking). From here the
run JSON is the citable source; the notebook output is not.

Provision-level recall on the full corpus, n=552:

| group        | n   | R@1  | R@5  | R@10 | R@20 | R@50 | R@100 |
|--------------|-----|------|------|------|------|------|-------|
| all          | 552 | 0.096| 0.239| 0.324| 0.413| 0.534| 0.627 |
| English-gold | 333 | 0.054| 0.159| 0.222| 0.288| 0.429| 0.531 |
| Bengali-gold | 108 | 0.213| 0.426| 0.546| 0.685| 0.741| 0.806 |
| mixed        | 111 | 0.108| 0.297| 0.414| 0.522| 0.649| 0.739 |

Three things follow, and the third is the one that changes a plan.

**1. The 8.3k pool overestimated, as expected.** Overall R@10 was 0.48 on the
pool and is 0.32 on the real 39,484-chunk haystack; English-gold 0.35 → 0.22.
The pool numbers are retired — they were a checkpoint-selection instrument, not a
result, and only the full-corpus figures are reportable. The *ordering* they were
used to decide (BGE-m3 ≫ e5-base) survives, so the switch decision stands.

**2. This is the denominator of the headline claim.** Fine-tuning is now measured
against 0.239 R@5 / 0.324 R@10 overall, locked before a single training step.
The index directory is frozen: a rebuild becomes a new directory, never an
in-place overwrite.

**3. Equal-weight RRF is worse than dense alone, and is not the default path.**

| R@k, n=552   | BM25  | BGE-m3 | RRF (equal weight) |
|--------------|-------|--------|--------------------|
| R@10 overall | 0.067 | 0.324  | 0.216              |
| R@10 English | 0.000 | 0.222  | 0.108              |
| R@10 Bengali | 0.250 | 0.546  | 0.444              |

Fusion loses on every slice, including Bengali-gold where BM25 is at its best.
This is not a bug in the fusion code — it is what reciprocal-rank fusion does
when one input is near-random: RRF weights by rank position only, ignoring score
and ignoring the fact that BM25 returns 0% relevant candidates on 60% of the
questions, so half of every fused score is noise that displaces good dense
candidates. The architecture in CLAUDE.md ("BM25 and dense in parallel → RRF")
was written before that asymmetry was measured.

Consequence for the pipeline: the default retrieval path is dense-only into the
reranker until fusion is shown to beat it. Fusion stays on the table as a
*weighted* variant (dense-dominant, BM25 as a tie-breaker for exact section
numbers and named acts, where lexical matching should genuinely help), evaluated
as an ablation with a paired-bootstrap CI on the difference. BM25 keeps its row
in the table regardless — an untuned lexical floor is still the floor, and its
0% English-gold recall is one of the things this project set out to measure.

Not fixed by this run, still open: BM25 `k1`/`b` are on old 6k-chunk defaults, so
the fusion inputs may improve slightly after tuning; that will not close a
0.216-vs-0.324 gap, but the ablation should use the tuned version.

`src/dhara/metrics.py` written this session (recall@k, MRR, paired bootstrap over
`per_query`) — previously specified but missing, and every rung-to-rung comparison
from here needs it.
