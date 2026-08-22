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
