# ধারা (Dhara)
### A Bangla Legal Retrieval System for Citizen Services
**Bridging the colloquial–legal lexical gap with fine-tuned contextual embeddings**

> *ধারা* — in law, a **section** of an Act; in ordinary Bangla, a **stream** or **current**. The system's unit of retrieval is the ধারা, and its purpose is to carry a citizen's question to it.

*Course: Natural Language Processing Lab · Supervisor: Shawon Sir*
*Team size: 4 · Submission: late September / October 2026 · Also targeting a resource paper (NLLP / BLP)*

---

## 1. Abstract

Bangladeshi citizens regularly need to know what the law says about inheritance, land mutation, workplace rights, consumer protection, or a bureaucratic procedure. The authoritative text exists and is publicly published — but it is written in formal, often archaic legal Bangla, while citizens ask questions in everyday colloquial Bangla. The two vocabularies barely overlap, so keyword search fails and general-purpose embeddings, never trained on Bangladeshi legal language, fail with it.

This project builds a retrieval system that takes a plain-Bangla citizen question and returns the exact law section that answers it, together with its official citation (Act name, section number, chapter). We construct our own section-level corpus from the Laws of Bangladesh portal, manufacture a question–section training set through a three-way strategy (mined, synthetic, hand-annotated), and evaluate a ladder of retrieval methods of increasing sophistication — BM25, self-trained Word2Vec, BiLSTM encoders, and a fine-tuned transformer bi-encoder with cross-encoder reranking.

The central measurable claim: **the lexical gap is real, it is quantifiable, and domain fine-tuning closes it.** We demonstrate this both numerically (Recall@k, MRR split by colloquial vs. formal query phrasing) and visually (embedding geometry before and after fine-tuning).

---

## 2. Problem Statement

Consider a real query:

> **Citizen asks:** "পুলিশ আমাকে ধরে নিয়ে গেছে, কিছু বলছে না — আমার কী অধিকার?"
>
> **The law says:** "গ্রেপ্তার ও আটক সম্পর্কে রক্ষাকবচ" — Constitution of Bangladesh, Article 33

There is **not a single shared content word** between the question and the correct answer. Any lexical matching method — TF-IDF, BM25, keyword search — is structurally incapable of connecting them. This is not an edge case; it is the normal condition for citizen legal queries.

Three properties make this a good NLP research problem rather than a search-engine exercise:

1. **The gap is semantic, not spelling.** It cannot be fixed with stemming or fuzzy matching. It requires representations that place "ধরে নিয়ে গেছে" near "গ্রেপ্তার ও আটক" in vector space.
2. **Off-the-shelf embeddings do not solve it.** Multilingual models see very little Bangladeshi legal text in pretraining. Their zero-shot performance is our second baseline, and beating it is our contribution.
3. **The gap is measurable.** By tagging test queries as colloquial or formal, we can isolate exactly how much of any model's improvement comes from closing this specific gap.

---

## 3. Objectives

**Primary**
1. Construct a clean, provision-level, citation-preserving Bangladeshi legal corpus covering the domains people meet in everyday life (`configs/domains.yaml`), built on the published BLAD dataset and completed from the bdlaws portal where BLAD is empty.
2. Construct a question–section dataset: ~2,500–3,000 training pairs and a 200-question hand-annotated gold evaluation set.
3. Implement and rigorously compare five retrieval configurations spanning the full course syllabus.
4. Fine-tune a transformer bi-encoder on the domain and quantify its improvement over both classical and zero-shot dense baselines.
5. Deliver a working input→output demo that returns retrieved sections with mandatory citations.

**Secondary**
6. Train an intent classifier that routes a query to its legal domain (supervised core), compared across all four representation types.
7. Cluster section embeddings without supervision and assess whether the discovered structure recovers the actual organisation of the law.

**Explicitly out of scope — no generation layer**

Dhara retrieves and cites; it never generates. An earlier draft listed a stretch
objective 8, "a grounded generation layer producing a plain-Bangla explanation".
That is retrieval-augmented generation, it has been **cut permanently**, and the
decision is recorded in `DECISIONS.md`.

The system's output is the provision as printed, with its citation. This is a
deliberate methodological position, not only a scoping one: the two closest
published systems, MINA (Findings of ACL 2026) and LegalRAG, are both RAG
pipelines evaluated on Bar Council examination questions. Not generating is part
of what distinguishes this work from them, and a retrieval system can be measured
where a generated answer cannot.

---

## 4. Scope and Delimitations

**The scope rule: frequency in ordinary civilian life.** Coverage is chosen by how often an ordinary person actually collides with a body of law, not by legal taxonomy. A tidy four-domain split that omits the questions people actually ask is worse than an untidy wider one that catches them.

**In scope — confirmed domains:**

| Domain | Representative sources |
|---|---|
| Family & inheritance | Muslim Family Laws Ordinance 1961, Muslim Marriage & Divorce Registration Act 2012, Hindu/Christian personal law provisions, succession rules |
| Land & property | Registration Act, State Acquisition & Tenancy Act, নামজারি (mutation) procedure |
| Labour & employment | Bangladesh Labour Act 2006 (selected chapters), wage board notifications |
| Consumer & information rights | Consumer Rights Protection Act 2009, Right to Information Act 2009 |
| **Cybercrime & digital rights** | Cyber Security Act 2023, online harassment, defamation, account takeover, mobile financial-service fraud |

Plus the **Constitution of Bangladesh** (~153 articles) as a cross-cutting source, since fundamental-rights queries appear across all domains.

Cybercrime is included on evidence rather than intuition: Prothom Alo's own round-up of the legal questions its readers sent in 2023 reports that family law and **cybercrime** were the two largest categories. A corpus that omits the second-largest thing citizens ask about is not a corpus of citizen-facing law.

**Under expansion.** The domain set is deliberately open. Further candidates, ranked by everyday frequency, are criminal procedure (FIR, GD, arrest, bail), women & children protection (নারী ও শিশু নির্যাতন দমন আইন 2000, dowry, domestic violence), urban tenancy (Premises Rent Control Act 1991), road transport (Road Transport Act 2018), civil registration (birth/death registration, NID correction), and money recovery (cheque dishonour). The authoritative list is `configs/domains.yaml`; each addition is recorded in `DECISIONS.md` with its reason and its risk assessment, because the higher-frequency domains are also the higher-harm ones.

**Explicitly out of scope**
- Complete coverage of all ~1,300+ Acts on the portal.
- Case law, judgments, and precedent.
- Multi-turn conversation or dialogue state.
- Any claim to give legal advice (see §12).
- Any generated or paraphrased answer (see §3).

**On language — this changed once the corpus was measured.** English legal text
was originally listed as out of scope with a fallback caveat. It is not a
fallback: 39% of the corpus is English because Bangladeshi law enacted before
about 1987 was written in English and has never been translated. Verified on the
portal in both directions, `?lang=` switches the interface chrome only and never
the text of an Act, so no bilingual version exists for any Act. Criminal
procedure is 100% English, family 14% Bangla, land 29% — and family and land are
two of the domains citizens ask about most. The corpus therefore carries
`language` as a field and every result is reported split by it. This makes the
project measure **two** gaps: the colloquial/formal register gap, and a
Bangla-query/English-provision language gap.

**Corpus size, as built.** Two artifacts are kept. The everyday-life corpus is
~6,100 chunks over the domains in `configs/domains.yaml`, and it is the default
for evaluation. The full corpus is ~37,700 chunks over 1,400 Acts, and it is the
released resource. Evaluation stays on the smaller one deliberately: a 35k-
provision index adds confusable near-duplicate provisions and makes retrieval
harder without making the measurement better.

---

## 5. Corpus Construction (Methodology Chapter)

### 5.1 Sources

Primary source is the Ministry of Law's Laws of Bangladesh portal (bdlaws), which publishes Acts as structured HTML rather than scanned images — this eliminates OCR entirely and is the single biggest practical advantage of choosing this domain. Secondary sources are procedural guidance pages from the relevant service portals (NID correction, land mutation, birth registration) and published legal-aid FAQ material.

**Before scraping:** check the site's terms of use and robots.txt, rate-limit requests politely (1–2 seconds between requests), and identify your crawler honestly. **Record the crawl date** — laws are amended, and a dated snapshot is the honest scientific claim.

### 5.2 Pipeline

```
bdlaws HTML  →  raw archive (.html, unmodified)
             →  section splitter (ধারা / article boundaries)
             →  chunk records with metadata
             →  normalization pipeline
             →  corpus_v1.jsonl  [FROZEN]
```

Keep the raw HTML. When your splitter turns out to be wrong in Week 4 — and it will be, on at least one Act with unusual formatting — you re-run from the archive instead of re-crawling.

### 5.3 Chunk schema

The unit of retrieval is the **section (ধারা)**, not the document. One section = one embedding = one citable answer.

```json
{
  "chunk_id": "labour_2006_s103",
  "act_name_bn": "বাংলাদেশ শ্রম আইন, ২০০৬",
  "act_year": 2006,
  "chapter": "নবম অধ্যায়",
  "section_no": "১০৩",
  "section_title_bn": "সাপ্তাহিক ছুটি",
  "text_bn": "...",
  "domain": "labour",
  "source_url": "...",
  "crawl_date": "2026-08-20",
  "char_len": 412
}
```

The metadata *is* the citation. A retrieved chunk that cannot be cited is useless for this application, so treat schema completeness as a correctness requirement, not bookkeeping.

**Handle long sections:** if a section exceeds ~400 words, split into sub-chunks with overlap, keeping the same citation metadata and adding `sub_idx`. Retrieval scores at sub-chunk level, deduplicates to section level for display.

### 5.4 Bangla preprocessing rules

This is a substantive chapter, not a formality. Document every rule you apply and why:

1. **Unicode normalization to NFC.** Bangla conjuncts and vowel signs have multiple valid byte sequences; without this, visually identical strings will not match.
2. **ZWJ / ZWNJ (U+200D / U+200C) handling.** Inconsistently inserted in web text, invisible, and they silently break tokenization. Strip or normalize by an explicit documented rule.
3. **নুক্তা (nukta) normalization.** ড় / ঢ় / য় exist in both precomposed and decomposed forms.
4. **Digit normalization.** Bangla numerals ↔ ASCII (১২৩ ↔ 123). **Critical** for this project — section numbers appear in both forms, in both queries and text.
5. **Punctuation.** Bangla দাঁড়ি (।) vs. period, and legal text uses idiosyncratic sub-clause markers.
6. **Whitespace and scraping artifacts.** Non-breaking spaces, repeated newlines, HTML entity leftovers.
7. **Stemming: document the limitation honestly.** Bangla stemmers are weak and lossy on inflected legal vocabulary. State that you evaluated the available options and report what you chose. An honest negative finding about tooling is a legitimate report contribution.

**Important:** if you use BanglaBERT (`csebuetnlp/banglabert`), its authors ship a required normalization utility that must be applied before tokenization. Apply *their* normalizer for *their* model — do not substitute your own pipeline and expect published performance.

---

## 6. Question–Section Dataset (The Critical Path)

**You have documents but no questions.** This is the single hardest and most time-consuming part of the project, and the part that determines whether every downstream model looks good or bad. Budget accordingly.

### 6.1 Three sources, used together

| Source | Volume | Used for | Method |
|---|---|---|---|
| **Mined real questions** | 200–400 | Training + gold candidates | Collect from legal-aid FAQ pages and public legal help discussions. Real citizen phrasing — the whole point. |
| **Synthetic generation** | 2,000–2,500 | Training only | For each section, prompt an LLM to write 2–3 plausible Bangla citizen questions it answers (the doc2query / InPars approach). Cheap, scalable. |
| **Hand annotation** | 200 | **Gold test set only** | Real questions manually mapped to correct sections by your team. Never trained on. |

Using all three in combination, and reporting how each contributes, is itself a defensible methodological contribution.

### 6.2 The synthetic-data trap — read this carefully

An LLM asked to write a question about a section will tend to **reuse the section's own vocabulary**. If you train and test on such questions, you have accidentally built a lexical-overlap task, your fine-tuned model will post excellent numbers, and you will have measured nothing.

Mitigations, all of which go in your report:
- Prompt explicitly for **colloquial register**: instruct the model to write as a villager or a worker with no legal education would speak, and to avoid the section's technical terms.
- Measure and report **lexical overlap** (e.g. token-level Jaccard) between synthetic questions and their source sections, and compare that distribution against your mined real questions. If they differ sharply, say so.
- **Evaluate only on the hand-annotated gold set of real questions.** This is non-negotiable and is what makes your headline numbers trustworthy.

### 6.3 Gold set construction

- 200 real questions, each annotated with the correct section (allow a small ranked list where more than one section genuinely applies).
- **Double-annotate 50** across two team members and report inter-annotator agreement. This costs one afternoon and materially raises the credibility of every number in your results chapter.
- Tag each gold question along two axes: `register` ∈ {colloquial, formal} and `domain` ∈ {family, land, labour, consumer, constitutional}. These tags produce your most interesting result slices.

### 6.4 Intent labels

Reuse the `domain` tag as the intent classification target — five or six classes. No separate annotation effort required; it comes free with the work above.

---

## 7. System Architecture

```
              Bangla citizen query
                       │
              ┌────────▼────────┐
              │  normalization  │   (§5.4 rules)
              └────────┬────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼─────┐              ┌────────▼────────┐
   │   BM25   │              │ fine-tuned BERT │
   │ (sparse) │              │   bi-encoder    │
   └────┬─────┘              └────────┬────────┘
        │                             │
        └──────────┬──────────────────┘
                   │  hybrid fusion (RRF)
          ┌────────▼────────┐
          │ top-50 candidates│
          └────────┬────────┘
                   │
        ┌──────────▼──────────┐
        │ cross-encoder rerank│
        └──────────┬──────────┘
                   │
        top-3 sections + citations  ──►  display
                   │
                   └──►  [stretch] grounded plain-Bangla explanation
```

Retrieve broadly and cheaply, then rerank precisely. Each stage is independently ablatable, which is exactly what your results chapter needs.

Running alongside the retrieval path: an **intent classifier** on the same query, reported separately (§8.5). It demonstrates the supervised requirement and, optionally, can filter candidates by predicted domain — but keep that filtering as an ablation, not a default, since a wrong intent prediction would otherwise silently destroy retrieval.

---

## 8. The Experimental Ladder

Every rung is a lab topic. The ladder is not a checklist — it is the narrative arc of your results chapter.

### 8.1 Rung 1 — BM25 (BoW / TF-IDF lineage)
Sparse lexical retrieval over normalized sections. Not a toy: BM25 is a genuinely strong baseline that routinely beats poorly-tuned dense models, and saying so in your report signals you know the literature. **Expected finding:** competitive on formal-register queries, poor on colloquial ones. That split *is* your motivating result — present it as such.

### 8.2 Rung 2 — Self-trained Word2Vec
Train Word2Vec (skip-gram) on your legal corpus. Two uses:

- **Retrieval baseline:** mean-pooled word vectors, cosine similarity.
- **Qualitative embedding evidence:** nearest-neighbour tables for আটক, নামজারি, খারিজ, রিট, জামিন in *your legal-trained* model versus a general-Bangla-trained model. The neighbourhoods differ dramatically, and this is the most direct, concrete answer to Sir's "explain how you did embedding" — a self-trained model whose learned semantics you can display and interpret.

*Note:* your corpus (~200k–400k words) is small for Word2Vec. Expect modest retrieval performance and **report that honestly** as a corpus-size finding rather than hiding it. Understanding *why* a method underperforms is worth more marks than the method performing well.

### 8.3 Rung 3 — BiLSTM encoder
BiLSTM over Word2Vec inputs, trained with contrastive loss on your question–section pairs. This is the first rung that actually *learns the mapping* from citizen language to legal language, rather than relying on static vectors. Expect a clear jump over Rung 2 and a clear gap below Rung 4 — that gap is your empirical argument for contextual representations.

### 8.4 Rungs 4 & 5 — Transformer bi-encoder, then reranker

**Rung 4a — zero-shot dense (control).** An off-the-shelf multilingual sentence encoder, no fine-tuning. This baseline is essential: it separates "transformers are good" from "*our fine-tuning* is good." Without it you cannot claim the contribution.

**Rung 4b — fine-tuned bi-encoder (the headline result).**
- Start from a multilingual sentence-embedding checkpoint with real Bangla coverage (candidates: `intfloat/multilingual-e5-base`, `sentence-transformers/LaBSE`, `BAAI/bge-m3`; benchmark 2–3 zero-shot in Week 5 and pick on evidence, not vibes).
- Train with `MultipleNegativesRankingLoss` in `sentence-transformers` — in-batch negatives, well-suited to your data shape.
- **Hard negatives:** mine top-ranked-but-wrong sections using BM25 and the zero-shot dense model. Sections from the same Act that look superficially similar are the negatives that teach fine distinctions. This single technique is typically worth more than any hyperparameter tuning you will do.
- Practical settings for a Colab-class GPU: base-size model, max_len 256–384, batch 16–32 with gradient accumulation, 2–4 epochs, lr ~2e-5.

**Rung 5 — cross-encoder reranker.** Retrieve top-50 via hybrid fusion, then score each (query, section) pair jointly with a cross-encoder and keep the top 3. Slower per pair but far more accurate, and only applied to 50 candidates. **Expected finding:** the largest single accuracy jump in the whole ladder.

### 8.5 Parallel track — intent classification
Same task, four representations: TF-IDF → Word2Vec-averaged → BiLSTM → fine-tuned BERT. Report accuracy and macro-F1. Add **Multinomial Naive Bayes on TF-IDF** as a generative counterpart to your discriminative models — this satisfies the generative-vs-discriminative comparison at essentially zero cost, since the data is already labelled.

### 8.6 Unsupervised track
K-Means (or HDBSCAN) over section embeddings, with k chosen by silhouette score. Then measure whether the discovered clusters align with actual Act/domain structure using **Adjusted Rand Index** and **Normalized Mutual Information** against your `domain` labels. A quantified answer to "did the model discover the structure of the law?" is much stronger than a picture of coloured dots.

---

## 9. Evaluation Plan

### 9.1 Metrics

| Task | Metrics |
|---|---|
| Retrieval | Recall@1, Recall@5, Recall@10, MRR@10, nDCG@10 |
| Classification | Accuracy, macro-F1, per-class F1, confusion matrix |
| Clustering | Silhouette, ARI, NMI vs. domain labels |

### 9.2 Protocol
- All retrieval numbers reported on the **200-question hand-annotated gold set only**.
- Training pairs split 90/10 train/dev; dev used for early stopping and checkpoint selection. Gold set touched **once**, at the end.
- Fix random seeds; report the seed. If time allows, run the fine-tuning 3× and report mean ± std — a variance figure distinguishes a real improvement from a lucky run.

### 9.3 The result slices that carry the report

1. **Main ladder table.** All five configurations × all retrieval metrics. Monotonic improvement is the story.
2. **Colloquial vs. formal breakdown.** The same table split by query register. Prediction: BM25's gap between the two is large; the fine-tuned bi-encoder's gap is small. **This table is the proof of your thesis** — it shows the lexical gap exists and that your method specifically closes it.
3. **Per-domain breakdown.** Which areas of law are hardest, and hypothesise why (section length? vocabulary density? overlapping provisions?).
4. **Error analysis.** Manually inspect 30–40 failures of the best model and categorise them. Genuine ambiguity, annotation error, sections too long, near-duplicate provisions across Acts. An honest error taxonomy is one of the most-rewarded sections in a project report and one of the least-often done.

### 9.4 The two figures

- **Figure A:** bar chart of Recall@5 climbing across the five rungs, split colloquial vs. formal. Your whole project in one image.
- **Figure B:** t-SNE (or UMAP) of gold questions and their correct sections, before vs. after fine-tuning. Before: questions and their answers scattered apart. After: collapsed together. This is the single most persuasive "here is what our embeddings learned" evidence you can put in front of an examiner — and it is a direct, visual answer to the requirement Sir emphasised most.

---

## 10. Timeline — 8 Weeks

Roles assume 3 members. **With 2 members, merge B and C and drop Rung 3 (BiLSTM retrieval) to classification-only** — keep the BiLSTM in the intent-classification track so the syllabus coverage survives.

- **A — Data:** scraping, sectioning, preprocessing, annotation coordination
- **B — Classical & classification:** BM25, Word2Vec, BiLSTM, NB, clustering
- **C — Transformers & system:** bi-encoder, reranker, demo, figures

| Week | Milestone | A | B | C |
|---|---|---|---|---|
| **1** | **Feasibility gate + toy pipeline** | Verify Bangla text availability per Act (§11 R1); scrape 3 Acts | Set up repo, eval harness, metric functions | Toy end-to-end: 50 sections, 30 questions, BM25 only |
| **2** | **Corpus v1 frozen** | Full scrape 4 domains + Constitution; sectioning; normalization pipeline | Benchmark 2–3 zero-shot dense models on 30 toy questions | Synthetic question generation pipeline built + overlap check |
| **3** | **Training pairs done** | Mine real questions; run synthetic generation at scale | **BM25 baseline — first real numbers** | Hard-negative mining infrastructure |
| **4** | **Gold set done (200 q)** | Lead annotation; double-annotate 50; agreement score | Word2Vec trained; nearest-neighbour tables; W2V retrieval baseline | Fine-tuning script working on dev set |
| **5** | **Neural baselines** | Intent labels finalised; corpus QA pass | BiLSTM encoder + intent classifier; TF-IDF & NB classifiers | Zero-shot dense baseline **locked** (the control) |
| **6** | **Headline result** | Begin methodology + data chapters | Clustering + ARI/NMI | **Fine-tuned bi-encoder — main experiment** |
| **7** | **Full system** | Error analysis (30–40 cases) | Classification results table finalised | Cross-encoder reranker; hybrid fusion; Figures A & B |
| **8** | **Ship** | Report assembly | Results tables finalised | Gradio demo; slides; buffer |

**Hard gates — do not pass without these:**
- End of Week 1: toy pipeline runs end to end. If not, you have an environment problem, not a research problem — fix it before scaling.
- End of Week 4: gold set complete. **If this slips, cut corpus scope, not evaluation quality.**
- End of Week 6: fine-tuned bi-encoder beats the zero-shot control. If it does not, you have a debuggable problem with two weeks left rather than a crisis with two days left.

**There is no generation layer.** It was cut permanently rather than deferred; see §3.

---

## 11. Risk Register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| **1** | **Bangla text coverage thinner than expected.** Many older Acts are published in English only; Bangla versions exist for a subset. | **High** | **Week 1 feasibility gate.** Verify per-Act availability *before* committing. Three fallbacks, in order: (a) reselect domains toward Acts with confirmed Bangla text; (b) **pivot to cross-lingual retrieval** — Bangla queries against English legal text, which multilingual encoders handle and which is *more* interesting, not less; (c) mixed corpus with language as a metadata field and a reported per-language result split. |
| 2 | Synthetic questions lexically leak from source sections | High | Colloquial-register prompting; measure and report overlap; evaluate on real gold questions only (§6.2) |
| 3 | Gold annotation slips past Week 4 | Medium | Cut corpus to 3 domains; 150 questions is an acceptable floor. Evaluation quality is never the thing you sacrifice. |
| 4 | Compute limits (Colab timeouts, VRAM) | Medium | Base-size models only; max_len 256; gradient accumulation; checkpoint to Drive every epoch; precompute and cache corpus embeddings |
| 5 | Section splitter fails on irregular Acts | Medium | Keep raw HTML archive; hand-verify a 50-section random sample in Week 2; per-Act splitter overrides |
| 6 | Word2Vec underperforms on a small corpus | High | **Expected, not a failure.** Report as a corpus-size finding with analysis. |
| 7 | Scope creep via a generation layer | Low | Removed from scope entirely (§3). Retrieval and citation only. |

---

## 12. Ethics, Safety, and Limitations

**This system is an information-retrieval tool, not legal advice.** This must be stated in three places: the demo interface, the report abstract, and the report's limitations section.

Design commitments that back that claim up:
- **Every output carries its citation.** No uncited output, ever.
- **The retrieved legal text is the output**, always shown as printed (`text_raw`). Nothing paraphrases or summarises a provision for the user; there is no generated text anywhere in the system.
- **Crawl date is recorded and displayed.** Laws are amended; a stale snapshot presented as current is the actual harm risk here.
- **No personal data.** Questions mined from public discussion are stripped of names, phone numbers, and identifying details, and are not republished verbatim with attribution.
- **Scraping conduct:** respect terms of use and robots.txt, rate-limit politely, identify the crawler.

**Stated limitations:** snapshot-in-time coverage; a bounded set of domains, listed explicitly, with everything outside them uncovered; standalone amendment Acts and repealed provisions excluded, so a provision amended after `crawl_date` is served in its pre-amendment consolidated form; no case law or precedent; no multi-turn reasoning; retrieval quality degrades on queries outside the covered domains — and the system should say "I don't have a confident match" rather than return a bad one. Add a score threshold below which the system explicitly declines.

Examiners consistently reward teams that draw these lines clearly and unprompted.

---

## 13. Deliverables

1. **Corpus** — `corpus_v1.jsonl`, section-level, with full citation metadata and crawl date
2. **Datasets** — training pairs, gold test set with register/domain tags, intent labels
3. **Codebase** — scraper, preprocessing pipeline, training scripts, unified evaluation harness
4. **Results** — ladder table, register-split table, per-domain table, classification table, clustering metrics, error taxonomy
5. **Figures** — Figure A (ladder bars), Figure B (t-SNE before/after), cluster map, confusion matrix
6. **Demo** — Gradio/Streamlit: Bangla text box in → top-3 sections with citations out, with disclaimer visible
7. **Report** — data collection · preprocessing · embedding methodology · models · results · error analysis · limitations
8. **Slides** — Figure B is your closing slide

---

## 14. Requirement Compliance

| Sir's point | How this project satisfies it |
|---|---|
| **1. Explain embeddings concretely; own-trained gives a better explanation; balanced approach welcome** | Self-trained Word2Vec with interpretable nearest-neighbour analysis **and** a fine-tuned transformer, both compared against pretrained baselines. Figure B makes the learned geometry visible. Exceeds the requirement. |
| **2. Reasonably sized corpus; toy for prototyping** | Toy set (50 sections) in Week 1, working corpus of 800–1,200 sections, expandable to ~2,000. Self-built and hand-verified. |
| **3. Primarily supervised; unsupervised adds value; classification not mandatory** | Supervised core on two fronts (contrastive retrieval fine-tuning + intent classification), with clustering quantified by ARI/NMI as the unsupervised layer. |
| **4. No fancy GUI; clear input→output** | Gradio text box: Bangla question in, cited sections out. |
| **5. Generative vs. discriminative — one is fine, both is a bonus** | Naive Bayes (generative) vs. LSTM/BERT (discriminative) on identical features, at near-zero marginal cost. |
| **6. Document data collection if you build a corpus** | Full methodology chapter: sources, scraping method, crawl date, section splitting, chunk schema, and seven documented Bangla normalization rules. |

---

## 15. Environment

Python 3.10+ · `requests` + `beautifulsoup4` (scraping) · `rank_bm25` (sparse) · `gensim` (Word2Vec) · `pytorch` (BiLSTM) · `transformers` + `sentence-transformers` (bi-encoder, cross-encoder) · `faiss-cpu` or brute-force cosine (a corpus this size does not need an ANN index) · `scikit-learn` (TF-IDF, NB, clustering, metrics) · `matplotlib` (figures) · `gradio` (demo). Colab / Kaggle T4 is sufficient throughout.

---

## 16. The One Thing to Get Right

If you take a single operational lesson from this plan: **build the whole pipeline end-to-end in Week 1 at toy scale — 50 sections, 30 questions, BM25 only — before you scale anything.**

Teams fail this kind of project by spending five weeks perfecting a scraper and then discovering in Week 6 that their evaluation harness has a bug. A working ugly pipeline in Week 1 converts every later problem from a crisis into an incremental fix.

And remember where the real bottleneck is: not the corpus, not the model, but the **question–section pairs**. If your gold set is weak, every model downstream will look bad no matter how good your code is.
