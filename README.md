# Dhara

A Bangla legal retrieval system for citizen services.

Dhara takes a plain-Bangla citizen question and returns the exact law section, along with its official citation, that answers it. The project focuses on the gap between colloquial public language and formal legal Bangla, and it evaluates whether domain-specific fine-tuning closes that gap in a measurable way.

## Project vision

Bangladeshi citizens often ask legal questions in everyday speech: they say things in plain Bangla that do not match the wording used in the law. A keyword search or a generic lexical system is not enough. Dhara is designed to bridge that gap by matching a citizen query to the relevant section of a law using retrieval methods that improve from BM25 to dense semantic retrieval and reranking.

The project is built around a concrete research claim:

- the lexical gap between citizen phrasing and legal wording is real,
- it can be measured,
- and domain fine-tuning improves retrieval on that gap.

The system is not legal advice. It is an information-retrieval tool that identifies the most relevant legal section and its citation.

## Problem statement

A citizen might ask something like:

> পুলিশ আমাকে ধরে নিয়ে গেছে, কিছু বলছে না — আমার কী অধিকার?

The matching legal language may be structured around formal phrasing such as:

> গ্রেপ্তার ও আটক সম্পর্কে রক্ষাকবচ

This is a semantic mismatch, not a spelling problem. The task is therefore not trivial keyword matching. It requires retrieval that can align informal Bangla with formal legal Bangla while preserving citation fidelity.

## Scope

The system targets four primary legal domains:

- family and inheritance
- land and property
- labour and employment
- consumer and information rights

A fifth cross-cutting source is also included:

- the Constitution of Bangladesh

The intended corpus size is approximately 800 to 1,200 section-level chunks, with room to expand. The design deliberately stays smaller and more carefully curated than a complete legal corpus of every Act in the country.

## Non-goals

The system does not aim to:

- replace legal professionals,
- provide advice in place of a lawyer,
- cover every Act in the national legal ecosystem,
- answer case-law or judicial precedent tasks,
- support multi-turn legal dialogue as a first version.

## Core architecture

The system follows a retrieval pipeline:

1. normalize the query
2. retrieve candidates with sparse and dense methods
3. fuse and rerank candidates
4. return the top relevant sections with citations
5. optionally support intent classification as a separate task

The intended flow is:

```text
Bangla citizen query
        |
        v
normalization
        |
        +--> BM25 (sparse)
        +--> fine-tuned bi-encoder (dense)
        |
        v
RRF fusion
        |
        v
top-50 candidates
        |
        v
cross-encoder reranker
        |
        v
top-3 sections with citations
```

The architecture includes an intent classifier as a parallel research track, but intent-based filtering is intentionally kept as an ablation, not the default path. Wrong intent predictions can silently damage retrieval quality.

## Experimental ladder

The project uses a staged ladder of retrieval models. Each rung is itself a research result, not merely a discarded prototype.

### Rung 1: BM25

This acts as the lexical baseline. It is expected to work better on formal queries than on colloquial ones. The split in performance across query register is a central part of the project story.

### Rung 2: Self-trained Word2Vec

A Word2Vec skip-gram model is trained on the legal corpus. The model is used both as a retrieval baseline and as a qualitative embedding test. Neighbourhoods around legal terms are inspected to understand whether the legal language learned meaningful semantic structure.

### Rung 3: BiLSTM dual encoder

This provides the first learned mapping from informal citizen phrasing to legal language.

### Rung 4a: Zero-shot multilingual dense baseline

This is the control model. It establishes whether generic multilingual sentence encoders already solve the problem without domain adaptation.

### Rung 4b: Fine-tuned bi-encoder

This is the primary headline result. Starting from a multilingual embedding model, the system is fine-tuned using question-section pairs and hard negatives. This is the heart of the domain adaptation claim.

### Rung 5: Cross-encoder reranker

After hybrid retrieval over the top candidates, a cross-encoder refines the ordering and produces the final ranking.

## Data pipeline

The repository is designed around an explicit data and modelling flow.

### 1. Act survey and feasibility gate

The project starts by checking which Acts have Bangla legal text available, how many sections they contain, and whether the corpus is large enough to proceed. This gives an early, honest signal about whether the project should stay on the planned path or pivot.

### 2. Scraping

Raw HTML from the official legal portal is archived before any processing. This is a strict requirement because parsing mistakes are expected; re-parsing from local archives is faster and more reliable than re-crawling.

### 3. Section splitting

The system retrieves at the section level rather than the Act level. Sections with their numbering, chapter metadata, title, and source citation are kept as the fundamental unit. Repealed or omitted provisions are dropped rather than included, because retrieving them would be harmful.

### 4. Normalization

The project uses two normalization levels:

- light normalization: used for transformer input
- aggressive normalization: used for BM25, Word2Vec, and classical retrieval

These are intentionally different. Applying aggressive normalization to transformer input would distort the task and produce misleading numbers.

Bangla-specific issues include:

- Unicode normalization
- ZWJ and ZWNJ handling
- digit normalization between Bangla and ASCII numerals
- punctuation handling
- whitespace and scraping artifact cleanup

Digit normalization is especially important because section numbers appear in both Bangla and ASCII forms.

### 5. Question generation and annotation

The dataset includes:

- mined real questions,
- synthetic questions generated from legal sections,
- hand-annotated gold evaluation questions.

The key design principle is that the gold set is kept isolated and is not used for debugging or iterative tuning.

The synthetic questions are intentionally inspected for lexical overlap with their source sections, because that is a common cause of misleadingly strong performance.

## Dataset and schema

The project operates with frozen artifact versions such as:

- corpus_v1.jsonl
- train_pairs_v1.jsonl
- dev_pairs_v1.jsonl
- gold_test_v1.jsonl
- hard_negatives_v1.jsonl

The chunk schema contains the citation metadata and is treated as a correctness requirement, not as optional bookkeeping. The retrieval unit is not the Act; it is the legal section.

The required metadata includes fields such as:

- chunk_id
- act_name_bn
- act_year
- chapter
- section_no
- section_title_bn
- text_bn
- domain
- source_url
- crawl_date
- char_len

The metadata is what makes the retrieved result citable and usable for end users.

## Repository structure

The repository is expected to follow this layout:

```text
configs/
  domains.yaml
  models.yaml
  paths.yaml
  split_overrides.yaml

data/
  raw/
  processed/
  external/

src/
  dhara/
    __init__.py
    normalize.py
    schema.py
    scrape.py
    section_split.py
    synth.py
    negatives.py
    metrics.py
    evaluate.py
    retrievers/
      base.py
      bm25.py
      word2vec.py
      bilstm.py
      biencoder.py
      hybrid.py
    rerank.py
    classify.py
    cluster.py
    service.py
  app/
    app.py

scripts/
  01_survey_acts.py
  02_scrape.py
  03_build_corpus.py
  04_generate_questions.py
  05_mine_negatives.py
  06_train_biencoder.py
  07_train_crossencoder.py
  08_build_index.py
  09_run_eval.py

results/
  tables/
  figures/
  runs/

notebooks/
report/
```

The repo is designed to keep business logic in the source package and keep scripts as thin command-line wrappers.

## Setup

The project is designed for Python 3.10+.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

For BanglaBERT-related workflows, install the required normalizer package separately:

```bash
pip install git+https://github.com/csebuetnlp/normalizer
```

The project is intended to run primarily on a Colab or Kaggle T4 environment, which is sufficient for the planned model training and evaluations.

## Required commands

The following commands represent the contract the project is designed around:

```bash
python scripts/01_survey_acts.py
python scripts/02_scrape.py --acts 3 --out data/raw/
python scripts/03_build_corpus.py --in data/raw/ --out data/processed/corpus_v1.jsonl
python scripts/03_build_corpus.py --qa
python scripts/04_generate_questions.py
python scripts/05_mine_negatives.py
python scripts/06_train_biencoder.py
python scripts/07_train_crossencoder.py
python scripts/08_build_index.py
python scripts/09_run_eval.py --corpus corpus_v1.jsonl --gold gold_test_v1.jsonl --retriever bm25
python -m src.app.app
```

The repo structure and command sequence are meant to support a clean end-to-end workflow from scraping to retrieval evaluation to demo.

## Evaluation plan

Evaluation is a central part of the project and is treated as carefully as the model itself.

### Retrieval metrics

- Recall@1
- Recall@5
- Recall@10
- MRR@10
- nDCG@10

### Classification metrics

- accuracy
- macro-F1
- per-class F1
- confusion matrix

### Clustering metrics

- silhouette score
- Adjusted Rand Index
- Normalized Mutual Information

The main evaluation focus is the comparison across the retrieval ladder and the ability to show how the fine-tuned dense model improves over lexical and zero-shot baselines, especially on colloquial queries.

## Golden rules

The implementation guide makes a few rules non-negotiable.

### Gold set isolation

The gold test set must be touched exactly once, at the end of the project. It is not for debugging or informal checks.

### Data splitting by chunk

Synthetic questions generated from a chunk must remain in the same split as that chunk. This prevents leakage through near duplicates.

### Frozen artifacts

A corpus snapshot should never be silently modified. If it must change, the artifact is versioned as the next file, such as corpus_v2.jsonl, and all results depending on the earlier version are re-run or explicitly labelled.

### No hand-typed numbers

Every reported number should come from a script written under results/runs and emitted in a reproducible file. Notebook-only numbers are not acceptable as final evidence.

### Citation and abstention

Every answer should carry a citation. The system is expected to abstain when the confidence is too low instead of giving a confident but incorrect answer.

### E5 prefix handling

If the final model family uses E5-style embeddings, the required query and passage prefixes must be applied consistently. Omitting them is a common and damaging baseline error.

### Hard negatives

Hard negatives should come from ranks 5 to 30, not from the top few. The top few are often relevant but unlabelled, and training against them teaches the model the wrong lesson.

## Importance of ethics and safety

This is an information-retrieval tool, not legal advice.

The project explicitly follows ethical constraints:

- the UI and documentation should state that the tool is not legal advice,
- repealed or omitted legislation is excluded from the corpus,
- the dropped count is recorded,
- crawl date is displayed,
- mined real questions are stripped of names, phone numbers, NID numbers, and addresses before they enter the dataset.

The purpose is to help citizens find the relevant legal section, not to act as a substitute for legal counsel.

## Data hygiene and repository practices

The repository guidelines require careful treatment of data and generated artifacts.

- raw scraped data is never committed,
- pretrained weights and model checkpoints are not committed,
- processed JSONL artifacts are tracked because they support reproducibility,
- results and metrics are generated by scripts and written to results/,
- decisions that are not obvious are recorded in DECISIONS.md.

## Development workflow

The project is planned as a weekly workflow with role-based ownership.

- Data engineering owns scraping, normalization, and corpus preparation.
- Dataset and annotation work owns question generation and gold set management.
- Classical modelling and evaluation own BM25, Word2Vec, BiLSTM, Naive Bayes, clustering, and evaluation harnesses.
- Transformer and system work owns the dense retriever, reranker, service layer, and demo.

Working in parallel is important, but the frozen interfaces and shared schema prevent drift and misalignment.

## Documentation and decisions

The repository is guided by two authoritative documents:

- docs/Dhara_Proposal.md
- docs/Dhara_Implementation_Guide.md

These documents define the problem framing, the implementation plan, dataset design, and the evaluation philosophy. They should be treated as the source of truth for the project.

A decision log should also be maintained in DECISIONS.md. Important choices should be recorded with date, decision, and reason so that the methodology remains reconstructable.

## Roadmap

### Phase 0: feasibility and toy pipeline

- verify the legal text exists,
- run a small-scale crawl and toy retrieval pipeline,
- confirm the whole stack works end to end.

### Phase 1: corpus curation

- scrape and archive HTML,
- split into sections,
- normalize text,
- build frozen corpus artifacts.

### Phase 2: question generation and annotation

- create synthetic and mined training data,
- construct evaluation gold data,
- build hard negatives and intent labels.

### Phase 3: classical and neural retrieval

- run BM25 and Word2Vec baselines,
- train BiLSTM and dense bi-encoder models,
- add reranking and hybrid fusion.

### Phase 4: evaluation and reporting

- generate reproducible metrics,
- compare the ladder results,
- document the findings honestly.

### Phase 5: demo and deployment

- expose the retrieval workflow via a user-facing interface,
- return citations and abstentions,
- keep the system grounded in the legal text.

## Summary

Dhara is a research-driven legal retrieval system built for Bangla citizen queries. It measures the lexical mismatch between informal public language and formal legal language, and it evaluates whether retrieval models can close that gap with careful domain adaptation. The project values reproducibility, evaluation integrity, and transparent reporting over headline-only performance.

This repository is meant to support a serious, methodologically grounded implementation that can be compared rigorously across baselines and model families while remaining faithful to the legal and ethical constraints of the domain.
