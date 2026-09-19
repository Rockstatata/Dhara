# Quarantined — gold-answer leakage

All three files here carry a `question_expanded` field written by
`scripts/73_generate_bilingual_query_expansions.py`, which hardcodes a
per-`qid` English gloss that already names the gold Act and section number
for that question (e.g. `prot_0066_02` -> "Penal Code 1860 Section 420
cheating fraud..."). Scoring retrieval against `question_expanded` measures
whether the model can find a provision when told its own answer, not
retrieval quality. See DECISIONS.md 2026-09-19 ("Bilingual concept expansion
claim retracted") for the full account, including the R@10 38.9%->72.2%
number in the same-day earlier entry that this leakage produced and that
does not hold.

Do not use these files for evaluation. Do not regenerate them by running
`scripts/73_generate_bilingual_query_expansions.py` against the live
`dev_retrieval_v4.jsonl` / `test_retrieval_v4.jsonl` — it will silently
overwrite those clean files in place with the same leak.

- `eval_retrieval_v4.jsonl` — pooled dev+test from `scripts/72_pool_eval_v4.py`, contaminated after `73_...` ran against it.
- `dev_retrieval_v4_bilingual.jsonl`, `test_retrieval_v4_bilingual.jsonl` — the direct output of `73_...`.
