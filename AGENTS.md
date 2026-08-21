# Repository Guidelines

## Project Structure & Module Organization

The repository currently contains the proposal and implementation specification under `docs/`. Read both before introducing code or data contracts.

The planned layout is `src/dhara/` for reusable Python modules, `src/app/` for Gradio, and `scripts/` for thin numbered CLIs. Put tests in `tests/`, exploration in `notebooks/`, configuration in `configs/`, and metrics in `results/`. Raw downloads belong in `data/raw/`; versioned JSONL artifacts belong in `data/processed/`.

## Build, Test, and Development Commands

There is no executable environment or test suite yet. When bootstrapping the implementation, document commands in `README.md` and keep these expected workflows working:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/02_scrape.py --acts 3 --out data/raw/
python scripts/09_run_eval.py --corpus data/processed/corpus_toy.jsonl --gold data/processed/gold_toy.jsonl --retriever bm25
python src/app/app.py
pytest
```

Run scripts from the repository root so paths resolve consistently.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, and type hints on public interfaces. Name modules, functions, and variables with `snake_case`; classes with `PascalCase`; constants with `UPPER_SNAKE_CASE`. Number pipeline wrappers by execution order, such as `03_build_corpus.py`. Keep business logic in `src/dhara/`, not in scripts or notebooks. Preserve Bangla display text as `text_raw`; use normalized text only for indexing and matching.

## Testing & Reproducibility

Use `pytest`; name files `test_<module>.py` and tests `test_<behavior>`. Add unit tests for normalization, section splitting, schemas, and metrics, plus one toy end-to-end pipeline test. Every reported number must be produced by a script and written under `results/`. Never tune against `gold_test_v*.jsonl`; use development data and assert that gold questions and near-duplicates are absent from training splits. Frozen artifacts are immutable—create `*_v2` rather than overwriting `*_v1`.

## Commits & Pull Requests

The repository has no commit history from which to infer a convention. Use imperative, scoped subjects, such as `feat(scraper): parse section headings`. Use focused branches such as `feat/A-scraper`. Pull requests should explain the change, list validation commands, note data or model effects, link the relevant issue or decision, and include screenshots for UI changes. Require one teammate review before merging.

## Security & Data Hygiene

Do not commit credentials, raw scraped data, pretrained weights, checkpoints, `*.bin`, or `*.safetensors`. Record non-obvious architectural and dataset decisions in `DECISIONS.md` with the date, decision, rationale, and accepted trade-off.
