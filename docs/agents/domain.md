# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the glossary of domain terms.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This is a single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-section-is-the-retrieval-unit.md
│   └── 0002-two-level-bangla-normalization.md
└── src/
```

There is no `CONTEXT-MAP.md` and no per-context `src/<context>/docs/adr/`. If the project ever splits into multiple bounded contexts, add a root `CONTEXT-MAP.md` and update this file.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Relationship to `DECISIONS.md`

This repo also keeps `DECISIONS.md` at the root (date · decision · reason), which is the raw material for the report's methodology chapter. It is a running log, not a substitute for ADRs: a decision that is hard to reverse, surprising without context, and the result of a real trade-off gets a full ADR under `docs/adr/` as well, and the log entry points at it.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0002 (two-level Bangla normalization) — but worth reopening because…_
