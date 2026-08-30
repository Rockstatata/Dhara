"""Consolidate the four annotator sheets into one gold set, and score agreement.

    python scripts/08_merge_gold.py
    python scripts/08_merge_gold.py --sheets data/annotation --strict

Reads every `annotate_*.csv` under `data/annotation/`, validates each row against
the corpus, resolves answers, reports agreement and per-annotator contribution,
and writes `data/processed/gold_test_v1.jsonl`.

Four things this script exists to get right:

**Answers are resolved, not trusted.** An annotator may write a candidate rank
(`3`), a chunk id (`labour_2006_s103`), several of either separated by commas, or
`NONE`. A rank is resolved back through the `<chunk_id>` markers embedded in that
row's own candidate cell, so a rank always means the same provision it meant on
the annotator's screen. Anything that does not resolve to a chunk actually
present in the corpus is an error, not a warning — a gold answer pointing at a
provision that does not exist is worse than a missing one.

**Agreement is two different statistics, and conflating them would be wrong.**
For `domain` and `register` — a handful of categories — Cohen's κ is the right
measure and is what reviewers expect. For the *answer* it is not: with 6,359
possible chunks, chance agreement is essentially zero, so κ collapses onto raw
agreement and reporting it as κ dresses a simple percentage in borrowed
authority. So the answer is reported as raw agreement at two levels — exact chunk
and parent provision, since two annotators picking different sub-chunks of one
long section have not actually disagreed about the law.

**The BM25 miss rate is computed here.** Every row records whether its answer
came from the candidate list or from the annotator's own search, and the unaided
block carries no candidates at all. Together those give an honest estimate of how
often a lexical retriever simply cannot reach the right provision — which is the
project's own thesis, and it is measured rather than asserted.

**Contribution stays attributable.** Counts are reported per annotator, and the
`annotator` field is carried into every gold record, so the report can say who
labelled what instead of crediting one laptop.

**An annotator should never have to memorise or hand-copy a chunk_id.** When the
right answer is not one of the ten candidates, the honest path is to search for
it and cite it — but a `chunk_id` like `family_476_s3` is an internal key, not
something a person doing legal research thinks in. So `resolve()` accepts a
plain citation instead: "Muslim Marriages and Divorces (Registration) Act, 1974,
section 3" resolves the same as pasting `<family_476_s3>`. See `resolve_citation`
below for how the Act name is matched.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import light  # noqa: E402

SHEETS = pathlib.Path("data/annotation")
CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
GOLD = pathlib.Path("data/processed/gold_test_v1.jsonl")
STATS = pathlib.Path("results/tables/annotation_stats.csv")

CHUNK_MARKER = re.compile(r"<([^>\s]+)>")
RANK_LINE = re.compile(r"^\s*(\d+)\.\s*<([^>\s]+)>")

# "section 3", "sec. 3", "s. 3", "ধারা ৩", "অনুচ্ছেদ ৩" — every spelling an
# annotator or a legal-research source actually uses for a provision number.
SECTION_MARKER = re.compile(
    r"(?:section|sec\.?|অনুচ্ছেদ|ধারা)\s*(?:no\.?)?\s*[:\-]?\s*([০-৯0-9]+[ক-হA-Za-z]?)",
    re.I,
)
# Words too common to help identify an Act — matching on these alone produces a
# false hit ("the act of 1974" would otherwise match every Act from that year).
_STOP = frozenset(
    "the a an of and or act ordinance no act, acts law laws".split()
    + "আইন অধ্যাদেশ এর ও".split()
)


def _title_words(text: str) -> frozenset[str]:
    words = re.split(r"[\s,()\-]+", light(text).lower())
    return frozenset(w for w in words if w and w not in _STOP)

# These vocabularies are the ones docs/annotation_guideline.md hands the
# annotators, and that document is the contract — the code follows it rather than
# the other way round. Changing a value here without changing it there produces
# exactly the silent config/code divergence that already cost this project once.
#
# `register` is deliberately two-valued, not three. The guideline gives an
# explicit tie-break for the mixed case ("one borrowed legal word does not make
# someone a lawyer — tag it colloquial"), and a two-way split with a stated rule
# yields better agreement than a three-way split with a fuzzy middle category.
VALID_REGISTER = {"colloquial", "formal"}
VALID_CONFIDENCE = {"3", "2", "1", "?"}
VALID_DOMAINS = {
    "family", "land", "labour", "consumer", "cybercrime", "constitutional",
    "criminal_procedure", "women_children", "tenancy", "civil_registration",
    "road_transport", "money_recovery", "local_government", "other",
}
# `candidate_list` is accepted as a synonym of `candidate` so sheets produced
# against either wording still merge.
SOURCE_ALIASES = {"candidate_list": "candidate"}
VALID_SOURCE = {"candidate", "own_search", "none", ""}

# Two different empty answers, and collapsing them would destroy a real
# distinction. `none` means *the annotator could not find it* — a limit of the
# candidate list or of the search. `unanswerable` means *no law answers this* — a
# factual question, a request for a lawyer, a topic outside our domains. Only the
# second calibrates the abstention threshold; the first is a retrieval miss. A
# merged count would silently mix a property of our corpus with a property of our
# annotators.
NONE_TOKENS = {"none", "n/a", "na", "-", "নেই", "কিছু নেই"}
UNANSWERABLE_TOKENS = {"unanswerable", "no law", "অপ্রযোজ্য"}

MAX_ANSWERS = 3


class RowError(Exception):
    pass


def rank_map(candidates: str) -> dict[str, str]:
    """`{"1": chunk_id, "2": chunk_id, ...}` for one row's own candidate cell."""
    out = {}
    for line in candidates.splitlines():
        m = RANK_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class ActIndex:
    """Look up a chunk_id from what a person actually writes down: an Act name
    (however abbreviated) and a provision number.

    Built once per run and reused across every row — rebuilding it per row would
    mean re-scanning the whole corpus 794 times for a four-sheet run.
    """

    def __init__(self, chunks: list[schema.Chunk]):
        self.act_words: dict[str, frozenset[str]] = {}
        self.by_act_section: dict[tuple[str, str], list[schema.Chunk]] = {}
        titles: dict[str, str] = {}
        for c in chunks:
            titles.setdefault(c.act_id, c.act_title_bn or c.act_title_en)
            key = (c.act_id, c.provision_no_ascii)
            self.by_act_section.setdefault(key, []).append(c)
        for act_id, title in titles.items():
            self.act_words[act_id] = _title_words(title)
        for chunks_ in self.by_act_section.values():
            chunks_.sort(key=lambda c: c.sub_idx)

    def match_act(self, text: str) -> str | None:
        """Best-matching act_id for a fragment of Act name, or None.

        Scored as the fraction of the *typed* words found in the Act's title —
        deliberately not the reverse, since a real citation is always shorter
        than the official title ("Shariat Act" for "The Muslim Personal Law
        (Shariat) Application Act, 1937").

        Bangladesh re-titles the same generic name every year — 26 groups of
        acts in this corpus share an identical name once the year is dropped:
        14 separate "Finance Act"s, 5 Chittagong Hill-Tracts regulations, and so
        on. A citation without a year is genuinely ambiguous among them, and
        picking whichever happened to be scanned first would be a silently
        wrong answer with no warning — worse than refusing to guess. So a tie
        at the top score returns None exactly like no match at all, and the
        caller's error message is what tells the annotator to add the year.
        """
        words = _title_words(text)
        if not words:
            return None
        scored = []
        for act_id, title_words in self.act_words.items():
            if not title_words:
                continue
            score = len(words & title_words) / len(words)
            if score >= 0.8:
                scored.append((score, act_id))
        if not scored:
            return None
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None            # ambiguous — do not guess which one
        return scored[0][1]

    def resolve_one(self, line: str) -> str | None:
        """One line of free-text citation to a single chunk_id, or None."""
        m = SECTION_MARKER.search(line)
        if not m:
            return None
        section = schema.to_ascii_digits(m.group(1))
        act_text = line[: m.start()] + line[m.end():]
        act_id = self.match_act(act_text)
        if act_id is None:
            return None
        hits = self.by_act_section.get((act_id, section))
        return hits[0].chunk_id if hits else None


def _try_legacy_tokens(answer: str, ranks: dict, by_id: dict) -> list[str] | None:
    """Old contract: comma/semicolon/whitespace-separated ranks or bare chunk
    ids. Returns None (not a raise) on the first unrecognised token, so the
    caller can fall through to citation-text resolution instead of failing —
    this path is tried first only because it is cheap and unambiguous when it
    works."""
    resolved = []
    for token in re.split(r"[,;\s]+", answer):
        if not token:
            continue
        chunk_id = ranks.get(token, token)
        if chunk_id not in by_id:
            return None
        resolved.append(chunk_id)
    return resolved or None


def resolve(
    answer: str, candidates: str, by_id: dict, act_index: "ActIndex | None" = None
) -> tuple[list[str], str]:
    """Annotator's `answer` cell to `(chunk_ids, verdict)`.

    `verdict` is one of `answered`, `not_found` (the annotator wrote `none`) or
    `unanswerable` (no law answers this). The two empty verdicts are kept apart
    on purpose — see the note on NONE_TOKENS above.

    Three ways to write an answer, tried in this order:

    1. **`<chunk_id>` markers.** Either pasted alone or inside a full citation
       line as `09_autolabel_pass1.py` writes it, e.g. `<other_850_s34> [সালিস
       আইন, ২০০১, ধারা ৩৪] পক্ষগণ ভিন্নভাবে সম্মত না হইলে...`. Everything outside
       the markers is read as decoration for the human, not data for the parser.
    2. **A bare rank or chunk_id** (`3`, `3, 7`, `family_138_s52`) — the original
       contract, kept for anyone who already types this way.
    3. **Plain-language citation, one per line** — "Muslim Marriages and
       Divorces (Registration) Act, 1974, section 3". Nobody should have to
       memorise or hand-copy a `chunk_id` to cite law they found themselves;
       see `ActIndex.resolve_one`. Requires `act_index` (built once in `main()`
       from the whole corpus) — without it this path is skipped and such a line
       fails like any other unrecognised token.
    """
    answer = answer.strip()
    if not answer:
        raise RowError("answer is blank")
    if answer.lower() in NONE_TOKENS:
        return [], "not_found"
    if answer.lower() in UNANSWERABLE_TOKENS:
        return [], "unanswerable"

    ranks = rank_map(candidates)
    markers = CHUNK_MARKER.findall(answer)
    resolved: list[str] = []
    if markers:
        for chunk_id in markers:
            if chunk_id not in by_id:
                raise RowError(f"{chunk_id!r} (from <...>) is not a chunk_id in the corpus")
            if chunk_id not in resolved:          # a repeated marker is not two answers
                resolved.append(chunk_id)
    else:
        legacy = _try_legacy_tokens(answer, ranks, by_id)
        if legacy is not None:
            resolved = legacy
        elif act_index is not None:
            for line in answer.splitlines():
                line = line.strip(" -*•")
                if not line:
                    continue
                chunk_id = act_index.resolve_one(line)
                if chunk_id is None:
                    raise RowError(
                        f"could not identify a provision in {line!r} — "
                        f"name the Act and a section/ধারা number, or paste a "
                        f"<chunk_id> from the search tool"
                    )
                if chunk_id not in resolved:
                    resolved.append(chunk_id)
        else:
            raise RowError(f"could not read answer {answer!r}")
    if not resolved:
        raise RowError(f"could not read answer {answer!r}")
    if len(resolved) > MAX_ANSWERS:
        # The guideline caps this at three: past that, the annotator is almost
        # certainly listing provisions that *relate* to the topic rather than the
        # one that answers the question, and every extra id inflates recall for
        # free at evaluation time.
        raise RowError(
            f"{len(resolved)} answers; the guideline allows at most {MAX_ANSWERS}"
        )
    return resolved, "answered"


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's κ over paired categorical labels. None if fewer than two pairs."""
    if len(pairs) < 2:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    marg_a, marg_b = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(marg_a[k] * marg_b.get(k, 0) for k in marg_a) / (n * n)
    if expected >= 1.0:
        return None                      # every label identical; κ undefined
    return (observed - expected) / (1 - expected)


def read_sheets(sheetdir: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(sheetdir.glob("annotate_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for lineno, row in enumerate(csv.DictReader(fh), 2):
                row["_file"] = path.name
                row["_line"] = lineno
                # Older sheets predate the `annotator` column; fall back to the
                # filename rather than dropping attribution on the floor.
                if not row.get("annotator"):
                    row["annotator"] = path.stem.replace("annotate_", "")
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheets", type=pathlib.Path, default=SHEETS)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--out", type=pathlib.Path, default=GOLD)
    ap.add_argument("--stats", type=pathlib.Path, default=STATS)
    ap.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any row failed validation (use in CI)",
    )
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    by_id = {c.chunk_id: c for c in chunks}
    act_index = ActIndex(chunks)          # built once, reused for every row below
    rows = read_sheets(args.sheets)
    if not rows:
        print(f"no annotate_*.csv under {args.sheets}")
        return 1

    done, blank, errors = [], 0, []
    for row in rows:
        if not row.get("answer", "").strip():
            blank += 1
            continue
        try:
            chunk_ids, verdict = resolve(
                row["answer"], row.get("candidates", ""), by_id, act_index
            )
            register = row.get("register", "").strip().lower()
            if register and register not in VALID_REGISTER:
                raise RowError(f"register {register!r} not in {sorted(VALID_REGISTER)}")
            src = row.get("answer_source", "").strip().lower()
            src = SOURCE_ALIASES.get(src, src)
            if src not in VALID_SOURCE:
                raise RowError(f"answer_source {src!r} not in {sorted(VALID_SOURCE)}")
            domain = (row.get("domain") or "").strip().lower()
            if domain and domain not in VALID_DOMAINS:
                raise RowError(f"domain {domain!r} not in configs/domains.yaml")
            conf = (row.get("confidence") or "").strip()
            if conf and conf not in VALID_CONFIDENCE:
                raise RowError(
                    f"confidence {conf!r} not in {sorted(VALID_CONFIDENCE)}"
                )
            row["_chunk_ids"] = chunk_ids
            row["_verdict"] = verdict
            row["_register"] = register
            row["_source"] = src
            row["_domain"] = domain
            row["_confidence"] = conf
            row["_verified"] = bool((row.get("verified") or "").strip())
            row["_auto"] = (row.get("pass1_auto") or "").strip() == "1"
            done.append(row)
        except RowError as exc:
            errors.append(f"  {row['_file']}:{row['_line']}  {row['qid']}  {exc}")

    print(f"{len(rows)} rows | {len(done)} annotated | {blank} blank | {len(errors)} invalid")

    n_auto = sum(1 for r in done if r["_auto"])
    n_verified = sum(1 for r in done if r["_verified"])
    if n_auto:
        print(
            f"\n{n_verified}/{n_auto} machine-proposed rows verified by a human "
            f"({100 * n_verified // n_auto if n_auto else 0}%)"
        )
        if n_verified == 0:
            print(
                "  *** every stat below is the auto-labeller measured against itself. ***\n"
                "  *** agreement, miss rate and unanswerable count are not real until   ***\n"
                "  *** rows are verified — see docs/annotation_guideline.md.            ***"
            )
    if errors:
        print("\ninvalid rows:")
        for line in errors[:40]:
            print(line)
        if len(errors) > 40:
            print(f"  … and {len(errors) - 40} more")

    # ---- contribution, kept per person on purpose --------------------------
    print("\nper annotator")
    per_person = defaultdict(Counter)
    for row in done:
        per_person[row["annotator"]][row["block"].split("+")[0]] += 1
    for person in sorted(per_person):
        c = per_person[person]
        print(
            f"  {person:12s} {sum(c.values()):4d}  "
            f"(calibration {c['CALIBRATION']}, overlap {c['OVERLAP']}, main {c['MAIN']})"
        )

    # ---- agreement ---------------------------------------------------------
    # Computed over verified rows only once any exist: an unverified row is a
    # machine proposal, and two sheets carrying the same machine's guess on the
    # same question is not a second opinion. Falls back to everyone once nobody
    # has been verified yet, purely so the pipeline is exercisable before Pass 1
    # starts — the loud warning above is what keeps that fallback from being
    # mistaken for a real number.
    agreement_pool = [r for r in done if r["_verified"]] if n_verified else done
    shared: dict[str, list[dict]] = defaultdict(list)
    for row in agreement_pool:
        head = row["block"].split("+")[0]
        if head in {"OVERLAP", "CALIBRATION"}:
            shared[row["qid"]].append(row)

    answer_pairs, provision_pairs = [], []
    domain_pairs, register_pairs = [], []
    for qid, group in shared.items():
        for a, b in itertools.combinations(group, 2):
            if a["annotator"] == b["annotator"]:
                continue           # a re-annotation, not a second opinion
            answer_pairs.append((set(a["_chunk_ids"]), set(b["_chunk_ids"])))
            provision_pairs.append((
                {by_id[c].provision_id for c in a["_chunk_ids"]},
                {by_id[c].provision_id for c in b["_chunk_ids"]},
            ))
            if a["_domain"] and b["_domain"]:
                domain_pairs.append((a["_domain"], b["_domain"]))
            if a["_register"] and b["_register"]:
                register_pairs.append((a["_register"], b["_register"]))

    print(f"\nagreement over {len(answer_pairs)} double-annotated pairs")
    if answer_pairs:
        exact = sum(1 for a, b in answer_pairs if a == b) / len(answer_pairs)
        overlap = sum(1 for a, b in answer_pairs if a & b) / len(answer_pairs)
        prov = sum(1 for a, b in provision_pairs if a & b) / len(provision_pairs)
        # Deliberately NOT reported as κ: with 6,359 candidate chunks, chance
        # agreement rounds to zero and κ would be a relabelled percentage.
        print(f"  answer, exact set match      {exact:.3f}  (raw agreement)")
        print(f"  answer, any chunk shared     {overlap:.3f}  (raw agreement)")
        print(f"  answer, same provision       {prov:.3f}  (raw agreement)")
    for label, pairs in (("domain", domain_pairs), ("register", register_pairs)):
        k = cohen_kappa(pairs)
        if k is None:
            print(f"  {label:28s} —      (need >=2 pairs with >1 label)")
        else:
            print(f"  {label:28s} {k:.3f}  (Cohen's kappa, n={len(pairs)})")

    # ---- what the candidate list could not reach ---------------------------
    # Same gate as agreement above: a verdict or source the auto-labeller wrote
    # and nobody has checked is not evidence about the corpus or the candidate
    # list, only about the labeller. Falls back to `done` before any row is
    # verified so the script stays runnable pre-Pass-1; the loud warning is what
    # stops that fallback from being quoted as a result.
    reach_pool = [r for r in done if r["_verified"]] if n_verified else done
    sources = Counter(row["_source"] for row in reach_pool if row["_source"])
    verdicts = Counter(row["_verdict"] for row in reach_pool)
    unaided = [r for r in reach_pool if r.get("annotation_mode") == "unaided"]
    # An `unanswerable` row belongs in neither half of the miss rate: BM25 did not
    # fail to surface the right provision, there was no right provision to
    # surface. Leaving those in the denominator would deflate the miss rate in
    # proportion to how many unanswerable questions the pool happened to contain,
    # which has nothing to do with lexical reach.
    #
    # A `not_found` row is the opposite and *does* belong in the numerator: the
    # candidate list failed and so did the annotator's own search, which is the
    # most complete failure of lexical retrieval there is. Dropping those would
    # bias the miss rate downwards exactly where the lexical gap is widest.
    assisted = [
        r for r in reach_pool
        if r.get("annotation_mode", "assisted") == "assisted"
        and r["_verdict"] != "unanswerable"
    ]
    missed = sum(
        1 for r in assisted
        if r["_source"] == "own_search" or r["_verdict"] == "not_found"
    )
    print("\ncandidate-list reach")
    print(f"  answer_source                {dict(sources)}")
    print(f"  verdict                      {dict(verdicts)}")
    if assisted:
        print(
            f"  candidate-list miss rate     {missed / len(assisted):.3f}  "
            f"({missed}/{len(assisted)} answerable assisted rows)"
        )
    print(
        f"  unaided control rows         {len(unaided)}  "
        f"(from {len({r['qid'] for r in unaided})} questions)"
    )
    print(
        f"  unanswerable                 {verdicts['unanswerable']}"
        f"   <- calibrates abstention"
    )
    print(
        f"  not found by annotator       {verdicts['not_found']}"
        f"   <- lexical gap, not abstention"
    )

    # ---- write -------------------------------------------------------------
    # One record per (qid, annotator). Adjudicating the double-annotated rows
    # into a single answer is a human decision and is deliberately not automated
    # here; 09_run_eval consumes the adjudicated file, not this one.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in done:
            fh.write(json.dumps({
                "qid": row["qid"],
                "annotator": row["annotator"],
                "block": row["block"],
                "annotation_mode": row.get("annotation_mode", "assisted"),
                "source": "mined",
                "question_bn": row["question_bn"],
                "relevant_chunk_ids": row["_chunk_ids"],
                "relevant_provision_ids": sorted(
                    {by_id[c].provision_id for c in row["_chunk_ids"]}
                ),
                "verdict": row["_verdict"],
                "answer_source": row["_source"],
                "register": row["_register"] or None,
                "domain": row["_domain"] or None,
                "confidence": row["_confidence"] or None,
                "paraphrase": (row.get("paraphrase") or "").strip() or None,
                "formal_version": (row.get("formal_version") or "").strip() or None,
                "notes": (row.get("notes") or "").strip() or None,
            }, ensure_ascii=False) + "\n")
    print(f"\n{len(done)} records -> {args.out}")

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["annotator", "block", "n"])
        for person in sorted(per_person):
            for block, n in sorted(per_person[person].items()):
                w.writerow([person, block, n])
    print(f"{args.stats}")

    return 1 if (args.strict and errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
