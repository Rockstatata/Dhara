"""Recover the advocate's published reply for every mined question.

    python scripts/11_extract_answers.py

Writes `data/interim/lawyer_answers.jsonl` — one record per question that had a
reply, keyed by the same `qid` the question pool uses.

Every question in the pool was answered in print by a practising advocate, and
that reply has been sitting in `mined_articles.jsonl` all along: the splitter
selects segments containing a question mark, and a reply rarely has one, so the
replies were parsed and then dropped on the floor.

They are worth recovering because of the register they are written in. A reader
writes «কো-অপারেটিভ সংস্থার কাছে আমার টাকা আটকে আছে»; the advocate replies
«দেওয়ানি আইনের আওতায় মানি মোকদ্দমা করতে পারেন». The second is the vocabulary the
statute itself uses, which makes it a far better retrieval query than the
question — and the gap between the two phrasings is the thing this project exists
to measure.

**A reply is evidence, not a label.** Advocates name a remedy or an Act and only
sometimes a section; on questions with no statutory answer they give practical
advice citing no law whatsoever. So this feeds two things — a better retrieval
query for the machine proposal, and a column the human verifier reads — and it
never becomes the answer by itself.

**Copyright.** The reply is the newspaper's text, exactly like the question.
`data/interim/` is git-ignored, and the reply must not enter the released
dataset: it stays alongside `text_verbatim`, not alongside the paraphrase.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import questions as q  # noqa: E402

ARTICLES = pathlib.Path("data/interim/mined_articles.jsonl")
POOL = pathlib.Path("data/interim/questions_pool.jsonl")
OUT = pathlib.Path("data/interim/lawyer_answers.jsonl")

# An Act named in the reply, or a section number, is the strongest signal a reply
# can carry and is worth counting separately — those are the replies that point a
# verifier straight at a provision.
CITES_ACT = re.compile(r"আইন|অধ্যাদেশ|ধারা|Act|Ordinance|section", re.I)
CITES_SECTION = re.compile(r"ধারা\s*[০-৯0-9]|section\s*\d", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--articles", type=pathlib.Path, default=ARTICLES)
    ap.add_argument("--pool", type=pathlib.Path, default=POOL)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    articles = [
        json.loads(line) for line in args.articles.open(encoding="utf-8") if line.strip()
    ]
    pool_ids = {
        json.loads(line)["qid"]
        for line in args.pool.open(encoding="utf-8")
        if line.strip()
    }

    replies: dict[str, str] = {}
    for index, article in enumerate(articles):
        replies.update(q.replies_for_article(article, index))

    matched = {qid: text for qid, text in replies.items() if qid in pool_ids}
    stats = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for qid, text in sorted(matched.items()):
            names_act = bool(CITES_ACT.search(text))
            names_section = bool(CITES_SECTION.search(text))
            stats["names_an_act_or_provision"] += names_act
            stats["names_a_section_number"] += names_section
            stats["no_law_mentioned"] += not names_act
            fh.write(
                json.dumps(
                    {
                        "qid": qid,
                        "answer_verbatim": text,
                        "names_act": names_act,
                        "names_section": names_section,
                        "n_words": len(text.split()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"articles                     {len(articles)}")
    print(f"replies parsed               {len(replies)}")
    print(f"questions in pool            {len(pool_ids)}")
    print(f"pool questions with a reply  {len(matched)} "
          f"({100 * len(matched) // max(len(pool_ids), 1)}%)")
    for key in sorted(stats):
        print(f"  {key:28s} {stats[key]:5d}")
    print(f"\n-> {args.out}   (git-ignored: this is the newspaper's text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
