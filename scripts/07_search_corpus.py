"""The annotator's search tool — for the questions the candidate list missed.

    python scripts/07_search_corpus.py
    python scripts/07_search_corpus.py --query "ভরণপোষণ" --k 10
    python scripts/07_search_corpus.py --act "Succession"
    python scripts/07_search_corpus.py --cite labour_2006_s103

Why this exists, and why it is not cheating: the candidate lists in the
annotation sheets come from BM25 over Bangla text, and roughly a third of this
corpus is English-only — including the Succession Act, the Registration Act and
the Code of Criminal Procedure. A Bangla question about inheritance therefore has
*no* reachable answer in its candidate list, not because the annotator is wrong
but because the lexical retriever cannot cross languages. That failure is one of
the things this project set out to measure.

So the annotator needs a way to find the true provision anyway. When they do,
they record `answer_source = own_search`, and the fraction of gold answers that
BM25 never offered becomes a reported number instead of a silent hole in the
gold set.

Three search modes, because the three ways an annotator gets stuck are different:

  --query : BM25 over provision text. What the sheet already did, but live, and
            with as many results as you want.
  --act   : substring match on act title, Bangla or English, then lists that
            Act's provisions. This is the escape hatch for English Acts — you
            know it is the Succession Act, you just need to see its sections.
  --cite  : look up one chunk_id and print it in full, to check an answer before
            committing it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara import schema  # noqa: E402
from dhara.normalize import aggressive  # noqa: E402
from dhara.retrievers.bm25 import BM25Retriever  # noqa: E402

CORPUS = pathlib.Path("data/processed/corpus_v1.jsonl")
RULE = "-" * 78


def show(chunk: schema.Chunk, score: float | None = None, full: bool = False) -> None:
    head = f"<{chunk.chunk_id}>"
    if score is not None:
        head = f"{score:6.2f}  {head}"
    print(head)
    print(f"        {chunk.citation()}   [{chunk.language}, {chunk.domain}]")
    if chunk.provision_title_bn:
        print(f"        {chunk.provision_title_bn}")
    body = chunk.text_raw if full else chunk.text_raw[:300].replace("\n", " ")
    print(f"        {body}{'' if full else ' …'}")
    if full:
        print(f"        {chunk.source_url}   (crawled {chunk.crawl_date})")
    print()


def search_acts(chunks: list[schema.Chunk], needle: str) -> None:
    """List provisions of every Act whose title contains `needle`.

    Matches on the aggressively-normalized title so that a Bangla title typed
    with different ZWNJ placement, or an English title typed in lower case, still
    finds its Act.
    """
    key = aggressive(needle)
    hits = [
        c
        for c in chunks
        if key in aggressive(c.act_title_bn) or key in aggressive(c.act_title_en)
    ]
    if not hits:
        print(f"no Act title contains {needle!r}")
        return
    acts = sorted({(c.act_id, c.act_title_bn, c.act_title_en) for c in hits})
    for act_id, title_bn, title_en in acts:
        provisions = sorted(
            (c for c in hits if c.act_id == act_id),
            key=lambda c: (len(c.provision_no_ascii), c.provision_no_ascii, c.sub_idx),
        )
        print(f"\n{RULE}\n{title_bn or title_en}  ({act_id}, {len(provisions)} chunks)\n{RULE}")
        for c in provisions:
            title = c.provision_title_bn or c.text_raw[:70].replace("\n", " ")
            print(f"  <{c.chunk_id}>  {c.provision_no_bn:>6}  {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS)
    ap.add_argument("--query", help="BM25 search over provision text")
    ap.add_argument("--act", help="substring of an Act title, Bangla or English")
    ap.add_argument("--cite", help="one chunk_id, printed in full")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    chunks = list(schema.read_jsonl(args.corpus))
    by_id = {c.chunk_id: c for c in chunks}

    if args.cite:
        chunk = by_id.get(args.cite)
        if chunk is None:
            print(f"no chunk with id {args.cite!r}")
            return 1
        show(chunk, full=True)
        return 0

    if args.act:
        search_acts(chunks, args.act)
        return 0

    # Indexing costs a few seconds, so in interactive mode it happens once and
    # the prompt loops — an annotator will run twenty searches in a sitting.
    retriever = BM25Retriever()
    retriever.index(chunks)

    def run(q: str) -> None:
        results = retriever.search(q, k=args.k)
        if not results:
            print("  (nothing — every word in that query is a stopword)")
            return
        for chunk_id, score in results:
            show(by_id[chunk_id], score)

    if args.query:
        run(args.query)
        return 0

    print(f"{len(chunks)} chunks indexed.\n")
    print("  type a query, or:   act:<title fragment>    id:<chunk_id>    quit\n")
    while True:
        try:
            line = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"quit", "exit", "q"}:
            return 0
        if line.startswith("act:"):
            search_acts(chunks, line[4:].strip())
        elif line.startswith("id:"):
            chunk = by_id.get(line[3:].strip())
            show(chunk, full=True) if chunk else print("  no such chunk_id")
        else:
            run(line)


if __name__ == "__main__":
    raise SystemExit(main())
