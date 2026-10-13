"""Capture real input -> output pairs from the served system, for the report.

    python scripts/77_capture_demo_io.py

Runs the same code path the Gradio demo runs (`Dhara.search` + `insights`), on a
fixed list of questions written for this purpose, and stores everything the report
shows in `docs/report/generated/demo_io.json`. The questions are deliberately NOT
taken from the gold set: they are illustrations, not evaluation, so capturing
them touches no held-out data. There is no ground truth here -- the report says
so, and judges relevance only by reading the retrieved titles.

`show_all=True` matches the demo, so low-confidence results are still recorded
alongside the `abstained` flag rather than dropped.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dhara.insights import DomainClassifier, QueryTagger, domain_names  # noqa: E402
from dhara.service import Dhara  # noqa: E402

OUT = pathlib.Path("docs/report/generated/demo_io.json")

# (label, question). Chosen to span a Bangla-Act domain, English-only-Act domains,
# an explicit section number, a colloquial word absent from the vocabulary, and an
# out-of-scope question -- including the cases the system handles badly.
QUESTIONS = [
    ("family_talaq", "আমার স্বামী মুখে তালাক দিয়েছে, এখন কী করব?"),
    ("labour_wages", "মালিক তিন মাস ধরে বেতন দিচ্ছে না, আমি কী করতে পারি?"),
    ("cyber_hack", "কেউ আমার ফেসবুক আইডি হ্যাক করে আমার নামে কুৎসা ছড়াচ্ছে, কী করব?"),
    ("land_mutation", "জমির নামজারি করতে কী কী কাগজ লাগে?"),
    ("explicit_section", "দণ্ডবিধির ৩২৩ ধারায় কী শাস্তি হয়?"),
    ("out_of_scope", "আজকে ঢাকার আবহাওয়া কেমন?"),
]


def main() -> None:
    dhara = Dhara.load()
    clf, tagger, names = DomainClassifier.fit(), QueryTagger.load(), domain_names()

    records = []
    for label, question in QUESTIONS:
        answer = dhara.search(question, k=3, show_all=True)
        tq = tagger.tag(question)
        records.append({
            "label": label,
            "question": question,
            "abstained": answer.abstained,
            "top_score": answer.top_score,
            "threshold": answer.threshold,
            "risk_tier": answer.risk_tier,
            "legal_aid_referral": answer.legal_aid_referral,
            "hits": [{
                "score": round(h.score, 4), "act_title": h.act_title, "act_year": h.act_year,
                "provision_kind": h.provision_kind, "provision_no": h.provision_no,
                "provision_title": h.provision_title, "domain": h.domain,
                "risk_tier": h.risk_tier, "text_raw": h.text_raw[:400],
                "source_url": h.source_url, "crawl_date": h.crawl_date,
            } for h in answer.hits],
            "classifier": [{"domain": d, "name_en": names[d][1], "p": round(p, 4)}
                           for d, p in clf.predict(question, top_n=3)],
            "tags": [{"token": t, "tag": g} for t, g in zip(tq.tokens, tq.tags)],
            "rule_section_numbers": tq.rule_section_numbers,
        })
        print(f"{label}: top={answer.top_score:.3f} abstained={answer.abstained} "
              f"clf={records[-1]['classifier'][0]['domain']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "index": dhara.manifest.get("built_by"), "n_chunks": len(dhara.chunk_ids),
        "checkpoint": dhara.manifest.get("checkpoint"), "thresholds": dhara.thresholds,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
