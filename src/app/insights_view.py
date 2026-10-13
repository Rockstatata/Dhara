"""HTML for the classification + sequence-tagging side panel and results tab.

Presentation only: it renders what `dhara.insights` returns and what the recorded
run JSONs say. Every score shown as a *result* is read from `results/runs/*.json`
(no hand-typed numbers, CLAUDE.md); the only literals here are labels and colours.
"""

from __future__ import annotations

import html
import json

from dhara.insights import (
    CLASSIFY_RUN, TAGGER_RUN, DomainClassifier, QueryTagger, TaggedQuery, domain_names,
)

# (background, text, Bangla gloss). Light backgrounds with dark text stay legible
# on both of Gradio's themes, same approach as the risk banner in app.py.
TAG_STYLE = {
    "ACT":        ("#dbeafe", "#1e3a8a", "আইন"),
    "SECTION_NO": ("#fde68a", "#78350f", "ধারা নম্বর"),
    "LEGAL_TERM": ("#e9d5ff", "#581c87", "আইনি পরিভাষা"),
    "PARTY":      ("#bbf7d0", "#14532d", "পক্ষ"),
}

_CARD = ("border:1px solid var(--border-color-primary); border-radius:12px; "
         "padding:12px 14px; margin-bottom:12px; background:var(--background-fill-secondary);")
_MUTED = "font-size:0.82em; color:var(--body-text-color-subdued);"


def _chip(tok: str, tag: str) -> str:
    if tag == "O":
        return f'<span style="display:inline-block;margin:2px 3px 2px 0;">{html.escape(tok)}</span>'
    bg, fg, _ = TAG_STYLE[tag]
    return (f'<span title="{tag}" style="display:inline-block;margin:2px 3px 2px 0;padding:1px 7px;'
            f'border-radius:6px;background:{bg};color:{fg};font-weight:600;">'
            f'{html.escape(tok)}<sup style="font-size:0.62em;margin-left:3px;opacity:0.8;">{tag}</sup></span>')


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class InsightsView:
    """Owns the loaded classifier/tagger and the static numbers shown beside them."""

    def __init__(self, classifier: DomainClassifier, tagger: QueryTagger):
        self.classifier = classifier
        self.tagger = tagger
        self.names = domain_names()
        self.f1 = DomainClassifier.recorded_f1()
        cls_run = _load_json(CLASSIFY_RUN)["human_only_fine_grained"]
        lr = cls_run["models"]["logistic_regression"]
        self.cls_accuracy, self.cls_macro_f1 = lr["accuracy"], lr["macro_f1"]
        self.cls_n_test, self.cls_n_train = cls_run["n_test"], cls_run["n_train"]
        self.tag_prf = _load_json(TAGGER_RUN)["dev_per_tag_prf_self_consistency"]

    def _dom(self, domain_id: str) -> str:
        bn, en = self.names.get(domain_id, (domain_id, domain_id))
        return f"{html.escape(bn)} <span style='{_MUTED}'>· {html.escape(en)}</span>"

    # ------------------------------------------------------------ classifier
    def classifier_html(self, query: str, top_hit_domain: str | None = None) -> str:
        preds = self.classifier.predict(query, top_n=3)
        rows = []
        for i, (dom, p) in enumerate(preds):
            f1 = self.f1.get(dom)
            trust = ("" if f1 is None else
                     f"<span style='{_MUTED}'>held-out F1 {f1:.2f}</span>")
            weight = "700" if i == 0 else "400"
            rows.append(
                f'<div style="margin:6px 0;"><div style="display:flex;justify-content:space-between;'
                f'gap:8px;font-weight:{weight};"><span>{self._dom(dom)}</span><span>{p:.0%}</span></div>'
                f'<div style="height:6px;border-radius:3px;background:var(--border-color-primary);">'
                f'<div style="height:6px;border-radius:3px;width:{max(p, 0.02) * 100:.0f}%;'
                f'background:{"#10b981" if i == 0 else "#94a3b8"};"></div></div>{trust}</div>'
            )
        cross = ""
        if top_hit_domain:
            agree = preds[0][0] == top_hit_domain
            cross = (f"<div style='{_MUTED}margin-top:6px;'>retrieval's top result is in "
                     f"<b>{html.escape(top_hit_domain)}</b> — "
                     f"{'agrees with' if agree else 'differs from'} the classifier's top guess.</div>")
        return (
            f'<div style="{_CARD}"><div style="font-weight:700;margin-bottom:2px;">'
            f'বিষয় শনাক্তকরণ · Domain classifier</div>'
            f'<div style="{_MUTED}">Logistic Regression on TF-IDF · trained on {self.cls_n_train} human questions</div>'
            f'{"".join(rows)}{cross}'
            f'<div style="{_MUTED}margin-top:8px;border-top:1px dashed var(--border-color-primary);padding-top:6px;">'
            f'Accuracy {self.cls_accuracy:.1%}, macro-F1 {self.cls_macro_f1:.3f} on {self.cls_n_test} held-out '
            f'questions. Scores are uncalibrated and often low — a side track shown for '
            f'transparency; it never filters or re-ranks the retrieved provisions.</div></div>'
        )

    # ---------------------------------------------------------------- tagger
    def tagger_html(self, query: str) -> str:
        tq: TaggedQuery = self.tagger.tag(query)
        if not tq.tokens:
            return ""
        chips = " ".join(_chip(t, g) for t, g in zip(tq.tokens, tq.tags))
        ents = tq.entities()
        if ents:
            summary = "".join(
                f'<div style="margin-top:4px;"><b>{TAG_STYLE[t][2]}</b> '
                f'<span style="{_MUTED}">{t}</span>: {html.escape(" · ".join(toks))}</div>'
                for t, toks in ents.items()
            )
        else:
            summary = f"<div style='{_MUTED}margin-top:4px;'>No entity tags predicted for this question.</div>"

        rule = ""
        if tq.rule_section_numbers:
            rule = (f'<div style="margin-top:6px;padding:6px 8px;border-radius:8px;background:#fef3c7;'
                    f'color:#78350f;">🔎 <b>Rule-based:</b> explicit section number '
                    f'{html.escape(", ".join(tq.rule_section_numbers))} found in the question '
                    f'(a regex, not the neural tagger).</div>')

        a, p = self.tag_prf["ACT"]["f1"], self.tag_prf["PARTY"]["f1"]
        s, lt = self.tag_prf["SECTION_NO"]["f1"], self.tag_prf["LEGAL_TERM"]["f1"]
        legend = " ".join(
            f'<span style="padding:0 6px;border-radius:5px;background:{bg};color:{fg};font-size:0.8em;">'
            f'{t}</span>' for t, (bg, fg, _) in TAG_STYLE.items()
        )
        return (
            f'<div style="{_CARD}"><div style="font-weight:700;margin-bottom:2px;">'
            f'প্রশ্নের অংশ চিহ্নিতকরণ · Sequence tagger</div>'
            f'<div style="{_MUTED}">BiRNN, token-level · {legend}</div>'
            f'<div style="margin-top:8px;line-height:1.9;">{chips}</div>'
            f'{summary}{rule}'
            f'<div style="{_MUTED}margin-top:8px;border-top:1px dashed var(--border-color-primary);padding-top:6px;">'
            f'Trained on weak (rule-generated) labels, never hand-corrected. Scores are agreement with '
            f'those rules, not human-verified accuracy: ACT F1 {a:.2f}, PARTY F1 {p:.2f}; '
            f'SECTION_NO {s:.2f} and LEGAL_TERM {lt:.2f} were not learned. Display only.</div></div>'
        )

    def panel_html(self, query: str, top_hit_domain: str | None = None) -> str:
        """Both cards. Each fails independently -- a broken side track must never
        take the retrieval result down with it."""
        parts = []
        for name, fn in (("classifier", lambda: self.classifier_html(query, top_hit_domain)),
                         ("tagger", lambda: self.tagger_html(query))):
            try:
                parts.append(fn())
            except Exception as exc:  # noqa: BLE001 - presentation boundary
                parts.append(f'<div style="{_CARD}{_MUTED}">{name} unavailable: {html.escape(str(exc))}</div>')
        return "".join(parts)


# --------------------------------------------------------------- results tab

def results_markdown() -> str:
    """Tables for the results tab, straight from the recorded runs."""
    cls = _load_json(CLASSIFY_RUN)
    tag = _load_json(TAGGER_RUN)

    variants = [
        ("human_only_fine_grained", "human questions only"),
        ("balanced_mix", "human + 800 filtered synthetic"),
        ("fine_grained_rebalanced", "human + all synthetic"),
    ]
    models = [("naive_bayes", "Naive Bayes (generative)"), ("logistic_regression", "Logistic Regression"),
              ("vanilla_rnn", "Vanilla RNN"), ("stacked_bilstm", "Stacked BiLSTM")]

    lines = ["### Classification — predicting the legal domain of a question", "",
             f"Test set: {cls['n_test']} real human questions, never trained on. "
             "Metric is **macro-F1** (accuracy alone would reward always answering the biggest class).", "",
             "| training data | n train | " + " | ".join(n for _, n in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for key, label in variants:
        v = cls[key]
        cells = []
        best = max(v["models"][m]["macro_f1"] for m, _ in models)
        for m, _ in models:
            x = v["models"][m]
            cell = f"{x['macro_f1']:.3f} (acc {x['accuracy']:.2f})"
            cells.append(f"**{cell}**" if x["macro_f1"] == best else cell)
        lines.append(f"| {label} | {v['n_train']} | " + " | ".join(cells) + " |")
    lines += ["",
              "The discriminative model wins on identical features, and **more synthetic training data made it "
              "worse, not better** — synthetic questions do not share the register of real ones. The served "
              "classifier is the bold cell in the first row: Logistic Regression on human questions only.", "",
              "### Sequence tagging — ACT / SECTION_NO / LEGAL_TERM / PARTY", "",
              f"Trained on {tag['n_train_examples']} questions with weak, rule-generated labels; evaluated on "
              f"{tag['n_dev_examples']} held-out questions **against the same rules**. This measures how well the "
              "network learned the rules, not how well it tags against a human gold standard — hand-correction "
              "was not done.", "",
              "| tag | weak-labelled tokens (train+dev) | precision | recall | F1 |", "|---|---|---|---|---|"]
    counts = tag["weak_label_tag_distribution"]
    for t, v in tag["dev_per_tag_prf_self_consistency"].items():
        lines.append(f"| {t} | {counts[t]} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |")
    lines += ["",
              f"Entity macro-F1 {tag['dev_entity_macro_f1_self_consistency']:.3f}. Token accuracy "
              f"({tag['dev_token_accuracy_self_consistency']:.1%}) is not informative — "
              f"{100 * counts['O'] / sum(counts.values()):.0f}% of tokens are `O`. SECTION_NO has almost no "
              "training signal because citizens rarely cite section numbers, so explicit numbers are extracted "
              "by a regex instead (shown separately in the search tab)."]
    return "\n".join(lines)
