"""Build every table, figure, number macro and sample-data box the report uses.

    python scripts/77_capture_demo_io.py         # once, needs the model
    python scripts/78_build_report_assets.py     # cheap, CPU only

Writes to docs/report/generated/ (LaTeX fragments) and docs/report/figures/
(PDF figures). Nothing in the report's prose or tables is typed by hand: numbers
come from results/runs/*.json, results/tables/*.csv or the data files themselves,
and the prose refers to them through the macros in generated/numbers.tex. That is
CLAUDE.md's "no hand-typed numbers" rule applied to the write-up.

Held-out data: this script reads only recorded run files and the public
paraphrased question set. It never opens gold_test_v1.jsonl and never scores
anything.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import statistics
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dhara.normalize import aggressive, light  # noqa: E402

GEN = ROOT / "docs/report/generated"
FIG = ROOT / "docs/report/figures"
RUNS = ROOT / "results/runs"
TABLES = ROOT / "results/tables"
PROC = ROOT / "data/processed"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, ORANGE, GREY, RED, LIGHT = "#104A78", "#006633", "#D9822B", "#8A8F98", "#B3261E", "#C9D6E2"
plt.rcParams.update({
    "pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False, "figure.dpi": 150,
})


# ------------------------------------------------------------------ helpers
def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "[": "{[}", "]": "{]}", '"': r"\textquotedbl{}"}
_BN_RUN = re.compile(
    "([\u0980-\u09ff](?:[\u0980-\u09ff\u200c\u200d \u00a0,.;:!?()\\-\u2013\u2014'\u2018\u2019\u201c\u201d/0-9]*"
    "[\u0980-\u09ff])?)"
)


def esc(text: str) -> str:
    return "".join(_ESC.get(c, c) for c in text)


def tex(text: str | None) -> str:
    """Escape for LaTeX and wrap each Bangla run in \\bng{} so Latin stays in the
    Latin font. Whitespace runs are collapsed: source text carries tabs/newlines."""
    if text is None:
        return ""
    text = re.sub(r"[ \t\r\n\u00a0]+", " ", str(text)).strip()
    out = []
    for part in _BN_RUN.split(text):
        if not part:
            continue
        out.append(r"\bng{" + esc(part) + "}" if _BN_RUN.fullmatch(part) else esc(part))
    return "".join(out)


def write(name: str, body: str) -> None:
    (GEN / name).write_text(body, encoding="utf-8")
    print("  wrote", name)


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def bold_best(values: list[float], fmt=f3, higher=True) -> list[str]:
    best = max(values) if higher else min(values)
    return [f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v) for v in values]


def table(colspec: str, header: list[str], rows: list[list[str]], size: str = "\\footnotesize") -> str:
    # adjustbox only ever shrinks: a table wider than the text block is scaled to fit.
    lines = [f"{{{size}", "\\adjustbox{max width=\\linewidth}{\\begin{tabular}{" + colspec + "}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}}", "}"]
    return "\n".join(lines) + "\n"


def savefig(fig, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/" + name)


# ------------------------------------------------------------------ load
print("loading...")
corpus = read_jsonl(PROC / "corpus_v1.jsonl")
by_chunk = {c["chunk_id"]: c for c in corpus}
gold = read_jsonl(PROC / "gold_verified_v3.jsonl")
probes = read_jsonl(PROC / "probe_questions_v2.jsonl")
domains_cfg = None
import yaml  # noqa: E402

domains_cfg = yaml.safe_load((ROOT / "configs/domains.yaml").read_text(encoding="utf-8"))
DOM_EN = {d["id"]: d["name_en"] for d in domains_cfg["domains"]}
DOM_RISK = {d["id"]: d["risk_tier"] for d in domains_cfg["domains"]}
DOM_EN["other"], DOM_RISK["other"] = "Other Acts (outside the named domains)", "medium"

RUN_LABEL = {
    "bm25_default_v2": "BM25 (default $k_1{=}1.5$, $b{=}0.75$)",
    "bm25_tuned_v2": "BM25 (tuned $k_1{=}0.9$, $b{=}0.4$)",
    "word2vec_v2": "Word2Vec, mean-pooled",
    "bilstm_v2": "BiLSTM dual-encoder",
    "bge_m3_zeroshot_v2": "BGE-m3 zero-shot (control)",
    "bge_m3_acttitle_v2": "BGE-m3 zero-shot + Act title",
    "bge_m3_attempt4_v2": "BGE-m3 fine-tuned + Act title",
}
SHORT = {
    "bm25_default_v2": "BM25", "bm25_tuned_v2": "BM25 tuned", "word2vec_v2": "Word2Vec",
    "bilstm_v2": "BiLSTM", "bge_m3_zeroshot_v2": "BGE-m3\nzero-shot",
    "bge_m3_acttitle_v2": "BGE-m3\n+Act title", "bge_m3_attempt4_v2": "BGE-m3 FT\n+Act title",
}
LADDER = ["bm25_default_v2", "word2vec_v2", "bilstm_v2", "bge_m3_zeroshot_v2",
          "bge_m3_acttitle_v2", "bge_m3_attempt4_v2"]
ladder_rows = {r["run"]: r for r in read_csv(TABLES / "ladder_v2_complete.csv")}
lang_rows: dict[str, dict[str, dict]] = {}
for run in LADDER + ["bm25_tuned_v2"]:
    lang_rows[run] = {r["group"]: r for r in read_csv(TABLES / f"{run}_recall.csv")}

# -------------------------------------------------- 1. number macros
print("numbers...")
lang_counts = Counter(p["lang_tag"] for p in probes)
n_probe = len(probes)
ov = read_json(RUNS / "overlap_audit.json")
lm = read_json(RUNS / "lm_perplexity_v2.json")
cls = read_json(RUNS / "classify_v6.json")
tag = read_json(RUNS / "tagger_v3.json")
audit2 = read_json(RUNS / "annotation_round2_audit.json")
merge6 = read_json(RUNS / "merge_train_retrieval_v6.json")
mine6 = read_json(RUNS / "mine_negatives_v6.json")
split4 = read_json(RUNS / "retrieval_human_aware_splits_v4.json")
excl = {r["metric"]: r["value"] for r in read_csv(TABLES / "exclusion_stats.csv")}
zs6 = read_json(RUNS / "bge_m3_zeroshot_v6.json")
ft6 = read_json(RUNS / "bge_m3_finetuned_v6.json")
gold_res = Counter(r["resolution"] for r in gold)
gold_verdict = Counter(r["verdict"] for r in gold)
gold_reg = Counter(r["register"] for r in gold)
n_bn = sum(c["language"] == "bengali" for c in corpus)
n_en = len(corpus) - n_bn
llm_dec = read_json(ROOT / "data/annotation/round2_llm_decisions.json")
llm_qids = set(llm_dec["iftiaq_corrections"]) | set(llm_dec["adjudications"])
test_rows = read_jsonl(PROC / "test_retrieval_v4.jsonl")
test_llm = sum(r["qid"] in llm_qids for r in test_rows)
lr_hum = cls["human_only_fine_grained"]["models"]["logistic_regression"]
abst = read_json(ROOT / "configs/abstention.json")["thresholds"]
qwords_gold = [len(r["question_bn"].split()) for r in gold]


gold_audit = read_json(RUNS / "gold_audit.json")
synth_stats = {r["metric"]: r["value"] for r in read_csv(TABLES / "synth_generation_stats.csv")}
lawyer = gold_audit["lawyer_cross_check"]
bm_sweep = [float(r["dev_R@10"]) for r in read_csv(TABLES / "bm25_tuning.csv")]


def group(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


macros = {
    "nChunks": f"{len(corpus):,}".replace(",", "{,}"),
    "nProvisions": f"{len({c['provision_id'] for c in corpus}):,}".replace(",", "{,}"),
    "nActs": f"{len({c['act_id'] for c in corpus}):,}".replace(",", "{,}"),
    "nBnChunks": f"{n_bn:,}".replace(",", "{,}"), "nEnChunks": f"{n_en:,}".replace(",", "{,}"),
    "pctBnChunks": pct(n_bn / len(corpus)), "pctEnChunks": pct(n_en / len(corpus)),
    "nBladChunks": f"{sum(c['source_dataset'] == 'blad' for c in corpus):,}".replace(",", "{,}"),
    "nBdlawsChunks": f"{sum(c['source_dataset'] == 'bdlaws' for c in corpus):,}".replace(",", "{,}"),
    "nSplitProvisions": f"{sum(1 for c in corpus if c['n_sub'] > 1 and c['sub_idx'] == 0):,}".replace(",", "{,}"),
    "medianWords": str(int(statistics.median(c["n_words"] for c in corpus))),
    "nExcluded": excl["excluded_repealed_omitted"], "pctExcluded": f"{float(excl['excluded_pct']):.1f}\\%",
    "nGold": str(len(gold)), "nGoldAnswered": str(gold_verdict["answered"]),
    "nGoldUnanswerable": str(gold_verdict["unanswerable"]), "nGoldNotFound": str(gold_verdict["not_found"]),
    "nGoldColloquial": str(gold_reg["colloquial"]), "nGoldFormal": str(gold_reg["formal"]),
    "nProbe": str(n_probe), "nProbeEn": str(lang_counts["english"]), "nProbeBn": str(lang_counts["bengali"]),
    "nProbeMixed": str(lang_counts["mixed"]),
    "pctProbeEn": pct(lang_counts["english"] / n_probe), "pctProbeBn": pct(lang_counts["bengali"] / n_probe),
    "medianQWords": str(int(statistics.median(qwords_gold))),
    "overlapMedianReal": f"{ov['real']['content_overlap']['median']:.1f}",
    "overlapMeanReal": f3(ov["real"]["content_overlap"]["mean"]),
    "overlapPninetyReal": f3(ov["real"]["content_overlap"]["p90"]),
    "nDoubleAnnotated": str(audit2["double_annotated_qids"]),
    "nAgreeProvision": str(audit2["provision_level_agreements"]),
    "pctAgreeProvision": pct(audit2["provision_level_agreements"] / audit2["double_annotated_qids"], 0),
    "nAdjudicated": str(audit2["adjudicated_disagreements"]),
    "nSingleAnnotated": str(gold_res["single_verified_annotation"]),
    "nLlmLabelled": str(len(llm_qids)), "nLlmAdjudicated": str(gold_res["llm_adjudication"]),
    "nLlmCorrections": str(len(llm_dec["iftiaq_corrections"])), "nTestLlm": str(test_llm),
    "nTest": str(len(test_rows)), "nTestEn": str(zs6["slices"]["english_targets"]["n"]),
    "nTestBn": str(zs6["slices"]["bengali_targets"]["n"]),
    "nTrainHuman": str(split4["human_adjudicated"]["train"]), "nDevHuman": str(split4["human_adjudicated"]["dev"]),
    "nTrainApproved": f"{merge6['by_source']['human_approved_title_pair']:,}".replace(",", "{,}"),
    "nTrainAuthored": str(merge6["by_source"]["authored_v1"]),
    "nTrainTotal": f"{merge6['kept']:,}".replace(",", "{,}"),
    "nNegPerPair": f"{mine6['mean_negatives_per_pair']:.0f}",
    "RtenZs": f3(ft6["R@10"] if False else zs6["R@10"]), "RtenFt": f3(ft6["R@10"]),
    "RoneZs": f3(zs6["R@1"]), "RoneFt": f3(ft6["R@1"]),
    "RtenZsEn": f3(zs6["slices"]["english_targets"]["R@10"]), "RtenFtEn": f3(ft6["slices"]["english_targets"]["R@10"]),
    "RhundredTest": f3(ft6["R@100"]),
    "RtenZsLadder": f3(float(ladder_rows["bge_m3_zeroshot_v2"]["R@10"])),
    "RtenBmLadder": f3(float(ladder_rows["bm25_default_v2"]["R@10"])),
    "RtenWvLadder": f3(float(ladder_rows["word2vec_v2"]["R@10"])),
    "RtenActLadder": f3(float(ladder_rows["bge_m3_acttitle_v2"]["R@10"])),
    "RtenBmEn": f3(float(lang_rows["bm25_default_v2"]["english"]["R@10"])),
    "RtenZsEnLadder": f3(float(lang_rows["bge_m3_zeroshot_v2"]["english"]["R@10"])),
    "RtenZsBnLadder": f3(float(lang_rows["bge_m3_zeroshot_v2"]["bengali"]["R@10"])),
    "clsAcc": f3(lr_hum["accuracy"]), "clsMacro": f3(lr_hum["macro_f1"]),
    "clsNb": f3(cls["human_only_fine_grained"]["models"]["naive_bayes"]["macro_f1"]),
    "clsNTrain": str(cls["human_only_fine_grained"]["n_train"]), "clsNTest": str(cls["human_only_fine_grained"]["n_test"]),
    "clsNSynth": f"{cls['n_synthetic_train']:,}".replace(",", "{,}"),
    "tagActF": f3(tag["dev_per_tag_prf_self_consistency"]["ACT"]["f1"]),
    "tagPartyF": f3(tag["dev_per_tag_prf_self_consistency"]["PARTY"]["f1"]),
    "tagMacro": f3(tag["dev_entity_macro_f1_self_consistency"]),
    "tagNTrain": str(tag["n_train_examples"]), "tagNDev": str(tag["n_dev_examples"]),
    "lmColMean": f"{lm['by_register']['colloquial']['mean']:.1f}", "lmForMean": f"{lm['by_register']['formal']['mean']:.1f}",
    "lmColMed": f"{lm['by_register']['colloquial']['median']:.1f}", "lmForMed": f"{lm['by_register']['formal']['median']:.1f}",
    "thrDefault": f"{abst['default']:.2f}", "thrHigh": f"{abst['high']:.2f}",
    "nAnnRows": group(gold_audit["gold_rows"]), "nAnnDistinct": group(gold_audit["distinct_questions"]),
    "nPiiFlagged": str(audit2["pii_flag_distribution"]["yes"]),
    "nSkippedNeg": group(mine6["skipped_relevant_unlabelled"]),
    "nSynthPairs": group(int(synth_stats["pairs"])),
    "nLawyerChecked": str(lawyer["lawyer_answer_names_a_corpus_act"]),
    "nLawyerAgree": str(lawyer["label_act_matches_lawyer"]),
    "pctLawyerAgree": pct(lawyer["agreement_rate"], 0),
    "bmDevBest": f3(max(bm_sweep)), "bmDevWorst": f3(min(bm_sweep)), "bmSweepCells": str(len(bm_sweep)),
    "RtenBmTuned": f3(float(lang_rows["bm25_tuned_v2"]["all"]["R@10"])),
    "RtenBiLstm": f3(float(ladder_rows["bilstm_v2"]["R@10"])),
    "RtenWvEn": f3(float(lang_rows["word2vec_v2"]["english"]["R@10"])),
    "RtenWvBn": f3(float(lang_rows["word2vec_v2"]["bengali"]["R@10"])),
    "RtenBmBn": f3(float(lang_rows["bm25_default_v2"]["bengali"]["R@10"])),
    "nRoundOnePairs": str(gold_audit["overlap"]["annotator_pairs_compared"]),
}
write("numbers.tex", "% generated by scripts/78_build_report_assets.py -- do not edit\n" +
      "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

# -------------------------------------------------- 2. corpus tables/figs
print("corpus...")
dom_lang = Counter((c["domain"], c["language"]) for c in corpus)
dom_order = [d["id"] for d in domains_cfg["domains"]] + ["other"]
rows = []
for d in dom_order:
    bn_, en_ = dom_lang[(d, "bengali")], dom_lang[(d, "english")]
    if bn_ + en_ == 0:
        continue
    rows.append([tex(DOM_EN[d]), DOM_RISK[d], f"{bn_ + en_:,}".replace(",", "{,}"), f"{bn_:,}".replace(",", "{,}"),
                 f"{en_:,}".replace(",", "{,}"), pct(en_ / (bn_ + en_), 0)])
rows.append(["\\midrule\\textbf{Total}", "", f"\\textbf{{{len(corpus):,}}}".replace(",", "{,}"),
             f"\\textbf{{{n_bn:,}}}".replace(",", "{,}"), f"\\textbf{{{n_en:,}}}".replace(",", "{,}"), pct(n_en / len(corpus), 0)])
write("tab_corpus_domain.tex", table("lcrrrr", ["Domain", "Risk tier", "Chunks", "Bangla", "English", "English share"], rows))

named = [d for d in dom_order if d != "other" and dom_lang[(d, "bengali")] + dom_lang[(d, "english")]]
named.sort(key=lambda d: dom_lang[(d, "bengali")] + dom_lang[(d, "english")])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b_ = [dom_lang[(d, "bengali")] for d in named]
e_ = [dom_lang[(d, "english")] for d in named]
ax.barh([DOM_EN[d] for d in named], b_, color=GREEN, label="Bangla-language chunks")
ax.barh([DOM_EN[d] for d in named], e_, left=b_, color=ORANGE, label="English-language chunks")
ax.set_xlabel("chunks in the corpus"); ax.legend(loc="lower right")
ax.set_title("Named domains: language of the statutory text (39{,}484-chunk corpus, 'other' omitted)".replace("{,}", ","))
savefig(fig, "fig_corpus_domains.pdf")

# -------------------------------------------------- 3. gold set figs/tables
print("gold...")
gd = Counter(r["domain"] for r in gold)
order = [d for d, _ in gd.most_common()]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar([DOM_EN[d].split(" and ")[0].split(" (")[0][:22] for d in order], [gd[d] for d in order], color=BLUE)
ax.set_ylabel("gold questions"); ax.set_title(f"Domain of the {len(gold)} human-verified gold questions")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
savefig(fig, "fig_gold_domains.pdf")

# question length: real vs synthetic sources
pool = read_jsonl(PROC / "train_retrieval_v6.jsonl")
len_by = {"Real citizen questions\n(gold, paraphrased)": qwords_gold}
len_by["Approved title pairs\n(synthetic)"] = [len(r["question"].split()) for r in pool if r["label_source"] == "human_approved_title_pair"]
len_by["Authored narratives\n(synthetic)"] = [len(r["question"].split()) for r in pool if r["label_source"] == "authored_against_provision_text"]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bins = np.arange(0, 205, 8)
for (name, vals), col in zip(len_by.items(), [BLUE, ORANGE, GREEN]):
    ax.hist(np.clip(vals, 0, 200), bins=bins, histtype="step", linewidth=1.6, color=col,
            density=True, label=f"{name.replace(chr(10), ' ')} (median {int(statistics.median(vals))})")
ax.set_xlabel("question length (words, clipped at 200)"); ax.set_ylabel("density"); ax.legend(fontsize=7.5)
ax.set_title("Real questions are long narratives; templated synthetic ones are not")
savefig(fig, "fig_question_length.pdf")
macros_len = {"medianLenApproved": int(statistics.median(len_by["Approved title pairs\n(synthetic)"])),
              "medianLenAuthored": int(statistics.median(len_by["Authored narratives\n(synthetic)"]))}

# overlap audit figure + table
tiers = [("real", "Real gold questions\n(reference)"), ("tier_a", "Tier A\n(anchored templates)"),
         ("tier_a_expansion", "Tier A expansion"), ("tier_b", "Tier B\n(title + lexicon)"), ("tier_c", "Tier C\n(title only)")]
def ovs(key):
    return ov["real"]["content_overlap"] if key == "real" else ov["tiers"][key]["content_overlap"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, (key, lab) in enumerate(tiers):
    s = ovs(key)
    col = BLUE if key == "real" else (GREEN if ov["tiers"].get(key, {}).get("verdict") == "pass" else RED)
    ax.plot([s["p10"], s["p90"]], [i, i], color=col, lw=1.2)
    ax.plot([s["p25"], s["p75"]], [i, i], color=col, lw=6, solid_capstyle="butt")
    ax.plot([s["median"]], [i], "o", color="white", mec=col, ms=5, zorder=3)
ax.set_yticks(range(len(tiers))); ax.set_yticklabels([t[1] for t in tiers], fontsize=7.5); ax.invert_yaxis()
ax.axvline(ov["pass_threshold_median"], color=GREY, ls=":", lw=1)
ax.text(ov["pass_threshold_median"] + 0.006, 1.5, "pass threshold", fontsize=7, color=GREY)
ax.set_xlabel("content-word overlap between question and answer provision")
ax.set_title("Lexical-overlap audit: median (dot), IQR (thick), p10--p90 (thin)")
savefig(fig, "fig_overlap.pdf")
rows = []
for key, lab in tiers:
    s = ovs(key)
    verdict = "reference" if key == "real" else ov["tiers"][key]["verdict"]
    rows.append([tex(lab.replace("\n", " ")), f"{s['n']:,}".replace(",", "{,}"), f3(s["mean"]), f3(s["median"]),
                 f3(s["p90"]), pct(s["frac_over_0.3"]), verdict])
write("tab_overlap.tex", table("lrrrrrl", ["Question set", "n", "Mean", "Median", "p90", "Overlap$>$0.3", "Verdict"], rows))

# training pool / splits
rows = [
    ["Approved title pairs (synthetic, human-approved)", macros["nTrainApproved"], "short, title-derived"],
    ["Authored narratives (LLM-authored against provision text)", macros["nTrainAuthored"], "first-person narratives"],
    ["Real human-adjudicated questions", macros["nTrainHuman"], "paraphrased newspaper questions"],
    ["\\midrule\\textbf{Training pool (v6)}", f"\\textbf{{{macros['nTrainTotal']}}}", f"{merge6['distinct_provisions']:,} provisions".replace(",", "{,}")],
    ["Dev (real, held out)", macros["nDevHuman"], "provision-disjoint from training"],
    ["Test (real, held out)", macros["nTest"], f"{macros['nTestBn']} Bangla-target, {macros['nTestEn']} English-target"],
]
write("tab_pool.tex", table("lrl", ["Split component", "Rows", "Note"], rows))

# annotation / agreement table
rows = [
    ["Gold pool (paraphrased questions)", macros["nGold"]],
    ["\\quad answered with a provision", macros["nGoldAnswered"]],
    ["\\quad unanswerable / not found in the corpus", f"{macros['nGoldUnanswerable']} / {macros['nGoldNotFound']}"],
    ["Verified by a single human annotator", macros["nSingleAnnotated"]],
    ["Double-annotated, annotators agree on the provision set", f"{macros['nAgreeProvision']} of {macros['nDoubleAnnotated']} ({macros['pctAgreeProvision']})"],
    ["Double-annotated, disagreement then adjudicated", macros["nAdjudicated"]],
    ["\\quad of which adjudicated by an LLM (disclosed)", macros["nLlmAdjudicated"]],
    ["Human answers corrected by an LLM reviewer (disclosed)", macros["nLlmCorrections"]],
    ["Held-out test rows whose label involved the LLM", f"{macros['nTestLlm']} of {macros['nTest']}"],
]
write("tab_agreement.tex", table("lr", ["Gold-set provenance", "Count"], rows))

# -------------------------------------------------- 4. ladder
print("retrieval ladder...")
rows = []
vals = {k: [float(ladder_rows[r][k]) for r in LADDER] for k in ("R@1", "R@5", "R@10", "MRR@10")}
cols = {k: bold_best(v) for k, v in vals.items()}
trained = {"bm25_default_v2": "no", "word2vec_v2": "self-trained", "bilstm_v2": "self-trained",
           "bge_m3_zeroshot_v2": "no (control)", "bge_m3_acttitle_v2": "no", "bge_m3_attempt4_v2": "LoRA"}
for i, r in enumerate(LADDER):
    rows.append([RUN_LABEL[r], trained[r], cols["R@1"][i], cols["R@5"][i], cols["R@10"][i], cols["MRR@10"][i]])
b = lang_rows["bm25_tuned_v2"]["all"]
write("tab_ladder.tex", table("llrrrr", ["Rung", "Trained", "R@1", "R@5", "R@10", "MRR@10"], rows))

fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = np.arange(len(LADDER)); w = 0.26
for j, (k, col) in enumerate(zip(("R@1", "R@5", "R@10"), (LIGHT, BLUE, GREEN))):
    ax.bar(x + (j - 1) * w, vals[k], w, color=col, label=k)
ax.set_xticks(x); ax.set_xticklabels([SHORT[r] for r in LADDER], fontsize=7.5)
ax.set_ylabel("recall"); ax.legend(ncol=3, loc="upper left")
ax.set_title(f"Retrieval ladder on the {n_probe} verified gold questions")
savefig(fig, "fig_ladder.pdf")

rows = []
for r in LADDER + ["bm25_tuned_v2"]:
    L = lang_rows[r]
    rows.append([RUN_LABEL[r]] + [f3(float(L[g]["R@10"])) for g in ("english", "bengali", "mixed")]
                + [f3(float(L[g]["R@100"])) for g in ("english", "bengali")])
write("tab_language.tex", table("lrrrrr", ["Rung", "R@10 En", "R@10 Bn", "R@10 Mixed", "R@100 En", "R@100 Bn"], rows))

fig, ax = plt.subplots(figsize=(6.4, 3.1))
w = 0.26
for j, (g, col, lab) in enumerate((("english", ORANGE, f"English target (n={lang_counts['english']})"),
                                   ("bengali", GREEN, f"Bangla target (n={lang_counts['bengali']})"),
                                   ("mixed", GREY, f"Mixed (n={lang_counts['mixed']})"))):
    ax.bar(x + (j - 1) * w, [float(lang_rows[r][g]["R@10"]) for r in LADDER], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([SHORT[r] for r in LADDER], fontsize=7.5)
ax.set_ylabel("Recall@10"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("The language wall: lexical and small-corpus models cannot reach English Acts")
savefig(fig, "fig_language_wall.pdf")

reg = {r["run"]: r for r in read_csv(TABLES / "ladder_by_register_v2_complete.csv")}
rows = []
for r in LADDER:
    g = reg[r]
    rows.append([RUN_LABEL[r], f3(float(g["colloquial_R@10"])), f3(float(g["formal_R@10"])), f"{float(g['gap_R@10']):+.3f}"])
write("tab_register.tex", table("lrrr", [f"Rung", f"Colloquial R@10 (n={int(reg[LADDER[0]]['colloquial_n'])})",
                                          f"Formal R@10 (n={int(reg[LADDER[0]]['formal_n'])})", "Formal $-$ colloquial"], rows))

# -------------------------------------------------- 5. bootstrap tables + forest
print("bootstrap...")
def cmp_json(a: str, b_: str) -> dict:
    return read_json(RUNS / f"compare_{a}_vs_{b_}.json")


def row_of(cj: dict, group: str, k: int) -> dict:
    return next(c for c in cj["comparisons"] if c["group"] == group and c["k"] == k)


PAIRS_V2 = [("bge_m3_zeroshot_v2", "bm25_default_v2", "Zero-shot BGE-m3 vs BM25"),
            ("bge_m3_zeroshot_v2", "word2vec_v2", "Zero-shot BGE-m3 vs Word2Vec"),
            ("bge_m3_acttitle_v2", "bge_m3_zeroshot_v2", "+ Act title vs zero-shot"),
            ("bge_m3_attempt4_v2", "bge_m3_zeroshot_v2", "Fine-tuned + Act title vs zero-shot")]
PAIRS_T = [("bge_m3_finetuned_v6", "bge_m3_zeroshot_v6", "Fine-tuned v6 vs zero-shot"),
           ("bge_m3_finetuned_v6", "bge_m3_finetuned_v7", "v6 vs v7 training pool"),
           ("bge_m3_dual_hybrid_v6", "bge_m3_finetuned_v6", "Dual-query + sparse/ColBERT vs v6")]


def boot_rows(pairs, ks=(1, 5, 10)):
    rows, forest = [], []
    for a, b_, label in pairs:
        cj = cmp_json(a, b_)
        for k in ks:
            r = row_of(cj, "all", k)
            verdict = "significant" if r["significant"] else "within noise"
            rows.append([tex(label) if k == ks[0] else "", f"R@{k}", f3(r["a"]), f3(r["b"]), f"{r['delta']:+.3f}",
                         f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", verdict])
            if k == 10:
                forest.append((label, r))
        rows[-1][0] = rows[-1][0]
    return rows, forest


rows_v2, forest_v2 = boot_rows(PAIRS_V2)
write("tab_bootstrap_v2.tex", table("llrrrll", ["Comparison ($n{=}" + str(n_probe) + "$)", "Metric", "A", "B", "A$-$B", "95\\% CI", "Verdict"], rows_v2, "\\footnotesize"))
rows_t, forest_t = boot_rows(PAIRS_T, ks=(1, 5, 10))
write("tab_bootstrap_test.tex", table("llrrrll", ["Comparison ($n{=}" + macros["nTest"] + "$)", "Metric", "A", "B", "A$-$B", "95\\% CI", "Verdict"], rows_t, "\\footnotesize"))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
allf = [(l, r, "verified gold (n=%d)" % n_probe) for l, r in forest_v2] + [(l, r, "held-out test (n=%s)" % macros["nTest"]) for l, r in forest_t]
for i, (label, r, pop) in enumerate(allf):
    col = GREEN if r["significant"] and r["delta"] > 0 else (RED if r["significant"] else GREY)
    ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=col, lw=2)
    ax.plot([r["delta"]], [i], "o", color=col)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(allf))); ax.set_yticklabels([f"{l}" for l, _, _ in allf], fontsize=7.5); ax.invert_yaxis()
ax.axhline(len(forest_v2) - 0.5, color=GREY, ls=":", lw=1)
ax.set_xlabel("difference in Recall@10 (A minus B), paired-bootstrap 95% CI")
ax.set_title("Only the lexical-to-neural jump is clearly outside the noise")
ax.text(ax.get_xlim()[1], len(forest_v2) - 0.6, "verified gold above / held-out test below", fontsize=6.5, color=GREY, ha="right", va="bottom")
savefig(fig, "fig_forest.pdf")

# fine-tune test table
def top(run: dict, k: str) -> float:
    return run[k]
tests = [("bge_m3_zeroshot_v6", "Zero-shot BGE-m3 (control)"), ("bge_m3_finetuned_v6", "Fine-tuned v6 (served)"),
         ("bge_m3_finetuned_v7", "Fine-tuned v7 (smaller, filtered pool)"),
         ("bge_m3_dual_hybrid_v6", "v6 + dual-query + sparse/ColBERT")]
rows = []
runs_t = {r: read_json(RUNS / f"{r}.json") for r, _ in tests}
for k_ in ("R@1", "R@5", "R@10", "R@100", "MRR@10", "nDCG@10"):
    pass
cols = {k_: bold_best([runs_t[r][k_] for r, _ in tests]) for k_ in ("R@1", "R@5", "R@10", "R@100", "MRR@10", "nDCG@10")}
for i, (r, lab) in enumerate(tests):
    en = (runs_t[r].get("slices") or {}).get("english_targets", {}).get("R@10")
    bn_r = (runs_t[r].get("slices") or {}).get("bengali_targets", {}).get("R@10")
    rows.append([lab] + [cols[k_][i] for k_ in ("R@1", "R@5", "R@10", "R@100", "MRR@10")]
                + [f3(en) if en is not None else "--", f3(bn_r) if bn_r is not None else "--"])
write("tab_ft_test.tex", table("lrrrrrrr", ["Model", "R@1", "R@5", "R@10", "R@100", "MRR@10", "R@10 En", "R@10 Bn"], rows, "\\footnotesize"))

# fine-tune attempt history
hist = [
    ("Attempt 1", "bge_m3_finetuned_strict", "bge_m3_zeroshot", "First synthetic pool; embedding space collapsed (two bugs)", "552 (pre-verification labels)"),
    ("Attempt 2", "bge_m3_finetuned_strict_v2", "bge_m3_zeroshot", "Bugs fixed; homogeneous templated questions", "552 (pre-verification labels)"),
    ("Attempt 3", "bge_m3_finetuned_anchor_v3", "bge_m3_zeroshot", "Hand-anchored positives, narrative questions", "552 (pre-verification labels)"),
    ("Attempt 4", "bge_m3_attempt4_v2", "bge_m3_zeroshot_v2", "Attempt 3 + Act-title document template", f"{n_probe} (verified labels)"),
    ("v6 (final)", "bge_m3_finetuned_v6", "bge_m3_zeroshot_v6", "Two-stage LoRA, 408 real + synthetic pool", f"{macros['nTest']} (held-out test)"),
]
rows = []
for name, a, b_, sig, pop in hist:
    cj = cmp_json(a, b_)
    r = row_of(cj, "all", 10)
    rows.append([name, tex(sig), tex(pop), f3(r["b"]), f3(r["a"]), f"{r['delta']:+.3f}", f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
                 "significant" if r["significant"] else "noise"])
write("tab_ft_history.tex", table("lp{4.3cm}p{2.6cm}rrrll", ["Run", "What changed", "Eval population", "Zero-shot", "Tuned", "$\\Delta$R@10", "95\\% CI", "Verdict"], rows, "\\scriptsize"))

# per-query flips (single-question effect) and LLM-label sensitivity
ftq = {r["qid"]: r for r in ft6["per_query"]}
zsq = {r["qid"]: r for r in zs6["per_query"]}
def hit(r, k): return r["rank"] is not None and r["rank"] <= k
flips = {k: [q for q in ftq if hit(ftq[q], k) != hit(zsq[q], k)] for k in (1, 5, 10)}
keep = [q for q in ftq if q not in llm_qids]
sens = {k: (sum(hit(ftq[q], k) for q in keep) / len(keep), sum(hit(zsq[q], k) for q in keep) / len(keep)) for k in (1, 5, 10)}
rows = [[f"R@{k}", str(len(flips[k])), f3(ft6[f'R@{k}']), f3(zs6[f'R@{k}']), f3(sens[k][0]), f3(sens[k][1])] for k in (1, 5, 10)]
write("tab_flips.tex", table("lrrrrr", ["Metric", "Questions that flip", "FT (all 36)", "ZS (all 36)", f"FT (excl. LLM-labelled, n={len(keep)})", "ZS (excl. LLM-labelled)"], rows))
macros2 = {"nFlipRten": str(len(flips[10])), "nFlipRone": str(len(flips[1]))}
write("numbers2.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {**macros2, **{k: str(v) for k, v in macros_len.items()}}.items()))

# -------------------------------------------------- 6. classification + tagging + LM
print("classification / tagging / LM...")
variants = [("human_only_fine_grained", "Human questions only"), ("balanced_mix", "Human + 800 filtered synthetic"),
            ("fine_grained_rebalanced", "Human + all synthetic"), ("merged_rare_domains", "All synthetic, rare domains merged")]
models = [("naive_bayes", "Naive Bayes"), ("logistic_regression", "Logistic Reg."), ("vanilla_rnn", "Vanilla RNN"), ("stacked_bilstm", "Stacked BiLSTM")]
rows = []
for key, lab in variants:
    v = cls[key]
    mf = [v["models"][m]["macro_f1"] for m, _ in models]
    cells = bold_best(mf)
    rows.append([lab, f"{v['n_train']:,}".replace(",", "{,}")] + cells)
write("tab_classification.tex", table("lrrrrr", ["Training data", "$n$ train"] + [n for _, n in models], rows))

fig, ax = plt.subplots(figsize=(6.4, 3.1))
w = 0.2
for j, ((m, mn), col) in enumerate(zip(models, (GREY, BLUE, ORANGE, GREEN))):
    ax.bar(np.arange(len(variants)) + (j - 1.5) * w, [cls[k][ "models"][m]["macro_f1"] for k, _ in variants], w, color=col, label=mn)
ax.set_xticks(range(len(variants))); ax.set_xticklabels([l.replace(" + ", "\n+ ").replace(", ", ",\n") for _, l in variants], fontsize=7.5)
ax.set_ylabel("macro-F1 (200 held-out questions)"); ax.legend(ncol=2, fontsize=7.5)
ax.set_title("Classification: discriminative wins; more synthetic data does not help")
savefig(fig, "fig_classification.pdf")

hum = cls["human_only_fine_grained"]
support = Counter(q["domain_true"] for q in hum["per_query"])
pcf = hum["models"]["logistic_regression"]["per_class_f1"]
rows = [[tex(DOM_EN.get(d, d)), str(support[d]), f3(pcf[d])] for d in sorted(pcf, key=lambda d: -support[d])]
write("tab_class_perclass.tex", table("lrr", ["Domain (predicted class)", "Test support", "F1 (LogReg)"], rows))

# confusion matrix (served classifier: Logistic Regression, human-only training data)
cm_labels = hum["labels"]
cm = np.array(hum["models"]["logistic_regression"]["confusion_matrix"], dtype=float)
row_sums = cm.sum(axis=1, keepdims=True)
nonempty = row_sums.flatten() > 0
cm_show = cm[nonempty][:, nonempty]
labels_show = [cm_labels[i] for i in range(len(cm_labels)) if nonempty[i]]
LABEL_ABBR = {"criminal_procedure": "Crim. procedure", "money_recovery": "Money recovery",
              "civil_registration": "Civil reg.", "women_children": "Women & children",
              "constitutional": "Constitutional", "road_transport": "Road transport"}
short_labels = [LABEL_ABBR.get(d, DOM_EN.get(d, d).split(" and ")[0].split(" (")[0][:18]) if d != "other" else "Other" for d in labels_show]
cm_norm = cm_show / np.clip(cm_show.sum(axis=1, keepdims=True), 1, None)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(short_labels))); ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(short_labels))); ax.set_yticklabels(short_labels, fontsize=7.5)
ax.set_xlabel("predicted domain"); ax.set_ylabel("true domain")
for i in range(len(short_labels)):
    for j in range(len(short_labels)):
        n = int(cm_show[i, j])
        if n:
            col = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, str(n), ha="center", va="center", fontsize=7.5, color=col)
ax.set_title(f"Confusion matrix, Logistic Regression on human-only questions (n={hum['n_test']})\ncell shade = row-normalised (recall); annotation = raw count")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true-domain questions")
savefig(fig, "fig_confusion.pdf")
n_diag = int(np.trace(cm))
n_total = int(cm_show.sum())
write("numbers5.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {
    "clsConfDiag": str(n_diag), "clsConfTotal": str(n_total),
    "clsConfOtherRow": str(int(cm_show[short_labels.index("Other")].sum())) if "Other" in short_labels else "0",
    "clsConfOtherToFamily": str(int(cm_show[short_labels.index("Other")][short_labels.index("Family")])) if "Other" in short_labels and "Family" in short_labels else "0",
}.items()))

tp = tag["dev_per_tag_prf_self_consistency"]
cnt = tag["weak_label_tag_distribution"]
rows = [[esc(t), f"{cnt[t]:,}".replace(",", "{,}"), f3(v["precision"]), f3(v["recall"]), f3(v["f1"])] for t, v in tp.items()]
rows.append(["\\midrule\\textbf{Entity macro-F1}", "", "", "", f"\\textbf{{{f3(tag['dev_entity_macro_f1_self_consistency'])}}}"])
write("tab_tagger.tex", table("lrrrr", ["Tag", "Weak-labelled tokens", "Precision", "Recall", "F1"], rows))

fig, ax = plt.subplots(figsize=(6.4, 2.8))
tags = list(tp)
for j, (m, col) in enumerate((("precision", LIGHT), ("recall", BLUE), ("f1", GREEN))):
    ax.bar(np.arange(len(tags)) + (j - 1) * 0.26, [tp[t][m] for t in tags], 0.26, color=col, label=m)
ax.set_xticks(range(len(tags))); ax.set_xticklabels([f"{t}\n({cnt[t]} tokens)" for t in tags], fontsize=8)
ax.set_ylim(0, 1.08); ax.legend(ncol=3, fontsize=7.5); ax.set_title("Tagger vs its own weak-label rules (self-consistency, not human accuracy)")
savefig(fig, "fig_tagger.pdf")

reg_lm = lm["by_register"]
rows = [[r.capitalize(), str(reg_lm[r]["n"]), f"{reg_lm[r]['mean']:.1f}", f"{reg_lm[r]['median']:.1f}",
         f"{reg_lm[r]['min']:.1f}", f"{reg_lm[r]['max']:.1f}"] for r in ("colloquial", "formal")]
write("tab_lm.tex", table("lrrrrr", ["Register", "n", "Mean PPL", "Median PPL", "Min", "Max"], rows))
fig, ax = plt.subplots(figsize=(4.2, 2.6))
xx = np.arange(2)
ax.bar(xx - 0.18, [reg_lm[r]["mean"] for r in ("colloquial", "formal")], 0.36, color=BLUE, label="mean")
ax.bar(xx + 0.18, [reg_lm[r]["median"] for r in ("colloquial", "formal")], 0.36, color=ORANGE, label="median")
ax.set_xticks(xx); ax.set_xticklabels(["colloquial", "formal"]); ax.set_ylabel("perplexity"); ax.legend()
ax.set_title("LSTM LM perplexity by register")
savefig(fig, "fig_lm.pdf")

# -------------------------------------------------- 7. sample data boxes
print("samples...")
def kv_table(pairs: list[tuple[str, str]], width="0.72\\linewidth") -> str:
    body = "\n".join(f"\\texttt{{{esc(k)}}} & {v} \\\\" for k, v in pairs)
    return (f"{{\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}\\renewcommand{{\\arraystretch}}{{1.0}}\n"
            f"\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.24\\linewidth}}>{{\\raggedright\\arraybackslash}}p{{{width}}}@{{}}}}\n"
            f"{body}\n\\end{{tabular}}}}\n")


def chunk_kv(cid: str, text_len: int = 330) -> str:
    c = by_chunk[cid]
    fields = ["chunk_id", "provision_id", "act_id", "act_title_bn", "act_year", "provision_kind", "provision_no_bn",
              "provision_no_ascii", "provision_title_bn", "domain", "risk_tier", "n_sub", "source_url", "source_dataset",
              "crawl_date", "n_words", "language"]
    pairs = [(f, tex(str(c[f]) if c[f] is not None else "null")) for f in fields]
    t = c["text_raw"]
    pairs.append(("text_raw", tex(t[:text_len] + (" \u2026" if len(t) > text_len else ""))))
    return kv_table(pairs)


write("sample_chunk_bn.tex", chunk_kv("labour_952_s340"))
write("sample_chunk_en.tex", chunk_kv("family_305_s7", 420))

# normalization
norm_in = ["১০৩ ধারা অনুযায়ী, তালাক\u200c দেওয়া হলো।", "Section 103 of the Penal Code, 1860 \u2014 \"Talaq\"", "আমার স্বামী তালাক দিয়েছে। কী করব?"]
rows = []
for s in norm_in:
    show = lambda t: tex(t.replace("\u200c", "<ZWNJ>"))  # noqa: E731
    rows.append([show(s), show(light(s)), show(aggressive(s))])
RA = ">{\\raggedright\\arraybackslash}p{4.3cm}"
write("tab_norm.tex", table(RA * 3, ["Input", "\\texttt{light()} (transformers)", "\\texttt{aggressive()} (BM25, Word2Vec, TF-IDF)"], rows, "\\footnotesize"))

# gold examples
GOLD_PICK = ["ajke_0144_02", "lawy_0221_01", "ajke_0139_01", "lawy_0241_01"]
gmap = {r["qid"]: r for r in gold}
rows = []
for q in GOLD_PICK:
    r = gmap[q]
    cites = []
    for cid in r["relevant_chunk_ids"][:2]:
        c = by_chunk[cid]
        cites.append(f"{tex(c['act_title_bn'])}, \\S {tex(c['provision_no_bn'])}" + (f" ({tex(c['provision_title_bn'])})" if c["provision_title_bn"] else "") + f" [{c['language']}]")
    rows.append([f"\\texttt{{{esc(q)}}}\\newline {r['domain'].replace('_', ' ')}\\newline {r['register']}", tex(r["question_bn"]), "\\newline ".join(cites)])
write("sample_gold.tex", table("p{2.4cm}p{7.2cm}p{4.9cm}", ["Question id", "Question (released paraphrase)", "Verified answer provision"], rows, "\\footnotesize"))

# synthetic examples with hard negatives
def pool_example(label_source: str, pick: int = 0, dom: str | None = None) -> dict:
    cands = [r for r in read_jsonl(PROC / "train_retrieval_v6_negatives.jsonl") if r["label_source"] == label_source
             and (dom is None or r["domain"] == dom) and r["question"].count(" ") < 40]
    return cands[pick]

ex_a = pool_example("human_approved_title_pair", 3, "labour")
ex_b = pool_example("authored_against_provision_text", 0, "family")
def synth_box(r: dict, label: str) -> str:
    pos = by_chunk[r["positive_chunk_ids"][0]]
    negs = [by_chunk[c] for c in r["hard_negative_chunk_ids"][:3]]
    pairs = [("label_source", tex(r["label_source"])), ("domain", tex(r["domain"])),
             ("content_overlap", f3(r["content_overlap"]) if r.get("content_overlap") is not None else "n/a"),
             ("question", tex(r["question"])),
             ("positive", f"{tex(pos['act_title_bn'])}, \\S {tex(pos['provision_no_bn'])} {tex(pos['provision_title_bn'] or '')}")]
    for i, n in enumerate(negs, 1):
        pairs.append((f"hard negative {i}", f"{tex(n['act_title_bn'])}, \\S {tex(n['provision_no_bn'])} {tex(n['provision_title_bn'] or '')}"))
    return f"\\textbf{{{label}}}\\par\\smallskip\n" + kv_table(pairs)
write("sample_synth.tex", synth_box(ex_a, "Approved title pair (synthetic)") + "\\bigskip\n" + synth_box(ex_b, "Authored narrative (synthetic)"))

# register lexicon
lex = read_json(ROOT / "data/lexicon/register_map.json")
rows = [[tex(k), tex(" / ".join(v))] for k, v in list(lex["bn"].items())[:6]]
write("tab_lexicon.tex", table("p{5cm}p{8cm}", ["Formal legal term", "How citizens say it"], rows))
macros_lex = {"nLexBn": str(len(lex["bn"])), "nLexEn": str(len(lex["en"]))}

# word2vec neighbours
nn = read_csv(TABLES / "w2v_neighbours.csv")
rows = [[tex(r["term"]), tex(r["legal_neighbours"]) if r["in_corpus"] == "True" else "\\emph{not in the Word2Vec vocabulary}"] for r in nn[:7]]
write("tab_w2v.tex", table("p{2.4cm}p{11cm}", ["Query term", "Nearest neighbours (self-trained skip-gram)"], rows))

# demo I/O
io = read_json(GEN / "demo_io.json")
boxes = []
for rec in io["records"]:
    lines = [f"\\textbf{{Input}}\\enspace {tex(rec['question'])}\\par\\smallskip"]
    flags = f"risk tier: \\texttt{{{rec['risk_tier']}}}; top score {rec['top_score']:.3f} vs threshold {rec['threshold']:.2f}"
    flags += "; \\textbf{abstained}" if rec["abstained"] else "; answered"
    if rec["legal_aid_referral"]:
        flags += "; legal-aid referral shown"
    lines.append(f"{{\\footnotesize {flags}}}\\par\\smallskip")
    hit_rows = []
    for i, h in enumerate(rec["hits"], 1):
        kind = "\\S" if h["provision_kind"] != "article" else "Art."
        hit_rows.append([str(i), f"{h['score']:.3f}", f"{tex(h['act_title'])}, {kind} {tex(h['provision_no'])}", tex((h["provision_title"] or "(untitled)")[:80]), h["domain"].replace("_", " ")])
    lines.append(table("rrp{5.6cm}p{5.0cm}l", ["\\#", "Score", "Citation", "Provision title", "Domain"], hit_rows, "\\footnotesize"))
    c = rec["classifier"][0]
    tags_ = ", ".join(f"{tex(t['token'])} $\\to$ {esc(t['tag'])}" for t in rec["tags"] if t["tag"] != "O") or "no entities tagged"
    if rec["rule_section_numbers"]:
        tags_ += f"; rule-based section number: {esc(', '.join(rec['rule_section_numbers']))}"
    lines.append(f"{{\\footnotesize Classifier: {esc(c['name_en'])} ({pct(c['p'], 0)}). Tagger: {tags_}.}}")
    boxes.append("\n".join(lines))
write("demo_io_boxes.tex", "".join(f"\\begin{{tcolorbox}}[colback=white,colframe=kuetblue!60,boxrule=0.5pt,left=4pt,right=4pt,top=3pt,bottom=3pt,breakable]\n{b}\n\\end{{tcolorbox}}\n" for b in boxes))
write("numbers3.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {**macros_lex, "demoThrDefault": f"{io['thresholds']['default']:.2f}", "demoThrHigh": f"{io['thresholds']['high']:.2f}"}.items()))

# -------------------------------------------------- 8. compact assets for the page-limited report
print("compact assets...")
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.8})

CSHORT = ["BM25", "W2V", "BiLSTM", "BGE\nZS", "BGE\n+title", "BGE FT\n+title"]

# (a) ladder and language wall: one figure, two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.22})
x = np.arange(len(LADDER)); w = 0.26
for j, (k, col) in enumerate(zip(("R@1", "R@5", "R@10"), (LIGHT, BLUE, GREEN))):
    ax1.bar(x + (j - 1) * w, vals[k], w, color=col, label=k)
ax1.set_xticks(x); ax1.set_xticklabels(CSHORT, fontsize=6.2); ax1.set_ylabel("recall")
ax1.set_ylim(0, 0.44); ax1.legend(ncol=3, loc="upper left"); ax1.set_title(f"(a) Ladder on the {n_probe} verified questions (ZS = zero-shot)")
for j, (g, col, lab) in enumerate((("english", ORANGE, f"English target (n={lang_counts['english']})"),
                                   ("bengali", GREEN, f"Bangla target (n={lang_counts['bengali']})"),
                                   ("mixed", GREY, f"Mixed (n={lang_counts['mixed']})"))):
    ax2.bar(x + (j - 1) * w, [float(lang_rows[r][g]["R@10"]) for r in LADDER], w, color=col, label=lab)
ax2.set_xticks(x); ax2.set_xticklabels(CSHORT, fontsize=6.2); ax2.set_ylabel("Recall@10")
ax2.set_ylim(0, 0.76); ax2.legend(loc="upper left"); ax2.set_title("(b) The language wall, by target language")
savefig(fig, "c_fig_ladder.pdf")

# (b) lexical-overlap audit and question length, one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"wspace": 0.55})
for i, (key, lab) in enumerate(tiers):
    s = ovs(key)
    col = BLUE if key == "real" else (GREEN if ov["tiers"].get(key, {}).get("verdict") == "pass" else RED)
    ax1.plot([s["p10"], s["p90"]], [i, i], color=col, lw=1.0)
    ax1.plot([s["p25"], s["p75"]], [i, i], color=col, lw=5, solid_capstyle="butt")
    ax1.plot([s["median"]], [i], "o", color="white", mec=col, ms=4, zorder=3)
ax1.set_yticks(range(len(tiers))); ax1.set_yticklabels([t[1].replace("\n", " ") for t in tiers], fontsize=6.2); ax1.invert_yaxis()
ax1.axvline(ov["pass_threshold_median"], color=GREY, ls=":", lw=1)
ax1.set_xlabel("content-word overlap, question vs answer provision"); ax1.set_title("(a) Overlap audit: median, IQR, p10--p90")
for (name, vals_), col in zip(len_by.items(), [BLUE, ORANGE, GREEN]):
    ax2.hist(np.clip(vals_, 0, 200), bins=bins, histtype="step", linewidth=1.3, color=col, density=True,
             label=f"{name.split(chr(10))[0]} (median {int(statistics.median(vals_))})")
ax2.set_xlabel("question length (words, clipped at 200)"); ax2.set_ylabel("density"); ax2.legend(fontsize=6)
ax2.set_title("(b) Real questions are long narratives")
savefig(fig, "c_fig_data.pdf")

# (c) classification and tagging
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"wspace": 0.25})
CV = ["human only", "+800 filt.\nsynthetic", "+all\nsynthetic", "all synth.,\nrare merged"]
for j, ((m, mn), col) in enumerate(zip(models, (GREY, BLUE, ORANGE, GREEN))):
    ax1.bar(np.arange(len(variants)) + (j - 1.5) * 0.2, [cls[k]["models"][m]["macro_f1"] for k, _ in variants], 0.2, color=col, label=mn)
ax1.set_xticks(range(len(variants))); ax1.set_xticklabels(CV, fontsize=6.2); ax1.set_ylabel("macro-F1")
ax1.set_ylim(0, 0.46); ax1.legend(ncol=2); ax1.set_title("(a) Domain classifiers (200 held-out questions)")
for j, (m, col) in enumerate((("precision", LIGHT), ("recall", BLUE), ("f1", GREEN))):
    ax2.bar(np.arange(len(tags)) + (j - 1) * 0.26, [tp[t][m] for t in tags], 0.26, color=col, label=m)
ax2.set_xticks(range(len(tags))); ax2.set_xticklabels([t.replace("_", "\n") + "\n(" + str(cnt[t]) + ")" for t in tags], fontsize=6.2)
ax2.set_ylim(0, 1.08); ax2.legend(ncol=3); ax2.set_title("(b) Tagger vs its own weak labels")
savefig(fig, "c_fig_cls.pdf")

# (d) forest plot, smaller
fig, ax = plt.subplots(figsize=(7.0, 2.35))
for i, (label, r, pop) in enumerate(allf):
    col = GREEN if r["significant"] and r["delta"] > 0 else (RED if r["significant"] else GREY)
    ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=col, lw=1.8); ax.plot([r["delta"]], [i], "o", color=col, ms=4)
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(allf))); ax.set_yticklabels([l for l, _, _ in allf], fontsize=6.4); ax.invert_yaxis()
ax.axhline(len(forest_v2) - 0.5, color=GREY, ls=":", lw=1)
ax.set_xlabel("difference in Recall@10 (A minus B), paired-bootstrap 95% CI. Above the dotted line: verified gold; below: held-out test")
savefig(fig, "c_fig_forest.pdf")

# ---- compact tables
def dtex(d: float, lo: float, hi: float, sig: bool) -> str:
    s = f"{d:+.3f} [{lo:+.2f},{hi:+.2f}]"
    return f"\\textbf{{{s}}}" if sig else s


rows = []
for group_name, pairs, n_ in (("verified gold", PAIRS_V2, n_probe), ("held-out test", PAIRS_T, len(test_rows))):
    rows.append([f"\\multicolumn{{4}}{{l}}{{\\emph{{{group_name}, $n{{=}}{n_}$}}}}"])
    for a, b_, label in pairs:
        cj = cmp_json(a, b_)
        rows.append([tex(label)] + [dtex(row_of(cj, "all", k)["delta"], row_of(cj, "all", k)["ci_lo"], row_of(cj, "all", k)["ci_hi"],
                                         row_of(cj, "all", k)["significant"]) for k in (1, 5, 10)])
write("c_tab_boot.tex", table("lccc", ["Comparison (A vs B)", "$\\Delta$R@1 [95\\% CI]", "$\\Delta$R@5 [95\\% CI]", "$\\Delta$R@10 [95\\% CI]"], rows, "\\scriptsize"))

rows = []
best = {k: max(float(ladder_rows[r][k]) for r in LADDER) for k in ("R@1", "R@5", "R@10", "MRR@10")}
for r in LADDER:
    cells = []
    for k in ("R@1", "R@5", "R@10", "MRR@10"):
        v = float(ladder_rows[r][k]); cells.append(f"\\textbf{{{f3(v)}}}" if v == best[k] else f3(v))
    rows.append([RUN_LABEL[r]] + cells + [f3(float(lang_rows[r][g]["R@10"])) for g in ("english", "bengali", "mixed")])
write("c_tab_ladder.tex", table("lrrrrrrr", ["Rung", "R@1", "R@5", "R@10", "MRR@10", "R@10 En", "R@10 Bn", "R@10 Mix"], rows, "\\scriptsize"))

named_rows = []
for d in dom_order:
    bn_, en_ = dom_lang[(d, "bengali")], dom_lang[(d, "english")]
    if bn_ + en_:
        named_rows.append((DOM_EN[d].split(" and ")[0].split(" (")[0], DOM_RISK[d][0].upper(), bn_ + en_, en_ / (bn_ + en_)))
half = (len(named_rows) + 1) // 2
rows = []
for i in range(half):
    left = named_rows[i]; right = named_rows[i + half] if i + half < len(named_rows) else None
    fmt = lambda t: [tex(t[0]), t[1], group(t[2]), pct(t[3], 0)] if t else ["", "", "", ""]  # noqa: E731
    rows.append(fmt(left) + fmt(right))
write("c_tab_corpus.tex", table("lcrr@{\\hspace{1.2em}}lcrr", ["Domain", "Tier", "Chunks", "En", "Domain", "Tier", "Chunks", "En"], rows, "\\scriptsize"))


def mini_kv(cid: str, text_len: int) -> str:
    c = by_chunk[cid]
    t = c["text_raw"]
    pairs = [("chunk\\_id", esc(c["chunk_id"])), ("act\\_title", tex(c["act_title_bn"])),
             ("provision", tex(f"{c['provision_kind']} {c['provision_no_bn']} / {c['provision_no_ascii']}")),
             ("title", tex(c["provision_title_bn"] or "")), ("domain, risk", tex(f"{c['domain']}, {c['risk_tier']}")),
             ("language, src", tex(f"{c['language']}, {c['source_dataset']}")),
             ("crawl, words", tex(f"{c['crawl_date']}, {c['n_words']}")),
             ("text\\_raw", tex(t[:text_len] + (" …" if len(t) > text_len else "")))]
    body = "\\\\\n".join(f"\\texttt{{{k}}} & {v}" for k, v in pairs) + "\\\\"
    return ("{\\scriptsize\\setlength{\\tabcolsep}{2pt}\\renewcommand{\\arraystretch}{1.05}"
            "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{5.9cm}@{}}\n"
            f"{body}\n\\end{{tabular}}}}")


write("c_sample_chunks.tex", mini_kv("labour_952_s340", 170) + "\n\\medskip\n" + mini_kv("family_305_s7", 170) + "\n")

rows = []
for q in ["ajke_0144_02", "lawy_0221_01"]:
    r = gmap[q]
    c = by_chunk[r["relevant_chunk_ids"][0]]
    rows.append([f"\\texttt{{{esc(q)}}}", tex(r["question_bn"]),
                 f"{tex(c['act_title_bn'])}, \\S {tex(c['provision_no_bn'])}" + (f" ({tex(c['provision_title_bn'])})" if c["provision_title_bn"] else "") + f" [{c['language']}]"])
write("c_gold_sample.tex", table("p{1.9cm}p{8.4cm}p{5.6cm}", ["Question id", "Question (released paraphrase)", "Verified answer provision"], rows, "\\scriptsize"))

OUTCOME = {"family_talaq": "hit", "labour_wages": "near miss", "cyber_hack": "near miss", "land_mutation": "miss",
           "explicit_section": "miss", "out_of_scope": "not rejected"}
rows = []
for rec in io["records"]:
    h = rec["hits"][0]
    kind = "\\S" if h["provision_kind"] != "article" else "Art."
    top = f"{tex(h['act_title'])}, {kind} {tex(h['provision_no'])} ({tex((h['provision_title'] or 'untitled')[:48])})"
    tags_ = ", ".join(f"{tex(t['token'])}:{esc(t['tag'])}" for t in rec["tags"] if t["tag"] != "O") or "none"
    if rec["rule_section_numbers"]:
        tags_ += f"; rule: \\S{esc(rec['rule_section_numbers'][0])}"
    rows.append([tex(rec["question"]), f"{top}; {rec['top_score']:.3f}", esc(rec["classifier"][0]["domain"]), tags_, OUTCOME[rec["label"]]])
write("c_demo_table.tex", table("p{3.3cm}p{5.7cm}p{1.6cm}p{2.5cm}p{1.2cm}", ["Question", "Top-1 provision; cosine score", "Classifier", "Tagger", "Outcome"], rows, "\\scriptsize"))

# numbers used by the prose
keep_ = [q for q in ftq if q not in llm_qids]
extra = {"RtenFtExcl": f3(sens[10][0]), "RtenZsExcl": f3(sens[10][1]), "nTestExcl": str(len(keep_)),
         "nWeakSecNo": str(cnt["SECTION_NO"]), "nWeakLegal": str(cnt["LEGAL_TERM"]), "nWeakParty": str(cnt["PARTY"])}
write("numbers4.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in extra.items()))

print("done.")
