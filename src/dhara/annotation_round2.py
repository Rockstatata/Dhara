"""Round-2 annotation correction, adjudication, and release validation.

The working CSVs contain copyrighted source questions, so this module never
copies ``question_bn`` into a release artifact.  Versioned outputs use the
approved paraphrase as their public question text.

HISTORICAL NOTE (2026-09-19): this module's Sifat/Sorup-specific logic
(VERIFY_NAMES, _apply_sifat_sources) operates on the four-annotator sheet
layout that predates scripts/75_merge_annotator_identities.py. It already
ran, and its effects are already folded into gold_verified_v2/v3.jsonl. The
project is credited to two members (Sarwad, Iftiaq); Sifat/Sorup's rows
were reattributed, not dropped -- see DECISIONS.md 2026-09-19. This module
is not safe to re-run as-is against the post-merge file layout (verify_
Sifat.csv/verify_Sorup.csv no longer exist) and is kept here as a historical
record of a completed one-time correction pass, not an active script.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


VERIFY_NAMES = ("Iftiaq", "Sarwad", "Sifat", "Sorup")
SHORTENED_QIDS = {
    "prot_0039_04",
    "prot_0106_03",
    "prot_0109_03",
    "prot_0114_04",
    "ajke_0205_01",
    "prot_0102_02",
    "prot_0122_02",
}
AUTO_CHUNK_REPLACEMENTS = {"other_36_s8_p0": "other_36_s8"}
SPECIAL_ANSWERS = {"none", "unanswerable"}
VALID_SOURCES = {"candidate", "own_search", "none"}
VALID_VERDICTS = {"answered", "not_found", "unanswerable"}
PHONE_RE = re.compile(r"(?<!\d)(?:\+?88)?01\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
NID_RE = re.compile(r"(?:এনআইডি|জাতীয় পরিচয়|জাতীয় পরিচয়|NID)[^\d০-৯]{0,20}[\d০-৯]{8,17}", re.I)
SOURCE_BOILERPLATE_RE = re.compile(r"নাম(?: ও ঠিকানা)? প্রকাশে অনিচ্ছুক|ঠিকানা প্রকাশে অনিচ্ছুক")


@dataclass(frozen=True)
class ResolvedAnswer:
    raw: str
    chunk_ids: tuple[str, ...]
    provision_ids: tuple[str, ...]
    special: str | None = None


def read_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: pathlib.Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split()).casefold()


def word_count(text: str) -> int:
    normalized = normalized_text(text)
    return len(normalized.split()) if normalized else 0


def candidate_map(candidate_block: str) -> dict[str, str]:
    return dict(re.findall(r"(?m)^\s*(\d+)\.\s*<([^>]+)>", candidate_block or ""))


def answer_tokens(answer: str, candidates: str = "") -> tuple[str, ...]:
    raw = (answer or "").strip()
    special = raw.casefold()
    if special in SPECIAL_ANSWERS:
        return (special,)

    marked = re.findall(r"<([^>]+)>", raw)
    if marked:
        return tuple(dict.fromkeys(marked))

    if re.fullmatch(r"[\d,\s]+", raw):
        mapping = candidate_map(candidates)
        numbers = re.findall(r"\d+", raw)
        return tuple(dict.fromkeys(mapping[number] for number in numbers if number in mapping))

    tokens = []
    for part in re.split(r"[,\r\n]+", raw):
        token = part.strip()
        match = re.search(r"([a-z][\w-]*_\d+_s[^\s,<>]+)", token, re.I)
        if match:
            tokens.append(match.group(1))
    return tuple(dict.fromkeys(tokens))


def resolve_answer(row: dict[str, str], corpus: dict[str, dict]) -> ResolvedAnswer:
    raw = (row.get("answer") or "").strip()
    tokens = answer_tokens(raw, row.get("candidates", ""))
    if len(tokens) == 1 and tokens[0] in SPECIAL_ANSWERS:
        return ResolvedAnswer(raw=raw, chunk_ids=(), provision_ids=(), special=tokens[0])
    chunk_ids = tuple(tokens)
    provision_ids = tuple(
        dict.fromkeys(corpus[cid]["provision_id"] for cid in chunk_ids if cid in corpus)
    )
    return ResolvedAnswer(raw=raw, chunk_ids=chunk_ids, provision_ids=provision_ids)


def selected_outside_candidates(row: dict[str, str], corpus: dict[str, dict]) -> bool:
    selected = resolve_answer(row, corpus)
    if selected.special:
        return False
    candidate_ids = candidate_map(row.get("candidates", "")).values()
    candidate_provisions = {
        corpus[cid]["provision_id"] for cid in candidate_ids if cid in corpus
    }
    return any(pid not in candidate_provisions for pid in selected.provision_ids)


def answers_agree(left: ResolvedAnswer, right: ResolvedAnswer) -> bool:
    """Use the round-2 audit's any-overlap agreement definition.

    Provision labels are open multi-label sets.  Two annotators agree when they
    select at least one common parent provision; subchunks of the same section
    therefore agree.  Abstention decisions agree only when the special labels
    are identical.
    """
    if left.special or right.special:
        return bool(left.special and left.special == right.special)
    return bool(set(left.provision_ids) & set(right.provision_ids))


def validate_answer_row(row: dict[str, str], corpus: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    resolved = resolve_answer(row, corpus)
    verdict = (row.get("verdict") or "").strip()
    source = (row.get("answer_source") or "").strip()
    missing = [cid for cid in resolved.chunk_ids if cid not in corpus]
    if missing:
        errors.append(f"invalid chunk ids: {missing}")
    if verdict not in VALID_VERDICTS:
        errors.append(f"invalid verdict: {verdict!r}")
    if source not in VALID_SOURCES:
        errors.append(f"invalid answer_source: {source!r}")
    if resolved.special == "none":
        if verdict != "not_found" or source != "none":
            errors.append("none requires verdict=not_found and answer_source=none")
    elif resolved.special == "unanswerable":
        if verdict != "unanswerable" or source != "none":
            errors.append("unanswerable requires verdict=unanswerable and answer_source=none")
    else:
        if not resolved.chunk_ids:
            errors.append("answered row contains no parseable chunk ids")
        if verdict != "answered":
            errors.append("chunk answer requires verdict=answered")
        if source not in {"candidate", "own_search"}:
            errors.append("chunk answer requires candidate or own_search source")
        if source == "candidate" and selected_outside_candidates(row, corpus):
            errors.append("candidate source used for a provision outside candidate list")
    if len(resolved.chunk_ids) > 3:
        errors.append("more than three chunks selected")
    if len(resolved.provision_ids) > 3:
        errors.append("more than three provisions selected")
    return errors


def provision_details(resolved: ResolvedAnswer, corpus: dict[str, dict]) -> str:
    details = []
    for cid in resolved.chunk_ids:
        chunk = corpus.get(cid)
        if not chunk:
            details.append({"chunk_id": cid, "invalid": True})
            continue
        details.append(
            {
                "chunk_id": cid,
                "provision_id": chunk["provision_id"],
                "act_title": chunk.get("act_title_bn") or chunk.get("act_title_en"),
                "section_number": chunk.get("provision_no_bn") or chunk.get("provision_no_ascii"),
                "provision_title": chunk.get("provision_title_bn"),
                "provision_text": chunk.get("text_raw") or chunk.get("text_bn"),
            }
        )
    return json.dumps(details, ensure_ascii=False)


class Round2Finalizer:
    def __init__(self, root: pathlib.Path, decisions_path: pathlib.Path) -> None:
        self.root = root.resolve()
        self.annotation_dir = self.root / "data" / "annotation"
        self.processed_dir = self.root / "data" / "processed"
        self.results_runs = self.root / "results" / "runs"
        self.results_tables = self.root / "results" / "tables"
        self.decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        corpus_rows = read_jsonl(self.processed_dir / "corpus_v1.jsonl")
        self.corpus = {row["chunk_id"]: row for row in corpus_rows}
        self.gold_v1 = {
            row["qid"]: row for row in read_jsonl(self.processed_dir / "gold_test_v1.jsonl")
        }
        self.verify_fields: dict[str, list[str]] = {}
        self.verify_rows: dict[str, list[dict[str, str]]] = {}
        self.paraphrase_fields: dict[str, list[str]] = {}
        self.paraphrase_rows: dict[str, list[dict[str, str]]] = {}
        for name in VERIFY_NAMES:
            fields, rows = read_csv(self.annotation_dir / f"verify_{name}.csv")
            self.verify_fields[name] = fields
            self.verify_rows[name] = rows
            fields, rows = read_csv(self.annotation_dir / f"paraphrase_{name}.csv")
            self.paraphrase_fields[name] = fields
            self.paraphrase_rows[name] = rows
        self.original_verify = json.loads(json.dumps(self.verify_rows, ensure_ascii=False))
        self.original_paraphrases = json.loads(json.dumps(self.paraphrase_rows, ensure_ascii=False))
        self.changes: list[dict[str, str]] = []

    def _record_change(
        self,
        *,
        qid: str,
        file: str,
        field: str,
        old: str,
        new: str,
        reason: str,
        kind: str,
    ) -> None:
        if old == new:
            return
        self.changes.append(
            {
                "qid": qid,
                "file": file,
                "field": field,
                "old_value": old,
                "new_value": new,
                "reason": reason,
                "kind": kind,
            }
        )

    def apply_in_memory(self) -> None:
        self._apply_identifier_fixes()
        self._apply_iftiaq_corrections()
        self._apply_sifat_sources()
        self._apply_paraphrases()

    def _apply_identifier_fixes(self) -> None:
        filename = "verify_Iftiaq.csv"
        for row in self.verify_rows["Iftiaq"]:
            answer = row.get("answer", "")
            corrected = answer
            for old, new in AUTO_CHUNK_REPLACEMENTS.items():
                corrected = corrected.replace(old, new)
            if corrected != answer:
                self._record_change(
                    qid=row["qid"],
                    file=filename,
                    field="answer",
                    old=answer,
                    new=corrected,
                    reason="unambiguous nonexistent subchunk corrected to existing unsplit chunk",
                    kind="identifier",
                )
                row["answer"] = corrected

    def _apply_iftiaq_corrections(self) -> None:
        decisions = self.decisions["iftiaq_corrections"]
        by_qid = {row["qid"]: row for row in self.verify_rows["Iftiaq"]}
        if set(decisions) != {
            "prot_0108_04", "prot_0047_05", "prot_0122_01", "prot_0002_03", "ajke_0192_01"
        }:
            raise ValueError("Iftiaq legal-correction decision set is incomplete or unexpected")
        for qid, decision in decisions.items():
            row = by_qid[qid]
            for field, source_key in (
                ("answer", "answer"),
                ("verdict", "verdict"),
                ("answer_source", "answer_source"),
            ):
                old = row.get(field, "")
                new = decision[source_key]
                self._record_change(
                    qid=qid,
                    file="verify_Iftiaq.csv",
                    field=field,
                    old=old,
                    new=new,
                    reason=decision["notes"],
                    kind="legal_correction",
                )
                row[field] = new

    def _apply_sifat_sources(self) -> None:
        targeted = {"prot_0008_02", "prot_0138_02", "prot_0075_03"}
        found = set()
        for row in self.verify_rows["Sifat"]:
            if row["qid"] not in targeted:
                continue
            found.add(row["qid"])
            if row.get("answer_source") == "candidate" and selected_outside_candidates(row, self.corpus):
                old = row["answer_source"]
                row["answer_source"] = "own_search"
                self._record_change(
                    qid=row["qid"],
                    file="verify_Sifat.csv",
                    field="answer_source",
                    old=old,
                    new="own_search",
                    reason="at least one selected provision is outside the displayed candidate list",
                    kind="metadata",
                )
        if found != targeted:
            raise ValueError(f"missing targeted Sifat rows: {sorted(targeted - found)}")

    def _apply_paraphrases(self) -> None:
        sarwad = self.decisions["sarwad_paraphrases"]
        sarwad_rows = {row["qid"]: row for row in self.paraphrase_rows["Sarwad"]}
        current_exact = {
            qid
            for qid, row in sarwad_rows.items()
            if normalized_text(row.get("question_bn", ""))
            == normalized_text(row.get("paraphrase", ""))
        }
        already_applied = {
            qid
            for qid, replacement in sarwad.items()
            if qid in sarwad_rows
            and normalized_text(sarwad_rows[qid].get("paraphrase", ""))
            == normalized_text(replacement)
        }
        if current_exact | already_applied != set(sarwad):
            raise ValueError(
                "Sarwad decision set differs from the current or already-corrected sheet: "
                f"unrecognized={sorted(set(sarwad) - current_exact - already_applied)}, "
                f"unexpected_exact={sorted(current_exact - set(sarwad))}"
            )
        for qid, paraphrase in sarwad.items():
            row = sarwad_rows[qid]
            old = row.get("paraphrase", "")
            row["paraphrase"] = paraphrase
            self._record_change(
                qid=qid,
                file="paraphrase_Sarwad.csv",
                field="paraphrase",
                old=old,
                new=paraphrase,
                reason="exact copy of copyrighted source question rewritten by LLM with facts and register preserved",
                kind="paraphrase",
            )

        reviews = self.decisions["shortened_reviews"]
        if set(reviews) != SHORTENED_QIDS:
            raise ValueError("shortened-review decision set is incomplete or unexpected")
        iftiaq_rows = {row["qid"]: row for row in self.paraphrase_rows["Iftiaq"]}
        for qid, review in reviews.items():
            corrected = review.get("corrected_paraphrase", "").strip()
            if review["needs_rewrite"] == "yes":
                if not corrected:
                    raise ValueError(f"{qid}: rewrite required but corrected_paraphrase is empty")
                row = iftiaq_rows[qid]
                old = row.get("paraphrase", "")
                row["paraphrase"] = corrected
                self._record_change(
                    qid=qid,
                    file="paraphrase_Iftiaq.csv",
                    field="paraphrase",
                    old=old,
                    new=corrected,
                    reason=review["notes"],
                    kind="paraphrase_review",
                )

    def all_verify_rows(self) -> list[dict[str, str]]:
        return [row for name in VERIFY_NAMES for row in self.verify_rows[name]]

    def all_paraphrase_rows(self) -> list[dict[str, str]]:
        return [row for name in VERIFY_NAMES for row in self.paraphrase_rows[name]]

    def grouped_verifications(self) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.all_verify_rows():
            grouped[row["qid"]].append(row)
        return dict(grouped)

    def comparison(self) -> tuple[list[str], list[str]]:
        agreements: list[str] = []
        disagreements: list[str] = []
        for qid, rows in sorted(self.grouped_verifications().items()):
            if len(rows) != 2:
                continue
            left = resolve_answer(rows[0], self.corpus)
            right = resolve_answer(rows[1], self.corpus)
            (agreements if answers_agree(left, right) else disagreements).append(qid)
        return agreements, disagreements

    def build_adjudication_rows(self) -> list[dict[str, str]]:
        _agreements, disagreements = self.comparison()
        decisions = self.decisions["adjudications"]
        if set(disagreements) != set(decisions):
            raise ValueError(
                "adjudication decisions do not match post-correction disagreements: "
                f"missing={sorted(set(disagreements) - set(decisions))}, "
                f"unexpected={sorted(set(decisions) - set(disagreements))}"
            )
        rows_out: list[dict[str, str]] = []
        grouped = self.grouped_verifications()
        reviewer = self.decisions["reviewer"]
        for qid in sorted(disagreements):
            left, right = sorted(grouped[qid], key=lambda row: row["annotator"])
            left_answer = resolve_answer(left, self.corpus)
            right_answer = resolve_answer(right, self.corpus)
            decision = decisions[qid]
            final_probe = {
                "answer": decision["answer"],
                "verdict": decision["verdict"],
                "answer_source": decision["answer_source"],
                "candidates": "",
            }
            final_resolved = resolve_answer(final_probe, self.corpus)
            rows_out.append(
                {
                    "qid": qid,
                    "question_bn": left["question_bn"],
                    "annotator_a": left["annotator"],
                    "raw_answer_a": left["answer"],
                    "chunk_ids_a": ", ".join(left_answer.chunk_ids),
                    "provision_ids_a": ", ".join(left_answer.provision_ids),
                    "selection_details_a": provision_details(left_answer, self.corpus),
                    "verdict_a": left["verdict"],
                    "answer_source_a": left["answer_source"],
                    "confidence_a": left["confidence"],
                    "annotator_b": right["annotator"],
                    "raw_answer_b": right["answer"],
                    "chunk_ids_b": ", ".join(right_answer.chunk_ids),
                    "provision_ids_b": ", ".join(right_answer.provision_ids),
                    "selection_details_b": provision_details(right_answer, self.corpus),
                    "verdict_b": right["verdict"],
                    "answer_source_b": right["answer_source"],
                    "confidence_b": right["confidence"],
                    "final_answer": decision["answer"],
                    "final_chunk_ids": ", ".join(final_resolved.chunk_ids),
                    "final_provision_ids": ", ".join(final_resolved.provision_ids),
                    "final_verdict": decision["verdict"],
                    "final_answer_source": decision["answer_source"],
                    "adjudicator": reviewer,
                    "adjudication_notes": decision["notes"],
                }
            )
        return rows_out

    def correction_rows_iftiaq(self) -> list[dict[str, str]]:
        unresolved_reasons = {
            "prot_0108_04": "land_1447_s4 is nonexistent and points to the wrong Act/domain; fresh legal verification required",
            "prot_0047_05": "land_1447_s4 and land_1447_s7 are nonexistent and point to the wrong Act/domain; fresh legal verification required",
            "prot_0122_01": "land_1447_s4 is nonexistent and points to the wrong Act/domain; fresh legal verification required",
            "prot_0002_03": "land_1447_s5 is nonexistent and points to the wrong Act/domain; fresh legal verification required",
            "ajke_0192_01": "other_36_s54 is a parent provision id, not an existing chunk id; the relevant subchunk or a different remedy must be selected",
        }
        existing_path = self.annotation_dir / "corrections_Iftiaq.csv"
        existing = {}
        if existing_path.exists():
            _fields, existing_rows = read_csv(existing_path)
            existing = {row["qid"]: row for row in existing_rows}
        original = {row["qid"]: row for row in self.original_verify["Iftiaq"]}
        grouped_original: dict[str, list[dict[str, str]]] = defaultdict(list)
        for name in VERIFY_NAMES:
            for row in self.original_verify[name]:
                grouped_original[row["qid"]].append(row)
        out = []
        for qid, decision in self.decisions["iftiaq_corrections"].items():
            current = original[qid]
            second = next(
                (r for r in grouped_original[qid] if r["annotator"] != "Iftiaq"),
                None,
            )
            prior = existing.get(qid, {})
            out.append(
                {
                    "qid": qid,
                    "question_bn": prior.get("question_bn", current["question_bn"]),
                    "current_answer": prior.get("current_answer", current["answer"]),
                    "reason_for_rejection": prior.get(
                        "reason_for_rejection", unresolved_reasons[qid]
                    ),
                    "second_annotator": prior.get(
                        "second_annotator", second["annotator"] if second else ""
                    ),
                    "second_answer": prior.get(
                        "second_answer", second["answer"] if second else ""
                    ),
                    "corrected_answer": decision["answer"],
                    "corrected_verdict": decision["verdict"],
                    "corrected_source": decision["answer_source"],
                    "reviewer": self.decisions["reviewer"],
                    "notes": decision["notes"],
                }
            )
        return sorted(out, key=lambda row: row["qid"])

    def correction_rows_sarwad(self) -> list[dict[str, str]]:
        existing_path = self.annotation_dir / "corrections_paraphrase_Sarwad.csv"
        existing = {}
        if existing_path.exists():
            _fields, existing_rows = read_csv(existing_path)
            existing = {row["qid"]: row for row in existing_rows}
        original = {row["qid"]: row for row in self.original_paraphrases["Sarwad"]}
        out = []
        for qid, replacement in self.decisions["sarwad_paraphrases"].items():
            row = original[qid]
            prior = existing.get(qid, {})
            out.append(
                {
                    "qid": qid,
                    "annotator": prior.get("annotator", row["annotator"]),
                    "n_words": prior.get("n_words", row["n_words"]),
                    "question_bn": prior.get("question_bn", row["question_bn"]),
                    "paraphrase": prior.get("paraphrase", row["paraphrase"]),
                    "pii_removed": prior.get("pii_removed", row["pii_removed"]),
                    "notes": prior.get("notes", row.get("notes", "")),
                    "reason": "exact copy of copyrighted source question",
                    "corrected_paraphrase": replacement,
                    "reviewer": self.decisions["reviewer"],
                }
            )
        return sorted(out, key=lambda row: row["qid"])

    def review_rows_iftiaq(self) -> list[dict[str, str]]:
        existing_path = self.annotation_dir / "review_paraphrase_Iftiaq.csv"
        existing = {}
        if existing_path.exists():
            _fields, existing_rows = read_csv(existing_path)
            existing = {row["qid"]: row for row in existing_rows}
        original = {row["qid"]: row for row in self.original_paraphrases["Iftiaq"]}
        out = []
        for qid, review in self.decisions["shortened_reviews"].items():
            row = original[qid]
            prior = existing.get(qid, {})
            original_text = prior.get("question_bn", row["question_bn"])
            current_paraphrase = prior.get("current_paraphrase", row["paraphrase"])
            original_count = word_count(original_text)
            paraphrase_count = word_count(current_paraphrase)
            out.append(
                {
                    "qid": qid,
                    "question_bn": original_text,
                    "current_paraphrase": current_paraphrase,
                    "original_word_count": str(original_count),
                    "paraphrase_word_count": str(paraphrase_count),
                    "length_ratio": f"{paraphrase_count / max(original_count, 1):.6f}",
                    "facts_preserved": review["facts_preserved"],
                    "needs_rewrite": review["needs_rewrite"],
                    "corrected_paraphrase": review["corrected_paraphrase"],
                    "reviewer": self.decisions["reviewer"],
                    "notes": review["notes"],
                }
            )
        return sorted(out, key=lambda row: row["qid"])

    def write_support_files(self) -> None:
        correction_iftiaq = self.correction_rows_iftiaq()
        correction_sarwad = self.correction_rows_sarwad()
        review_iftiaq = self.review_rows_iftiaq()
        adjudication = self.build_adjudication_rows()
        write_csv(
            self.annotation_dir / "corrections_Iftiaq.csv",
            list(correction_iftiaq[0]),
            correction_iftiaq,
        )
        write_csv(
            self.annotation_dir / "corrections_paraphrase_Sarwad.csv",
            list(correction_sarwad[0]),
            correction_sarwad,
        )
        write_csv(
            self.annotation_dir / "review_paraphrase_Iftiaq.csv",
            list(review_iftiaq[0]),
            review_iftiaq,
        )
        write_csv(
            self.annotation_dir / "adjudication_round2.csv",
            list(adjudication[0]),
            adjudication,
        )

    def _consensus_chunks(self, rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
        resolved = [resolve_answer(row, self.corpus) for row in rows]
        common = set(resolved[0].provision_ids)
        for answer in resolved[1:]:
            common &= set(answer.provision_ids)
        ordered_provisions = [pid for pid in resolved[0].provision_ids if pid in common]
        selected_chunks = []
        for pid in ordered_provisions:
            cid = next(
                cid
                for answer in resolved
                for cid in answer.chunk_ids
                if self.corpus[cid]["provision_id"] == pid
            )
            selected_chunks.append(cid)
        return selected_chunks, ordered_provisions

    def build_final_records(self) -> list[dict]:
        paraphrases = {row["qid"]: row for row in self.all_paraphrase_rows()}
        grouped = self.grouped_verifications()
        disagreements = set(self.comparison()[1])
        records = []
        for qid in sorted(grouped):
            rows = grouped[qid]
            adjudicator = ""
            adjudication_notes = ""
            if qid in disagreements:
                decision = self.decisions["adjudications"][qid]
                probe = {
                    "answer": decision["answer"],
                    "verdict": decision["verdict"],
                    "answer_source": decision["answer_source"],
                    "candidates": "",
                }
                resolved = resolve_answer(probe, self.corpus)
                chunk_ids = list(resolved.chunk_ids)
                provision_ids = list(resolved.provision_ids)
                answer = decision["answer"]
                verdict = decision["verdict"]
                answer_source = decision["answer_source"]
                adjudicator = self.decisions["reviewer"]
                adjudication_notes = decision["notes"]
                resolution = "llm_adjudication"
            elif len(rows) == 2:
                first = resolve_answer(rows[0], self.corpus)
                if first.special:
                    chunk_ids, provision_ids = [], []
                    answer = first.special
                    verdict = rows[0]["verdict"]
                    answer_source = "none"
                else:
                    chunk_ids, provision_ids = self._consensus_chunks(rows)
                    answer = ", ".join(chunk_ids)
                    verdict = "answered"
                    answer_source = (
                        "candidate"
                        if all(row["answer_source"] == "candidate" for row in rows)
                        else "own_search"
                    )
                resolution = "annotator_agreement"
            else:
                resolved = resolve_answer(rows[0], self.corpus)
                chunk_ids = list(resolved.chunk_ids)
                provision_ids = list(resolved.provision_ids)
                answer = resolved.special or ", ".join(chunk_ids)
                verdict = rows[0]["verdict"]
                answer_source = rows[0]["answer_source"]
                resolution = "single_verified_annotation"

            public_question = paraphrases[qid]["paraphrase"].strip()
            base = self.gold_v1.get(qid, {})
            records.append(
                {
                    "qid": qid,
                    "question_bn": public_question,
                    "paraphrased": True,
                    "relevant_chunk_ids": chunk_ids,
                    "relevant_provision_ids": provision_ids,
                    "answer": answer,
                    "verdict": verdict,
                    "answerable": verdict == "answered",
                    "answer_source": answer_source,
                    "annotators": [row["annotator"] for row in rows],
                    "annotator_confidences": {
                        row["annotator"]: row["confidence"] for row in rows
                    },
                    "resolution": resolution,
                    "adjudicator": adjudicator,
                    "adjudication_notes": adjudication_notes,
                    "register": base.get("register"),
                    "domain": base.get("domain"),
                    "source": base.get("source", "mined"),
                    "annotation_mode": base.get("annotation_mode"),
                    "pii_removed": paraphrases[qid]["pii_removed"],
                }
            )
        return records

    def pii_findings(self) -> list[dict[str, str]]:
        findings = []
        for row in self.all_paraphrase_rows():
            text = row.get("paraphrase", "")
            normalized_digits = text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
            matched = []
            if PHONE_RE.search(normalized_digits):
                matched.append("phone-like sequence")
            if EMAIL_RE.search(text):
                matched.append("email address")
            if NID_RE.search(text):
                matched.append("NID-like sequence")
            if SOURCE_BOILERPLATE_RE.search(text):
                matched.append("source identity boilerplate")
            if matched:
                findings.append({"qid": row["qid"], "reason": "; ".join(matched)})
        return findings

    def audit(self) -> tuple[dict, list[dict]]:
        verify = self.all_verify_rows()
        grouped = self.grouped_verifications()
        paraphrases = self.all_paraphrase_rows()
        assignment_categories = Counter()
        invalid_rows = []
        consistency_errors = []
        for row in verify:
            resolved = resolve_answer(row, self.corpus)
            if resolved.special == "none":
                assignment_categories["none"] += 1
            elif resolved.special == "unanswerable":
                assignment_categories["unanswerable"] += 1
            elif any(cid not in self.corpus for cid in resolved.chunk_ids) or not resolved.chunk_ids:
                assignment_categories["invalid"] += 1
                invalid_rows.append(row["qid"])
            else:
                assignment_categories["valid"] += 1
            errors = validate_answer_row(row, self.corpus)
            if errors:
                consistency_errors.append({"qid": row["qid"], "annotator": row["annotator"], "errors": errors})

        expected_qids = {row["qid"] for row in paraphrases}
        actual_qids = set(grouped)
        agreements, disagreements = self.comparison()
        exact = [
            row["qid"]
            for row in paraphrases
            if normalized_text(row["question_bn"]) == normalized_text(row["paraphrase"])
        ]
        empty = [row["qid"] for row in paraphrases if not normalized_text(row["paraphrase"])]
        ratios = []
        for row in paraphrases:
            original_words = word_count(row["question_bn"])
            para_words = word_count(row["paraphrase"])
            ratios.append(
                {
                    "qid": row["qid"],
                    "original_words": original_words,
                    "paraphrase_words": para_words,
                    "length_ratio": para_words / max(original_words, 1),
                    "approved_shortening": (
                        row["qid"] in SHORTENED_QIDS
                        or row["qid"] in self.decisions["sarwad_paraphrases"]
                    ),
                }
            )
        shortened_awaiting = [
            row["qid"]
            for row in ratios
            if row["length_ratio"] < 0.5 and not row["approved_shortening"]
        ]
        pii_values = Counter(row.get("pii_removed", "") for row in paraphrases)
        pii_invalid = sorted(value for value in pii_values if value not in {"yes", "no"})
        pii_findings = self.pii_findings()
        final_records = self.build_final_records()
        final_errors = []
        for record in final_records:
            probe = {
                "answer": record["answer"],
                "verdict": record["verdict"],
                "answer_source": record["answer_source"],
                "candidates": "",
            }
            # Candidate provenance was validated on the contributing working
            # rows.  A consensus output intentionally has no candidate block.
            if probe["answer_source"] == "candidate":
                probe["answer_source"] = "own_search"
            errors = validate_answer_row(probe, self.corpus)
            if errors:
                final_errors.append({"qid": record["qid"], "errors": errors})

        summary = {
            "verification_assignments": len(verify),
            "unique_qids": len(grouped),
            "expected_qids": len(expected_qids),
            "missing_qids": sorted(expected_qids - actual_qids),
            "unexpected_qids": sorted(actual_qids - expected_qids),
            "valid_assignments": assignment_categories["valid"],
            "invalid_assignments": assignment_categories["invalid"],
            "none_assignments": assignment_categories["none"],
            "unanswerable_assignments": assignment_categories["unanswerable"],
            "double_annotated_qids": sum(len(rows) == 2 for rows in grouped.values()),
            "provision_level_agreements": len(agreements),
            "remaining_disagreements": len(disagreements),
            "remaining_disagreement_qids": disagreements,
            "adjudicated_disagreements": len(disagreements),
            "unresolved_adjudications": [],
            "unresolved_corrections": [],
            "paraphrases_filled": len(paraphrases) - len(empty),
            "empty_paraphrases": empty,
            "exact_copy_paraphrases": exact,
            "shortened_below_half": sum(row["length_ratio"] < 0.5 for row in ratios),
            "shortened_paraphrases_awaiting_approval": shortened_awaiting,
            "pii_flag_distribution": dict(sorted(pii_values.items())),
            "invalid_pii_flags": pii_invalid,
            "pii_review_findings": pii_findings,
            "working_consistency_errors": consistency_errors,
            "final_consistency_errors": final_errors,
            "final_record_count": len(final_records),
            "automatic_changes": self.changes,
            "paraphrase_length_ratios": ratios,
        }
        prior_audit_path = self.results_runs / "annotation_round2_audit.json"
        if not self.changes and prior_audit_path.exists():
            prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
            summary["automatic_changes"] = prior_audit.get("automatic_changes", [])
        metric_rows = []
        for key, value in summary.items():
            if key in {"automatic_changes", "paraphrase_length_ratios"}:
                continue
            metric_rows.append(
                {
                    "metric": key,
                    "value": json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else str(value),
                }
            )
        return summary, metric_rows

    @staticmethod
    def gates(summary: dict) -> list[str]:
        failures = []
        checks = {
            "missing qids": summary["missing_qids"],
            "unexpected qids": summary["unexpected_qids"],
            "invalid chunk answers": summary["invalid_assignments"],
            "working answer consistency errors": summary["working_consistency_errors"],
            "empty paraphrases": summary["empty_paraphrases"],
            "exact-copy paraphrases": summary["exact_copy_paraphrases"],
            "unapproved shortened paraphrases": summary["shortened_paraphrases_awaiting_approval"],
            "invalid PII flags": summary["invalid_pii_flags"],
            "PII review findings": summary["pii_review_findings"],
            "final answer consistency errors": summary["final_consistency_errors"],
            "unresolved adjudications": summary["unresolved_adjudications"],
            "unresolved corrections": summary["unresolved_corrections"],
        }
        for label, value in checks.items():
            if value:
                failures.append(f"{label}: {value}")
        if summary["verification_assignments"] != 592:
            failures.append(f"verification assignments: expected 592, got {summary['verification_assignments']}")
        if summary["unique_qids"] != 552:
            failures.append(f"unique qids: expected 552, got {summary['unique_qids']}")
        if summary["double_annotated_qids"] != 40:
            failures.append(
                f"double-annotated qids: expected 40, got {summary['double_annotated_qids']}"
            )
        if summary["remaining_disagreements"] != len(summary["remaining_disagreement_qids"]):
            failures.append("disagreement count is internally inconsistent")
        if summary["final_record_count"] != 552:
            failures.append(f"final records: expected 552, got {summary['final_record_count']}")
        return failures

    def print_report(self, summary: dict, failures: Sequence[str]) -> None:
        labels = (
            ("verification assignments", "verification_assignments"),
            ("unique qids", "unique_qids"),
            ("missing qids", "missing_qids"),
            ("unexpected qids", "unexpected_qids"),
            ("valid assignments", "valid_assignments"),
            ("invalid assignments", "invalid_assignments"),
            ("none assignments", "none_assignments"),
            ("unanswerable assignments", "unanswerable_assignments"),
            ("double-annotated qids", "double_annotated_qids"),
            ("provision-level agreements", "provision_level_agreements"),
            ("remaining disagreements", "remaining_disagreements"),
            ("adjudicated disagreements", "adjudicated_disagreements"),
            ("unresolved adjudications", "unresolved_adjudications"),
            ("unresolved corrections", "unresolved_corrections"),
            ("paraphrases filled", "paraphrases_filled"),
            ("exact-copy paraphrases", "exact_copy_paraphrases"),
            ("shortened below 0.5", "shortened_below_half"),
            ("shortened awaiting approval", "shortened_paraphrases_awaiting_approval"),
            ("PII flag distribution", "pii_flag_distribution"),
            ("PII review findings", "pii_review_findings"),
        )
        print("Dhara Round-2 dry-run report")
        print("=" * 31)
        for label, key in labels:
            print(f"{label}: {summary[key]}")
        print("automatic metadata / identifier corrections:")
        auto = [change for change in self.changes if change["kind"] in {"identifier", "metadata"}]
        for change in auto:
            print(
                f"  {change['file']} {change['qid']} {change['field']}: "
                f"{change['old_value']!r} -> {change['new_value']!r}"
            )
        print(f"all proposed changed cells: {len(self.changes)}")
        if failures:
            print("GATE: FAIL")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("GATE: PASS")

    def backup_working_files(self) -> pathlib.Path:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = (self.annotation_dir / "archive" / stamp).resolve()
        if self.annotation_dir.resolve() not in archive.parents:
            raise ValueError(f"refusing archive path outside annotation directory: {archive}")
        archive.mkdir(parents=True, exist_ok=False)
        for name in VERIFY_NAMES:
            for prefix in ("verify", "paraphrase"):
                source = self.annotation_dir / f"{prefix}_{name}.csv"
                shutil.copy2(source, archive / source.name)
        return archive

    def write_working_files(self) -> None:
        for name in VERIFY_NAMES:
            write_csv(
                self.annotation_dir / f"verify_{name}.csv",
                self.verify_fields[name],
                self.verify_rows[name],
            )
            write_csv(
                self.annotation_dir / f"paraphrase_{name}.csv",
                self.paraphrase_fields[name],
                self.paraphrase_rows[name],
            )

    def write_release_artifacts(self, summary: dict, metric_rows: list[dict]) -> None:
        records = self.build_final_records()
        write_jsonl(self.processed_dir / "gold_verified_v2.jsonl", records)
        questions = [
            {
                "qid": row["qid"],
                "question_bn": row["question_bn"],
                "paraphrased": True,
                "source": row["source"],
                "register": row["register"],
                "domain": row["domain"],
                "pii_removed": row["pii_removed"],
            }
            for row in records
        ]
        write_jsonl(self.processed_dir / "questions_release_v2.jsonl", questions)
        self.results_runs.mkdir(parents=True, exist_ok=True)
        (self.results_runs / "annotation_round2_audit.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(
            self.results_tables / "annotation_round2_audit.csv",
            ["metric", "value"],
            metric_rows,
        )
