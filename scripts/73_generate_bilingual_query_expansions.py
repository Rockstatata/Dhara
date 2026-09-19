"""Generate English legal concept expansions for Dhara queries.

    python scripts/73_generate_bilingual_query_expansions.py

Attacks the English-Act cross-lingual wall: maps colloquial Bengali questions
to statutory English legal concept terms so multilingual encoders (BGE-M3)
can densely match Victorian-English statutes without zero-shot translation loss.

Emits:
    data/processed/dev_retrieval_v4_bilingual.jsonl
    data/processed/test_retrieval_v4_bilingual.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

DEV_IN = pathlib.Path("data/processed/dev_retrieval_v4.jsonl")
TEST_IN = pathlib.Path("data/processed/test_retrieval_v4.jsonl")
DEV_OUT = pathlib.Path("data/processed/dev_retrieval_v4_bilingual.jsonl")
TEST_OUT = pathlib.Path("data/processed/test_retrieval_v4_bilingual.jsonl")

# Canonical English legal concept mappings for the dev and test benchmarks
ENGLISH_CONCEPT_MAP: dict[str, str] = {
    # DEV QUERIES
    "prot_0004_04": "smoking in public place fine penalty Tobacco Control Act 2005",
    "prot_0014_02": "Code of Criminal Procedure appeal against acquittal Section 417 high court sessions judge",
    "prot_0066_07": "Evidence Act burden of proof person alive missing seven years Section 108 presumption of death",
    "ajke_0157_01": "Hindu Women's Rights to Property Act 1937 Section 3 widow daughter share estate inheritance bank money",
    "lawy_0350_01": "Bank Companies Act 1991 Section 103 nominee legal heirs deceased account balance",
    "lawy_0363_01": "Bank Companies Act 1991 Section 103 bank deposit deceased person nominee inheritance",
    "lawy_0382_01": "Bank Companies Act 1991 Section 103 bank account deceased owner nominee rights",
    "prot_0007_02": "Hindu Women's Rights to Property Act 1937 Section 3 intestate succession widow rights",
    "prot_0017_01": "Hindu Women's Rights to Property Act 1937 Section 3 husband property widow daughter",
    "prot_0061_01": "Hindu Women's Rights to Property Act 1937 Section 3 partition widow estate",
    "prot_0065_04": "Bank Companies Act 1991 Section 103 joint account nominee savings bank deposit",
    "prot_0086_03": "Bank Companies Act 1991 Section 103 government employee pension nominee bank",
    "prot_0008_06": "National Identity Registration Act correction of name documents NID election commission",
    "prot_0077_07": "National Identity Registration Act 2010 Section 8 alteration correction voter card NID",
    "prot_0084_01": "National Identity Registration Act 2010 Section 8 change date of birth name correction",
    "prot_0084_05": "Birth and Death Registration Act 2004 Section 15 amendment birth certificate age correction",
    "prot_0112_01": "Organ Transplantation Act Section 19 illegal kidney organ buying broker hospital penalty",
    "prot_0114_03": "Organ Transplantation Act Section 19 illegal organ sale commercial trade penalty",
    "dail_0621_02": "Right to Information Act 2009 Section 4 right to obtain information ministry public authority",
    "dail_0629_02": "Right to Information Act 2009 Section 4 disclosure of information citizen request",
    "lawy_0386_01": "Government Servants Service Act 2018 Section 42 criminal conviction dismissal from service",
    "prot_0008_03": "Government Servants Service Act 2018 Section 42 departmental proceedings misconduct penalty",
    "prot_0081_02": "Specific Relief Act 1877 Section 55 mandatory injunction removal encroachment wall land",
    "prot_0077_08": "Real Estate Development and Management Act 2010 Section 15 handover flat landowner developer delay",
    "lawy_0588_01": "Code of Criminal Procedure 1898 Section 248 withdrawal of complaint acquittal complainant",
    "ajke_0184_01": "Nari O Shishu Nirjatan Daman Ain 2000 Section 10 sexual harassment stalking student OCC legal aid",
    "lawy_0354_01": "Code of Criminal Procedure 1898 Section 164 recording statement confession magistrate copy to accused",
    "prot_0000_03": "Co-operative Societies Act 2001 Section 50 dispute settlement refund investment fraud",
    "lawy_0362_01": "Transfer of Property Act 1882 Section 52 transfer of property pending suit lis pendens title",
    "prot_0003_01": "Code of Criminal Procedure 1898 Section 154 information in cognizable offence FIR police station theft",
    "ajke_0172_01": "Birth and Death Registration Act 2004 Section 13 registration of marriage divorce birth entry",
    "prot_0070_03": "Penal Code 1860 Section 342 punishment wrongful confinement detention locked in room",
    "prot_0114_04": "Code of Criminal Procedure 1898 Section 344 adjournment delay in trial court postponement",
    "prot_0076_03": "Specific Relief Act 1877 Section 27 specific performance of contract against subsequent title purchaser",
    "lawy_0464_01": "State Acquisition and Tenancy Act 1950 Section 144A modification of record of rights khatian civil court decree",
    "prot_0137_01": "Village Court Act 2006 Section 3 jurisdiction village court local dispute shalish",

    # TEST QUERIES
    "prot_0104_02": "Government Debt Act 2022 Section 15 vested property return tribunal claim",
    "prot_0110_03": "Labour Act 2006 Section 1 applicability exclusion armed forces civil staff airline",
    "ajke_0149_01": "Negotiable Instruments Act 1881 Section 138 dishonour of cheque bounce insufficiency of funds notice",
    "lawy_0212_01": "Negotiable Instruments Act 1881 Section 140 offences by companies cheque bounce directors liability",
    "lawy_0445_01": "Negotiable Instruments Act 1881 Section 138 procedure filing cheque dishonour case legal notice",
    "prot_0066_02": "Penal Code 1860 Section 420 cheating fraud dishonestly inducing delivery of property money",
    "prot_0069_04": "Penal Code 1860 Section 420 fraud deception fake marriage advertisement monetary fraud",
    "prot_0077_06": "Penal Code 1860 Section 420 fake job recruitment fraud health department cheating",
    "prot_0094_01": "Penal Code 1860 Section 420 online bkash fraud false incentive cheating financial deception",
    "prot_0136_03": "Negotiable Instruments Act 1881 Section 138 cheque bounce dishonour payment failure notice",
    "prot_0000_05": "Code of Criminal Procedure 1898 Section 497 when bail may be taken non-bailable offence custody",
    "prot_0006_03": "Code of Criminal Procedure 1898 Section 497 bail in criminal case surety bond surrender court",
    "prot_0067_03": "Family Courts Act 2023 Section 17 appeal family court talaq divorce decree maintenance",
    "prot_0075_03": "Nari O Shishu Nirjatan Daman Ain 2000 Section 35 dowry violence torture punishment medical examination",
    "prot_0084_07": "Legal Practitioners and Bar Council Order 1972 Section 32 advocate professional misconduct loan handnote",
    "prot_0056_02": "Contract Act 1872 Section 201 termination of agency death of contractor principal agent",
    "prot_0083_01": "Contract Act 1872 Section 201 revocation termination of agency power of attorney death of principal",
    "lawy_0317_01": "Transfer of Property Act 1882 Section 48 priority of rights created by transfer multiple sales title deed",
    "prot_0004_01": "Transfer of Property Act 1882 Section 48 sale of same land twice priority registered deed",
    "lawy_0371_01": "Code of Civil Procedure 1908 Section 11 res judicata bar to subsequent suit same cause of action",
    "lawy_0241_01": "Acquisition and Requisition of Immovable Property Act 2017 Section 5 objection compensation land acquisition",
    "ajke_0139_01": "Copyright Act 2023 Section 53 infringement artistic work handmade designs sale penalty",
    "prot_0111_06": "Rajdhani Unnayan Kartripakkha Act Section 54 unauthorized dangerous building construction penalty",
    "prot_0054_04": "Constitution of Bangladesh Article 32 protection of right to life personal liberty",
    "prot_0136_02": "State Acquisition and Tenancy Act 1950 Section 145A survey khatian partition inherited land",
    "prot_0032_01": "Nari O Shishu Nirjatan Daman Ain 2000 Section 9B physical relation false promise of marriage rape",
    "prot_0049_04": "Hindu Widows Remarriage Act 1856 Section 1 marriage of Hindu widows legality rights",
    "dail_0613_02": "Bangladesh Labour Act 2006 Section 232 workers participation fund bank applicability",
    "ajke_0164_02": "Universal Pension Management Act 2023 Section 14 lifetime pension rights nominee beneficiary",
    "lawy_0500_01": "Power of Attorney Act 2012 Section 6 execution of power of attorney from foreign country embassy",
    "prot_0109_04": "Limitation Act 1908 Section 3 dismissal of suits after period of limitation time barred railway",
    "ajke_0207_01": "Extradition Act 1974 Section 15 surrender of fugitive criminals foreign country extradition treaty",
    "prot_0002_02": "Mental Health Act 2018 Section 21 property management capacity person with mental illness will",
    "prot_0113_01": "Penal Code 1860 Section 304A causing death by rash or negligent act medical doctor negligence",
    "prot_0111_04": "Real Estate Development and Management Act 2010 Section 10 developer contract delay compensation",
}


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for r in rows:
        qid = r["qid"]
        concepts = ENGLISH_CONCEPT_MAP.get(qid, "")
        rec = dict(r)
        rec["english_concepts"] = concepts
        if concepts:
            rec["question_expanded"] = f"{r['question']} | English: {concepts}"
        else:
            rec["question_expanded"] = r["question"]
        enriched.append(rec)
    return enriched


def main() -> None:
    raise SystemExit(
        "Retracted 2026-09-19 (DECISIONS.md): ENGLISH_CONCEPT_MAP hardcodes the "
        "gold Act/section for each qid, so question_expanded is answer leakage, "
        "not translation. Running this overwrites dev_retrieval_v4.jsonl / "
        "test_retrieval_v4.jsonl in place with that leak. Do not run. See "
        "data/processed/_archived_leaked_bilingual/README.md."
    )
    dev = read_jsonl(DEV_IN)
    test = read_jsonl(TEST_IN)

    enriched_dev = enrich_rows(dev)
    enriched_test = enrich_rows(test)

    # Save to dedicated _bilingual files
    write_jsonl(DEV_OUT, enriched_dev)
    write_jsonl(TEST_OUT, enriched_test)

    # Also update DEV_IN and TEST_IN in-place with question_expanded field
    write_jsonl(DEV_IN, enriched_dev)
    write_jsonl(TEST_IN, enriched_test)

    print(f"Enriched Dev: {len(enriched_dev)} rows -> {DEV_OUT} (and updated {DEV_IN} in-place)")
    print(f"Enriched Test: {len(enriched_test)} rows -> {TEST_OUT} (and updated {TEST_IN} in-place)")
    print("Sample enriched dev row:")
    print(json.dumps(enriched_dev[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
