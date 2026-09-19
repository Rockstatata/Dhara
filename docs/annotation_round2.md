# Round 2 annotation — instructions

**Time:** about 2–4 minutes per row, 60 rows each, so roughly 2.5 hours.
**Your file:** `data/annotation/verify_<yourname>.csv`
**Deadline matters:** every result in the project is measured against these labels, and the paper cannot describe the evaluation set honestly until this round is done.

---

## Why we are doing this again

In round one the sheets were handed out with the `answer` column already filled in by a machine pass, and they came back unchanged — across 170 questions that two people both annotated, **not one answer differed**. That is what happens when a pre-filled sheet is returned as-is, and it means we currently cannot say what the labels are worth or measure agreement between annotators.

We also checked the machine's answers against the advocate's published reply for the 330 questions that have one. Roughly 48% look right, 24% look wrong, and the rest are unclear. So this is not a claim that the old labels are all bad — it is that nobody knows, and knowing is the point of this round.

This time the `answer` column starts **empty**. What you write is the label.

---

## Three rules

1. **Do not use ChatGPT, Claude, Gemini or any other AI to choose the answer.** This is the one rule that cannot be bent. The project's central claim is that our retrieval system finds the right law; if an AI picked the labels, we would only be measuring how well our system agrees with another AI, and the result would be worthless. Use the candidate list, the search tool, and your own reading.
2. **Do not compare sheets with each other.** Some questions deliberately appear in two people's files. If you discuss them, the agreement number we compute from them becomes meaningless.
3. **Do not look at the round-one sheets** (`annotate_*.csv`). Start from the question.

If you are unsure about a row, that is fine and expected — record low confidence and a note. An honest "not sure" is far more useful than a confident guess.

---

## What is in your file

| column | who fills it | what it is |
|---|---|---|
| `qid` | already filled | question id — **do not edit** |
| `annotator` | already filled | your name |
| `block` | ignore | intentionally blank |
| `question_bn` | already filled | the citizen's question — **do not edit** |
| `candidates` | already filled | ten provisions found by BM25 — **do not edit** |
| `answer` | **you** | the section(s) that answer the question |
| `answer_source` | **you** | where you found it |
| `verdict` | **you** | answered / not_found / unanswerable |
| `confidence` | **you** | 3, 2, 1 or ? |
| `notes` | **you** | anything worth recording |

### Reading a candidate line

Each of the ten lines looks like this:

```
3. <labour_952_s340> [বাংলাদেশ শ্রম আইন, ২০০৬, ধারা ৩৪০] নিয়োগ সম্পর্ক অনুমান — কোন ব্যক্তিকে কোন কারখানায় ...
   ^                 ^                                   ^                        ^
   rank              act title and section number        section title            first words of the text
        chunk id
```

---

## Filling each column

### `answer`

Write **either** the candidate numbers **or** the chunk ids — both work:

- `3` — candidate 3 answers the question
- `3, 7` — two sections together answer it
- `labour_952_s340` — a chunk id you found yourself
- `none` — you could not find the answer (in the list or by searching)
- `unanswerable` — **no law answers this question**

**`none` and `unanswerable` are different and must not be mixed up.**
`none` means the answer probably exists in Bangladeshi law but you could not find it. `unanswerable` means there is nothing to find — the person is asking for a lawyer's opinion, asking a factual question, asking about something outside our corpus, or the text is not really a question. Only `unanswerable` is used to teach the system when to stay silent, so mixing them corrupts a different measurement.

**At most 3 sections.** If more seem relevant, pick the three that most directly answer the question.

### What counts as "the answer"

The section a lawyer would cite when answering *this specific question*. Some guidance:

- Prefer the **operative** section over a definition. For "what is the punishment for taking dowry", the answer is the penalty section, not the section that defines dowry.
- Prefer the section that answers **what was asked**. If someone asks what punishment their parents face for marrying them off young, the answer is the section on the parents' punishment — not the general prohibition, and not the registration duty, even though both are about child marriage.
- If the question genuinely has two parts ("can I do this, and what happens if I don't"), two sections are fine.
- Do not add a section just because it is about the same topic. A near-miss recorded as correct is worse for us than a `none`.

### `answer_source`

- `candidate` — you picked from the ten shown
- `own_search` — you found it yourself with the search tool
- `none` — you did not find an answer

This column produces a real number in the paper: how often the BM25 candidate list simply did not contain the answer. Please keep it accurate.

### `verdict`

- `answered` — you gave one or more sections
- `not_found` — you wrote `none`
- `unanswerable` — you wrote `unanswerable`

### `confidence`

- `3` — sure
- `2` — fairly sure, a lawyer might pick a different section
- `1` — guessing
- `?` — could not judge

### `notes`

Optional but valuable. Especially useful: why you rejected an obvious-looking candidate, which part of the question decided it, or that the question text is garbled.

---

## When the candidates do not contain the answer

This happens often, and it is not your fault — it is one of the things the project measures. About a third of our corpus is English-only (the Succession Act, the Registration Act, the Code of Criminal Procedure, and others), and a Bangla keyword search cannot reach them at all.

Use the search tool:

```bash
python scripts/07_search_corpus.py --query "ভরণপোষণ" --k 10     # search by words
python scripts/07_search_corpus.py --query "maintenance of wife" # English works too
python scripts/07_search_corpus.py --act "Succession"            # list an Act's sections
python scripts/07_search_corpus.py --cite family_305_s10         # look up one section
```

Practical tips:

- If you know which Act applies, `--act` and read down its section titles. This is usually faster than guessing keywords.
- For English Acts, search in **English** — the text is English, so Bangla words will not match.
- Copy the chunk id (e.g. `family_305_s10`) into the `answer` column and set `answer_source = own_search`.

If you search and still cannot find it, `none` is the correct answer. Do not stretch to fit a candidate.

---

## Edge cases

| situation | what to do |
|---|---|
| The question text is cut off or has newspaper boilerplate glued on | Answer it if the ask is still clear; otherwise `unanswerable` with a note saying the text is broken |
| The question asks for advice, not law ("should I forgive him?") | `unanswerable` |
| The answer is a whole Act, not a section | Pick the section that most directly applies; note it if nothing fits |
| The section looks repealed or says "omitted" | Do not use it; find the replacement or write `none` with a note |
| Two sections say almost the same thing in different Acts | List both, most specific first |
| You recognise the question from round one | Ignore what you remember it was labelled; judge it fresh |

---

## When you are done

1. Save as **CSV, UTF-8** with the same filename (`verify_<yourname>.csv`).
   In Excel: *File → Save As → CSV UTF-8 (Comma delimited)*. In Google Sheets: *File → Download → Comma-separated values*.
2. Do not add, remove or reorder columns or rows.
3. Put the file in `data/annotation/verified_round2/` (or send it to Sarwad).

## What happens next

- These 200 questions are the project's **frozen test set** — every number in the paper is measured on them, so after this round every one of those labels is human-checked rather than machine-guessed.
- Comparing your answers against round one's machine proposals gives us the **accuracy of the old labels**.
- The questions that two people annotated give us a real **inter-annotator agreement (κ)**, the number reviewers ask for first.
- Both go into the paper's methodology section, and the wrong labels get corrected.

Roughly: 60 rows each, four people, and the test set stops being an unknown.
