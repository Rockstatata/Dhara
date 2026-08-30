# Annotation guide

Read this once before you start. It takes ten minutes and it is the difference
between a dataset and a pile of opinions.

---

## What you are doing, in one paragraph

Real Bangladeshi citizens wrote these questions to newspaper legal columns. Your
job is to say **which section of which law actually answers each question**. The
computer has already guessed ten candidates for you; you pick the right one, or
say none of them is right, or say the question has no answer in our corpus.

That is it. Everything else in this document is detail about edge cases.

**Why it matters:** every number in the paper is measured against your labels. A
model can be retrained in twenty minutes. Your labels cannot be regenerated, and
if they are sloppy the project has no measuring instrument at all.

---

## The two passes

Do them in order. Do not mix them — each pass is a different kind of thinking and
switching between them is what makes annotation slow and inconsistent.

| Pass | What you do | Time per question | Who |
|---|---|---|---|
| **Pass 1 — Adjudicate** | Pick the answer, tag register and domain | ~2 min | everyone |
| **Pass 2 — Paraphrase** | Rewrite the question in your own words | ~2 min | everyone, after Pass 1 is finished |

Pass 2 exists for a legal reason, not a linguistic one: the newspapers own the
words their readers wrote, so we cannot publish those words. We publish your
rewrite. Details in Pass 2 below.

---

## Setup

1. Open the shared sheet. **Find the tab with your name on it.** Only edit your
   own tab.
2. Your tab has one question per row. The first six columns — `qid`,
   `annotator`, `block`, `annotation_mode`, `question_bn`, `candidates` — are
   filled in already. **Do not edit them**, and do not reorder or delete rows: if
   a `qid` or a `candidates` cell changes, the merge cannot resolve your answers
   and your work on that row is lost.
3. Work top to bottom. Do not skip around; if you get stuck, put `?` in
   `confidence` and move on.
4. Save nothing, close nothing. Google Sheets saves as you type.

---

## Pass 1 — Verify, not author

**This is no longer a blank sheet.** Every row already has an `answer`,
`answer_source`, `register`, `domain` and `confidence` filled in by a script
(`scripts/09_autolabel_pass1.py`), and `pass1_auto = 1` marks every one of them
as a machine guess. Your job changed from *pick the answer* to *check the
machine's pick and fix it when it's wrong* — which is faster per row, but only
if you actually check rather than accept what's there.

**Check first, because the guess is wrong most of the time.** We measured this
rather than assumed it: where a practising advocate's published reply names an
Act, the retriever that produced these guesses put the right Act in its top 10
only 15% of the time. It does worse than that when the correct provision is in
English — the Succession Act, the Muslim Family Laws Ordinance, the Code of
Criminal Procedure, the Registration Act, the Negotiable Instruments Act, the
State Acquisition and Tenancy Act. A 7-question hand-checked probe found the
correct English-language answer in the top 50 results **zero times out of
five**, for both the lexical retriever and a zero-shot multilingual encoder.
Bengali-source answers fared much better. So: **the vocabulary above, weight
your suspicion accordingly** — a proposed answer in an English Act is the one
most likely to be wrong, not the one you can wave through fastest because it
looks confident.

### The `verified` column — mark every row you touch

New column, blank by default. **Type `y` once you have actually looked at a
row** — whether you kept the machine's answer or replaced it. This is what lets
`08_merge_gold.py` tell a real measurement from the auto-labeller grading its
own homework: agreement, the candidate-list miss rate, and the unanswerable
count are only computed over `verified` rows, and until at least one row is
verified the script prints a loud warning instead of numbers. An unverified row
is not data. Leaving `verified` blank on a row you actually checked is
indistinguishable, to the script and to the next person, from never having
looked at it — so mark it every time, including when you change nothing.

### How to work through a row

1. **Read the question.** As always.
2. **Read `lawyer_answer` first, if it is not blank.** This is the practising
   advocate's own published reply, recovered from the same newspaper column the
   question came from (see `scripts/11_extract_answers.py`). It is the single
   strongest verification aid on the sheet — 61% of rows have one. It is
   evidence, not a label: advocates name a remedy or an Act and only sometimes a
   section, and on questions with no statutory answer they give practical
   advice citing no law at all. `lawyer_names_law` tells you at a glance whether
   this row's reply names one.
3. **Compare the proposed `answer` against what the advocate said.** If the
   advocate names an Act and the proposed answer is from a different Act, the
   proposed answer is very likely wrong — treat this as your strongest single
   signal and go straight to a search rather than spending time on the
   candidate list.
4. **If there is no `lawyer_answer`,** fall back to your own judgement and the
   candidate list exactly as described below.
5. **Decide:** keep the proposed answer, replace it with something you find
   yourself (`answer_source = own_search`), mark `none` (nothing you could find
   answers it), or mark `unanswerable` (no law addresses it at all — see the
   table below).
6. **Fill in `register`, `domain`, `confidence` for real** — the machine's
   guesses there are weaker than its answer guess and should not be trusted
   further than a starting point.
7. **Mark `verified = y`.** Every time, whether or not you changed anything.

### Work order: confidence first, not top to bottom

The auto-labeller's `confidence` column (`1`/`2`/`3`) is a real signal, derived
from how much the machine preferred its own top answer over the runner-up — not
a placeholder. Work your sheet in this order:

1. **`confidence = 1` rows first.** This is where wrong answers concentrate,
   and where `none` and `unanswerable` are hiding — a low-confidence guess is
   often the machine's way of returning *something* when nothing in the corpus
   actually fits.
2. **`confidence = 2` next.**
3. **`confidence = 3` last**, and expect most of these to be quick confirmations
   rather than corrections.

Sort or filter your sheet by `confidence` before you start a session — the
generated CSV is not pre-sorted this way.

### Your columns

The sheet fills in `qid`, `annotator`, `block`, `annotation_mode`,
`question_bn`, `candidates`, `lawyer_names_law` and `lawyer_answer` for you.
**Do not edit those.** `answer`, `answer_source`, `register`, `domain` and
`confidence` are also pre-filled — this time as a proposal for you to check, not
a value to leave alone. `verified`, `notes`, `paraphrase` and `formal_version`
start blank and are yours alone.

### `answer` — which provision answers this question

The cell already shows the machine's proposal as a **full citation line**, the
same format the `candidates` column uses — `<chunk_id>` followed by the Act
title, ধারা number and a snippet, so you can read what it is without leaving
the sheet:

```
<other_850_s34> [সালিস আইন, ২০০১, ধারা ৩৪] পক্ষগণ ভিন্নভাবে সম্মত না হইলে-(ক) সালিসি...
```

If the machine's proposal is right, leave it and move on. If it is wrong,
replace the whole cell with any of the following — the parser reads the
`<chunk_id>` marker and ignores the rest of the text around it, so you can type
the short form or paste a full citation line, whichever is faster for you:

| What you type | What it means |
|---|---|
| `3` | Candidate 3 (from the `candidates` column) is the answer |
| `3, 7` | Both genuinely answer it — put the **better** one first |
| `family_138_s52` | A provision you found yourself — bare chunk id |
| a full `<chunk_id> [...] ...` line | Copy this straight out of the `candidates` column, the search tool's output, or what Claude shows you — no retyping needed |
| **"Act name, section N"** | **No chunk_id at all — see below.** |
| `none` | Nothing you could find answers it |
| `unanswerable` | No law answers this — a factual question, a request for a lawyer, or outside our domains |

For more than one genuine answer, put each on its own line inside the cell (the
same way `candidates` already does) rather than separating with commas — commas
appear inside Act titles and citation text, so they cannot be the separator.

**You never have to remember or type a `chunk_id`.** If you have done your own
research — Google, a legal site, a textbook — and you know the Act and section
but not our internal id for it, just write it the way you'd naturally cite it:

```
Muslim Marriages and Divorces (Registration) Act, 1974, section 3
```

or in Bangla:

```
সালিস আইন ধারা ৩৪
```

The merge script matches the Act name against every title in the corpus and
finds the exact provision — you do not need the exact official title, an
abbreviation that's still recognisable works ("Shariat Act section 2" resolves
correctly). One citation per line, same as everything else in this cell.

Two things worth knowing about how this works, so it doesn't surprise you:

- **It matches whatever language the corpus stored the title in.** Most English
  colonial-era Acts (the Penal Code, the Code of Criminal Procedure, the
  Succession Act...) only have an English title in our corpus, even though
  everyone calls them by a Bangla name in conversation (দণ্ডবিধি, ফৌজদারি
  কার্যবিধি). For those, write the **English** name — "Penal Code section 379",
  not "দণ্ডবিধি ধারা ৩৭৯". If you're not sure which language to use, ask Claude
  or run `scripts/07_search_corpus.py --act "<a fragment>"` to check first.
- **If it can't find a confident match**, `08_merge_gold.py` tells you exactly
  what it couldn't identify and asks you to try a `<chunk_id>` instead — it will
  never silently guess the wrong Act. If you hit this, it usually means the Act
  isn't in the corpus at all, which is itself worth a note.

### `answer_source` — where the answer came from

One word, and it matters more than it looks:

| Value | When |
|---|---|
| `candidate` | you picked one of the ten |
| `own_search` | you found it yourself because the ten were all wrong |
| `none` | you found nothing |

This column is why the sheet can report **how often the candidate list missed**.
That miss rate is a headline number in the report, and without this column an
answer you dug out by hand is indistinguishable from one the list handed you. It
matters most exactly where the list is weakest — the English-language acts in the
table below.

**Never guess.** `none` is a real, useful answer and we count it deliberately. A
wrong label is far worse than an honest `none`, because a wrong label silently
tells the model that a correct answer is incorrect.

**What counts as "the answer":** the section that *directly* answers the
question. Not a section that mentions the topic. Not a section that would come
up in a lawyer's full advice. If the question is "can my landlord throw me out
whenever he likes", the answer is the section stating the grounds for eviction —
not the section defining "landlord".

**Maximum three answers.** If more than three sections seem right, you are
probably picking sections that merely *relate* to the topic. Tighten up.

### Read this before you start on family, land, or police questions

Some of the law you are looking for **is written in English**, and the candidate
list will not find it.

Bangladeshi law enacted before roughly 1987 was written in English and has never
been translated. That includes the acts that actually govern the questions
citizens ask most:

| If the question is about | The governing law is |
|---|---|
| Inheritance, who gets what share | **The Succession Act, 1925** (English) |
| Muslim marriage, divorce, maintenance | **The Muslim Family Laws Ordinance, 1961** (English) |
| Land registration, deeds, mutation | **The Registration Act, 1908** (English) |
| Land tenancy, khatian | **The State Acquisition and Tenancy Act, 1950** (English) |
| Arrest, FIR, bail | **The Code of Criminal Procedure, 1898** (English) |
| Any offence and its punishment | **The Penal Code, 1860** (English) |
| Cheque bounced | **The Negotiable Instruments Act, 1881** (English) |

The candidates are generated by keyword matching on Bangla. A Bangla question
therefore **cannot** match an English provision, so for these topics the ten
candidates will often all be wrong — usually plausible-looking Bangla acts about
property or money.

**What to do:** if the question is on one of these topics and none of the ten
candidates is right, search the English act named above directly, and enter the
provision you find. If you genuinely cannot find it, type `none`. Do **not**
pick the least-bad Bangla candidate — that is the single worst thing you can do
to this dataset, because it teaches the model that a wrong answer is right.

This is not a bug in the sheet. It is the finding the project exists to measure,
and your `none` labels are what quantify it.

### How to search for it yourself

```bash
python scripts/07_search_corpus.py
```

It loads the corpus once and gives you a prompt with three modes:

```
search> ভরণপোষণ স্ত্রী            # keyword search over provision text
search> act:Succession            # list one Act's provisions by title
search> id:family_138_s125        # print one provision in full
```

**`act:` is the one that matters** for everything in the table above. Keyword
search runs on the same Bangla matching that produced your candidates, so it will
fail on an English act for exactly the same reason. Go at it by Act name instead
and read down the section titles — they are in English and they are descriptive.
`act:Succession`, `act:Registration`, `act:Criminal Procedure`, `act:Penal`,
`act:Negotiable`.

Copy the chunk id out of the angle brackets into `answer`, and set
`answer_source` to `own_search`.

One-shot forms, if you would rather not keep a prompt open:

```bash
python scripts/07_search_corpus.py --query "ভরণপোষণ স্ত্রী" --k 10
python scripts/07_search_corpus.py --act "Registration"
python scripts/07_search_corpus.py --cite family_138_s125
```

**Give yourself ten minutes on a hard row**, then write your best answer with
`confidence 1`, or `none`, and a line in `notes`. A row that eats forty minutes
is not worth four ordinary rows.

### `register` — how the person is speaking

This is the single most important tag in the project and it is the one people get
wrong. It is about **the words the person used**, not about who they are, not
about whether the writing is polished, and not about the topic.

| Tag | Means | Test |
|---|---|---|
| `colloquial` | Everyday speech. The person describes what happened to them. | Would you hear this sentence in a conversation? |
| `formal` | Legal or bureaucratic vocabulary. The person names the legal concept. | Does it use a term you would only learn from a lawyer or a form? |

**Colloquial markers:** narrative framing (`আমার বাবা মারা গেছেন, এখন…`),
everyday verbs (`ধরে নিয়ে গেছে`, `বের করে দিতে পারবে`), no technical terms,
emotion, first-person trouble.

**Formal markers:** legal nouns (`উচ্ছেদ`, `নামজারি`, `অগ্রক্রয়`, `জামিন`,
`রিট`, `অভিভাবকত্ব`), nominalised phrasing (`আইনগত ভিত্তি কী`,
`পদ্ধতি কী`), the shape of a question asked *about* the law rather than about a
situation.

**The rule when it is mixed:** if the person describes a real situation in
everyday words *and* uses one legal term they clearly picked up somewhere, tag it
`colloquial`. One borrowed word does not make someone a lawyer.

Two examples that look similar and are not:

> `colloquial` — বাড়িওয়ালা কি ইচ্ছা করলেই আমাকে বাসা থেকে বের করে দিতে পারবে?
> `formal` — ভাড়াটিয়া উচ্ছেদের আইনগত ভিত্তি কী?

Same legal issue. Completely different register. That contrast is the paper.

### `domain` — which area of law

Type one of these exactly:

`family` · `land` · `labour` · `consumer` · `cybercrime` · `constitutional` ·
`criminal_procedure` · `women_children` · `tenancy` · `civil_registration` ·
`road_transport` · `money_recovery` · `other`

Pick by **what the person needs**, not by which law technically applies. A woman
asking how to get her dower after divorce is `family`, even though enforcement
runs through a court procedure.

Use `other` freely. A pile of honest `other` tags tells us we missed a domain,
which is useful. A forced wrong tag tells us nothing and corrupts the per-domain
results table.

### `confidence` — how sure are you

| Value | Meaning |
|---|---|
| `3` | Certain |
| `2` | Fairly sure |
| `1` | Guessing — someone should look at this again |
| `?` | Stuck, skipped, needs discussion |

Be honest here. `1` and `?` rows get reviewed by the group; they are not
failures, they are how we find the genuinely hard cases, and the hard cases are
what the error-analysis section of the paper is made of.

### `notes` — optional

One line, only when something is worth saying. "Two sections both apply",
"question is really about procedure not the right", "seems to be about the old
repealed act". Leave blank otherwise.

---

## Working with Claude while you verify

You have access to Claude Code in this repo. Use it as a fast, tireless
research assistant for Pass 1 — not as the annotator. That boundary is not a
courtesy, it is load-bearing: the whole point of a human-adjudicated gold set is
that the evaluation measures the model against an independent judgement. If a
model ends up choosing the answers, the "independent" part is gone and the
project's headline claim — that fine-tuning closes the gap a general model
cannot — becomes circular, because the gold set and the thing being measured
came from the same family of model. **A machine may propose. Only you decide.**

### What to actually ask for

Claude can, in seconds, do the parts of verification that are slow by hand:

- **"Look up `<qid>` in `annotate_<name>.csv`. What does the candidate list say,
  and does the advocate's reply point somewhere else?"** — a fast first read on
  a row before you commit real time to it.
- **"Search the corpus for `<topic>`"** or **"list the provisions of the
  Succession Act / Registration Act / Code of Criminal Procedure"** — this runs
  `scripts/07_search_corpus.py` for you. Say it in plain language; you do not
  need to remember the command yourself.
- **"Show me the full text of `<chunk_id>`"** — before you commit an answer,
  especially a `pass1_auto` one you're inclined to trust, ask to see the
  provision in full rather than the 60-character snippet on the sheet.
- **"The advocate's reply says X — which provision actually matches that?"** —
  paste the `lawyer_answer` text and let Claude search on your behalf; this is
  usually the fastest path to a correct answer on a row the machine got wrong.
- **"I think it's between chunk A and chunk B — what's the difference?"** — for
  the genuinely ambiguous rows, ask for both texts side by side rather than
  guessing.
- **"Run the merge script and tell me where I stand"** — at any point, ask for
  current counts: how many of your rows are verified, the miss rate so far, how
  many `unanswerable`. Useful for pacing a session and for catching a systematic
  mistake (a misread column, a format the merge script rejects) early rather
  than at the end.

### What Claude will not do, and why

It will not hand you a final answer with no way to check it — it will show you
the provision's actual text, or the advocate's reply, and let you make the call.
If you ask it to "just pick the right one," expect it to push back or to give
you its reasoning and the source text rather than a bare answer, because a
label with no visible reasoning behind it is exactly as unverifiable coming from
Claude as it was coming from the BM25 auto-labeller — and the auto-labeller's
proposals are already 85% wrong on the cases we could check. Read what it shows
you. If it's wrong, say so; that correction is worth writing into `notes` or, if
it reveals a rule the guideline should state, worth raising at the next
check-in so this document gets the rule too.

### A worked example

> **You:** qid `prot_0003_02` — the machine proposed `labour_952_s103`, but the
> question is about a husband not paying maintenance after a second marriage.
> Can you check?
>
> **Claude:** That chunk is the Labour Act's provision on weekly holidays
> (সাপ্তাহিক ছুটি) — completely unrelated. Searching the corpus for family court
> maintenance provisions returns `family_1444_s5`, jurisdiction of the Family
> Court, which lists ভরণপোষণ (maintenance) as one of five matters it hears. The
> Muslim Family Laws Ordinance's maintenance section is `family_305_s9` — full
> text: [...]. Given the question, `family_1444_s5` is the better single answer
> since it establishes *where* to file; `family_305_s9` is a defensible second
> answer if you want to cite the substantive right as well as the forum.
>
> **You decide, write it in, mark `verified = y`.**

---

## Pass 2 — Paraphrase

Only after Pass 1 is completely finished.

Column `paraphrase`: **rewrite the question in your own words.**

Rules:

1. **Keep the register.** If it was colloquial, your rewrite stays colloquial. If
   the person wrote like a villager, write like a villager. This is the whole
   point — do not tidy it into textbook Bangla.
2. **Change the wording.** Different sentence structure, different phrasing. Not
   a synonym swap; a genuine rewrite.
3. **Keep the legal substance identical.** Same situation, same thing being
   asked. If your rewrite would have a different correct section, you changed too
   much.
4. **Remove every identifying detail.** Names, districts, phone numbers, ages
   when they identify someone, workplace names, court case numbers. Replace with
   generic equivalents: `আমার বোন`, `একটি গ্রামে`, `একটি প্রতিষ্ঠানে`.
5. **Length: 10–40 words.** The originals ramble; citizens searching a system
   type shorter. Compress, but do not turn a story into a keyword.

Example:

> **Original (do not publish):** আমার নাম রহিমা খাতুন, বাড়ি নোয়াখালী। আমার
> স্বামী গত বছর মারা গেছেন। আমাদের তিন ছেলে দুই মেয়ে। এখন আমার শ্বশুরবাড়ির লোকজন
> বলছে আমি নাকি জমির কোনো ভাগ পাব না…
>
> **Paraphrase (publish this):** স্বামী মারা যাওয়ার পর শ্বশুরবাড়ির লোকজন বলছে আমি
> জমির ভাগ পাব না। আমি কি সত্যিই কিছু পাব না?

---

## Special task — formal counterparts

About 100 questions are marked `PAIR` in column `C`. For those, fill one extra
column: `formal_version`.

Write the **same question in formal Bangla** — the way a law student or a clerk
would ask it. Same situation, same correct section, different register.

> `colloquial` — বাড়িওয়ালা কি ইচ্ছা করলেই আমাকে বাসা থেকে বের করে দিতে পারবে?
> `formal_version` — ভাড়াটিয়া উচ্ছেদের ক্ষেত্রে বাড়িওয়ালার আইনগত সীমা কী?

**Do not translate into English.** Formal means formal *Bangla*. If English
creeps in, the experiment stops measuring register and starts measuring
translation, and the result becomes meaningless.

This is the controlled half of the central experiment: because both versions
point at the same section, any difference in retrieval is caused by register and
nothing else.

---

## Calibration round — do this first, before anything else

Before the real work, **everyone annotates the same 20 questions** (the first 20
rows, marked `CALIBRATION`). Then we sit down for thirty minutes, compare, and
argue about the disagreements.

Do not skip this. It is where this guide actually gets written — every rule above
came from someone disagreeing about a real case. Expect to disagree on about a
third of them the first time. That is normal and it is why we do it.

---

## Agreement round

150 questions are annotated by **two** people independently, without looking at
each other's answers. Those rows appear in two tabs. If you find a question you
think you have seen before, that is the point — answer it fresh and do not go
hunting for what you said last time.

This produces the agreement statistic (κ) that goes in the paper. A reviewer who
sees no agreement number assumes the labels are unreliable, and they are right to.

---

## Pace, and when to stop

- **~2 minutes per question in Pass 1.** If you are consistently slower, you are
  over-thinking; mark `confidence 1` and move on.
- **Work in 45-minute blocks.** Accuracy falls off a cliff after that, and tired
  annotation is worse than no annotation because it looks the same.
- **Two hours a day is the plan.** Do not binge eight hours on one day — the last
  six will be noise.

---

## Common mistakes, ranked by how often they happen

1. **Tagging register by topic.** A question about the Constitution is not
   automatically `formal`. Read the words.
2. **Picking a section that mentions the topic** instead of the one that answers
   the question.
3. **Avoiding `none`.** If the ten candidates are all wrong, say so. `none` is
   data; a forced pick is corruption.
4. **Tidying the paraphrase into proper Bangla.** You are preserving how people
   actually talk. Bad grammar is a feature.
5. **Marking `unanswerable` for hard questions.** `unanswerable` means *no law
   answers this*, not *I could not find it*. Use `none` for the second.
6. **Editing someone else's tab.**

---

## Merging the four tabs

Export each tab back to `data/annotation/` as CSV — keeping the filename
`annotate_<yourname>.csv` — and run:

```bash
python scripts/08_merge_gold.py
```

It reads every sheet, resolves each answer to a real chunk id, checks it against
the corpus, and reports how many rows each person completed, the agreement
figures, the candidate-list miss rate, and how many questions came back
`unanswerable`. It writes `data/processed/gold_test_v1.jsonl`.

**Run it early — at ten or twenty rows each, not at the end.** It catches a
systematic mistake (a misread column, an answer format that does not resolve, a
value spelled differently from this guide) while it is twenty rows of rework
instead of two hundred. It prints the file and line number of every row it could
not read.

Two notes on what it reports, both of which go in the paper as stated:

- Agreement on the **answer** is a raw percentage, not κ. With several thousand
  candidate provisions, agreement by chance is effectively zero, so κ collapses
  onto the same percentage and calling it κ dresses a plain number in borrowed
  authority. Agreement on **`domain`** and **`register`** — a handful of
  categories each — *is* reported as Cohen's κ, which is what κ is for.
- Where two of you disagreed, the merge keeps **both** records and does not pick
  a winner. Adjudicating those is a decision the four of you make together, and
  it gets written down.

If you opened your CSV in Excel at any point: make sure it opened **as UTF-8**
and that Excel did not re-save it in another encoding. Bangla saved in the wrong
encoding is mojibake and the file cannot be recovered.

---

## Questions

Anything you are unsure about goes in `notes` and gets raised at the next
check-in. Do not invent a rule privately — if you needed a new rule, everyone
else will need it too, and it belongs in this file.
