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
2. Your tab has one question per row. Columns `A`–`D` are filled in already —
   do not edit them. You fill in columns `E` onward.
3. Work top to bottom. Do not skip around; if you get stuck, put `?` in
   `confidence` and move on.
4. Save nothing, close nothing. Google Sheets saves as you type.

---

## Pass 1 — Adjudicate

For each row you fill four columns.

### `answer` — which section answers this question

Look at the ten numbered candidates in column `D`. Type the **number** of the
right one.

| What you type | What it means |
|---|---|
| `3` | Candidate 3 is the answer |
| `3, 7` | Both 3 and 7 genuinely answer it — put the **better** one first |
| `none` | None of the ten is right, but a right answer probably exists in Bangladeshi law |
| `unanswerable` | No law answers this — it is a factual question, a request for a lawyer, or outside our domains |

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

## Questions

Anything you are unsure about goes in `notes` and gets raised at the next
check-in. Do not invent a rule privately — if you needed a new rule, everyone
else will need it too, and it belongs in this file.
