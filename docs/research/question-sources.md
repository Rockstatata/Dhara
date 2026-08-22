# Sources of real Bangladeshi citizen legal questions

**Research date:** 2026-08-22
**Purpose:** find real, colloquial-Bangla citizen legal questions to build a question → law-provision evaluation set for Dhara.
**Method:** every claim below comes from a page fetched during this investigation (`curl` with a `Dhara-Research-Crawler/1.0 (student NLP project)` UA, or WebFetch). Counts were obtained by enumerating listing pages or by API queries, and the method is stated per row. Nothing was scraped from Facebook — section (d) is a legality assessment only.

**Skeptical note up front:** several sources that "should" exist do not. BLAST's FAQ page is an empty template. Bissoy.com is no longer a general Q&A site. Jugantor, Ittefaq, Samakal, Bangladesh Pratidin, Dhaka Tribune and bdnews24 have no findable reader legal-advice column. Those negative results are recorded below rather than glossed over.

---

## 1. Summary

### The five best sources

| # | Source | Why |
|---|---|---|
| 1 | **Prothom Alo — পাঠকের উকিল** ([listing](https://www.prothomalo.com/lifestyle/%E0%A6%AA%E0%A6%BE%E0%A6%A0%E0%A6%95%E0%A7%87%E0%A6%B0-%E0%A6%89%E0%A6%95%E0%A6%BF%E0%A6%B2-36)) | 92 articles, ~4–5 reader letters each (~410 questions). Longest, most colloquial, most emotionally-real register available anywhere. Reachable through a public JSON API. |
| 2 | **Ajker Patrika — আইনি পরামর্শ** ([listing](https://www.ajkerpatrika.com/women/legal-advice)) | 71 articles, exactly one Q&A each, and the *only* source where question and answer are cleanly machine-separable (`উত্তর:` marker). Static HTML. Highest data quality per unit of parsing effort. |
| 3 | **Lawyers Club Bangladesh — দৈনন্দিন জীবনে আইন** | 382 posts, 26% titled as a plain-Bangla question. Crucially **land-heavy** (নামজারি, খতিয়ান, অগ্রক্রয়, বাটোয়ারা) — the domain every other source misses. Open WP REST API, permissive robots.txt. |
| 4 | **`momahadi/bangladesh-legal-qa-dataset`** ([HF](https://huggingface.co/datasets/momahadi/bangladesh-legal-qa-dataset)) | 2,165 question→section records, CC-BY-4.0, with statutory corpus. Not citizen register (LLM-generated scenarios), but it is the closest published prior work and changes the related-work section. |
| 5 | **Daily Star — Your Advocate** ([listing](https://www.thedailystar.net/ds/law-our-rights/your-advocate)) | 43 reader queries, and the only source with real **labour, tax, consumer and company-law** coverage. English, and the question text is often not rendered — see the caveat in the table. |

### Estimated total real citizen questions reachable

| Bucket | Estimate |
|---|---|
| Prothom Alo পাঠকের উকিল | ~350–460 (92 articles × 4–5 letters) |
| Prothom Alo পাঠকের প্রশ্ন (law variants) | ~95 (24 articles × ~4) |
| Ajker Patrika আইনি পরামর্শ | 71 (exact count) |
| Daily Star Your Advocate | 43 (exact count; question text partly missing) |
| Jagonews24 আইনি পরামর্শ topic | 10 (exact count) |
| **Total genuine citizen-written questions** | **≈ 570–680** |
| Question-*titled* editorial explainers (LCB, usable as queries not as citizen text) | ~99 |
| Already-extracted Prothom Alo Q&A on Hugging Face (subset of the above, not additive) | 63 |

**≈ 600 real citizen questions is the realistic ceiling**, against a corpus target of 800–1,200 chunks. That is enough for a gold test set and a held-out dev set, but not enough to train on — which is consistent with Dhara's plan of synthetic training questions plus a real-question gold set.

---

## 2. Full source tables

### (a) Newspaper legal-advice columns

| Source & URL | ~Questions & how counted | Domains | Lang | Structure | robots.txt / ToS | PII in questions | Q separable from A? |
|---|---|---|---|---|---|---|---|
| **Prothom Alo — পাঠকের উকিল**<br>[listing](https://www.prothomalo.com/lifestyle/%E0%A6%AA%E0%A6%BE%E0%A6%A0%E0%A6%95%E0%A7%87%E0%A6%B0-%E0%A6%89%E0%A6%95%E0%A6%BF%E0%A6%B2-36) · articles at `/lifestyle/পাঠকের-উকিল-1…92` | **92 articles.** Counted via Quintype API `advanced-search?q="পাঠকের উকিল"` → `total: 93`, of which 92 have the exact headline. Date range 2009-10-20 → 2017-10-17. Sampled 10 articles: mean 4,478 chars, mean 6.5 `?` marks → **~4–5 reader letters/article ≈ 410 questions**. | Family/marriage/talaq/দেনমোহর dominant; inheritance & land partition; some criminal, co-operative/money recovery, child custody | Bangla | **Listing is JS-rendered** (no `__PRELOADED_STATE__`), but a **public JSON API** makes it static-equivalent: `/api/v1/advanced-search?fields=headline,url&q=…` and `/api/v1/stories-by-slug?slug=lifestyle/<slug>`. requests-only, no browser. | robots.txt: `User-agent: *` / `Allow: /`; only `/shell.html`, `/api/auth/`, `/api/comments/get_comments_json`, `/story/*/element/` disallowed — **the API paths used here are not disallowed**. ToS is restrictive, see §3. | **High.** Real first names + district appear as signatures (e.g. `ইব্রাহীম খলিল`, `সুমা, চট্টগ্রাম`). Some anonymised in-source (`নাম প্রকাশে অনিচ্ছুক` — 0–4 per article in the sample). No phone/NID observed. | **Hard.** Older migrated articles collapse the whole column into **one `<p>` blob** with no `*` delimiters and no `প্রশ্ন:`/`উত্তর:` markers. Boundary must be inferred from the signature line. A few newer ones retain `*` bullets. Budget real parser effort here. |
| **Prothom Alo — পাঠকের প্রশ্ন: আইন / আইনি পরামর্শ**<br>[listing](https://www.prothomalo.com/lifestyle/%E0%A6%AA%E0%A6%BE%E0%A6%A0%E0%A6%95%E0%A7%87%E0%A6%B0-%E0%A6%AA%E0%A7%8D%E0%A6%B0%E0%A6%B6%E0%A7%8D%E0%A6%A8-%E0%A6%86%E0%A6%87%E0%A6%A8) | **~24 law articles.** API `q="পাঠকের প্রশ্ন" আইন` → `total: 36`, broken down by exact headline: `পাঠকের প্রশ্ন, আইনি পরামর্শ` ×14, `পাঠকের প্রশ্ন: আইন` ×8, `পাঠকের প্রশ্ন আইনি পরামর্শ` ×1, `পাঠকের প্রশ্ন, আইনজীবীর উত্তর` ×1. The other 12 are diet/mind/editor columns, **not law**. Range 2009→2020. ≈ **95 questions**. | Family, cybercrime (per the paper's own 2023 round-up), general civil | Bangla | Same Quintype API | Same as above | Same as above | Same as above |
| **Ajker Patrika — আইনি পরামর্শ**<br>[listing](https://www.ajkerpatrika.com/women/legal-advice) (also served at `/lifestyle/legal-advice`, **same content — do not double-count**) | **71 articles = 71 questions.** Counted by enumerating `?page=1..6` and deduping `/women/legal-advice/<id>` hrefs: 16+16+16+16+7+0. One reader question per article. | Family/inheritance (widow's share, guardianship, দেনমোহর), women-focused framing; some civil procedure | Bangla | **Static server-rendered HTML.** Next.js app but the listing hrefs and full article body are in the initial HTML — `requests` + `bs4` is sufficient. | robots.txt: `User-agent: *` / `Allow: /`, disallows only `/cgi-bin/`, `/cdn-cgi/`, `/register`, `/login`, `/api/`, `/*?_rsc=`. **Article and listing paths are allowed.** | **Moderate.** Questions are unsigned in body; age/relationships/district often present in the narrative ("আমার বয়স ৪০ বছর…"). Answering lawyer is named (`পরামর্শ দিয়েছেন, ব্যারিস্টার ইফফাত গিয়াস আরেফিন`) — that's a professional, not a citizen. | **Yes — cleanly.** Verified structure: reader's question paragraph → `উত্তর:` → answer → `পরামর্শ দিয়েছেন, …`. A single split on `উত্তর:` works. **Best separability of any source found.** |
| **The Daily Star — Your Advocate**<br>[listing](https://www.thedailystar.net/ds/law-our-rights/your-advocate) | **43 articles.** Counted by walking `?page=0,1,2,3` and taking the union of unique `/law-our-rights/your-advocate/…` hrefs: 17 → 36 → 43 → 43 (no new). Deep pages (5, 8, 12, 20, 25) return a constant 7 links which are sidebar/related, **not real pagination** — do not mistake this for hundreds of articles. | **Labour** (retrenchment, gratuity, CBA), **tax**, company/minority shareholders, IP, cyber-bullying, interfaith marriage, alcohol law, voting | **English** | Static HTML (Drupal). Note `/law-our-rights` 301-redirects to `/ds/law-our-rights`. | robots.txt is a stock Drupal file: `User-agent: *`, disallows `/core/`, `/profiles/`, `/README.md`. **Nothing relevant is disallowed.** | Low — queries are editorially summarised and typically unsigned. | **Problem.** On the article checked ([Pandemic, retrenchment and labour law](https://www.thedailystar.net/law-our-rights/your-advocate/news/pandemic-retrenchment-and-labour-law-2209706)) the rendered page carries **only the lawyer's answer**; the reader's query is not in the HTML (the answer opens "Thank you for your query…"). Treat the 43 as *answers* whose questions may need reconstruction from the title. |
| **Jagonews24 — আইনি পরামর্শ topic**<br>[topic page](https://www.jagonews24.com/topic/%E0%A6%86%E0%A6%87%E0%A6%A8%E0%A6%BF-%E0%A6%AA%E0%A6%B0%E0%A6%BE%E0%A6%AE%E0%A6%B0%E0%A7%8D%E0%A6%B6) | **10 articles** — counted unique article hrefs on the topic page. Small. | Mixed; mostly `law-courts` section | Bangla | Static HTML | robots.txt carries a **Cloudflare Content-Signal**: `User-agent: *` / `Content-Signal: search=yes,ai-train=no,use=reference` / `Allow: /`. **`ai-train=no` is an express reservation of rights against using this content to train or fine-tune models.** Blocks Amazonbot, Applebot-Extended, Bytespider outright. | Low | Not assessed (volume too low to matter) |
| **Kaler Kantho — আইন-আদালত**<br>[/online/court](https://www.kalerkantho.com/online/court) | **0 reader Q&A.** Section loads (HTTP 200, title `আইন-আদালত`) but is court *news*, and zero `/online/court/<article>` hrefs were extractable from the listing HTML. No reader-question column found. | — | Bangla | — | robots.txt allows `/` but **disallows `/search`, `/search?*`, `/topic/`, `/api/`** — which removes the obvious ways to find a column by name. | — | — |
| **Jugantor** | **None found.** `/law`, `/ain-adalat`, `/আইন-আদালত` all 404. `/todays-paper/law-and-justice` returns 200 but with an empty `<title>`. No reader legal-advice column located. | — | Bangla | — | `User-agent: *` / `Allow: /`, disallows `/ajax/*` | — | — |
| **Ittefaq / Dhaka Tribune / Bangladesh Pratidin / Samakal / bdnews24 / Banglanews24** | **None found.** `ittefaq.com.bd/law-and-justice` → 404; `dhakatribune.com/topic/legal-advice` → 404; `bd-pratidin.com/law-and-justice` → 404; `samakal.com/law-and-justice` → 200 but generic homepage title. `samakal.com/robots.txt` and `bdnews24.com/robots.txt` both return Cloudflare **301/302 redirects and no robots body** — I could not read their robots.txt at all. Targeted Bangla web searches for reader legal-advice columns in these outlets returned nothing. | — | — | — | Ittefaq & Dhaka Tribune share the same robots template (`Allow: /`, disallow `/api/`, `/login`, `/register`). **Samakal and bdnews24 robots.txt unverified.** | — | — |

### (b) Government portals

| Source & URL | ~Questions | Domains | Lang | Structure | robots.txt / ToS | PII | Q/A separable |
|---|---|---|---|---|---|---|---|
| **land.gov.bd FAQ**<br>[/faq](https://land.gov.bd/faq) | **9 questions — exact count** (Radix accordion items extracted from server-rendered HTML). | **None legal.** All 9 are account/service-technical: password reset, hotline number, profile completion %, NID verification, registering for ভূমি উন্নয়ন কর. | Bangla | Static server-rendered HTML (Next.js RSC, but content is in the initial payload) | robots.txt present and carries a Cloudflare **Content-Signal** block; **no `Disallow` for `/faq`**. | None | N/A |
| **NLASO — nlaso.gov.bd** | **0 citizen questions found.** Homepage loads (HTTPS, HTTP 200, `হোম \| বাংলাদেশ আইনগত সহায়তা অধিদপ্তর`). Scanned all homepage hrefs for প্রশ্ন / জিজ্ঞাস / FAQ — **only two matches, both the site's own title text**. `/faq`, `/site/view/faq`, `/site/page/faq`, `/site/faq` all 404. | — | Bangla | Static HTML (national portal template) | **No robots.txt** — `nlaso.gov.bd/robots.txt` returns the site's styled HTML 404 page, not a robots file. Same for minlaw.gov.bd, dife.gov.bd, dncrp.gov.bd. No robots file means no declared crawl restriction, but also no permission. | — | — |
| **minlaw.gov.bd**, **dife.gov.bd**, **dncrp.gov.bd** | **0 FAQ pages found.** All three load over HTTPS (200) but every FAQ path probed returns the portal 404 (`সমস্যা হয়েছে \| …`). | — | Bangla | Static HTML | No robots.txt (see above) | — | — |
| **dncrp.portal.gov.bd** | **Unreachable.** Returns HTTP 200 with body `Domain is not available: dncrp.portal.gov.bd` / `ওয়েবসাইটটিকে সাময়িকভাবে নিষ্ক্রিয় রাখা হয়েছে`. **The working host is [dncrp.gov.bd](https://www.dncrp.gov.bd/)** — use that, not the portal subdomain in the brief. | — | — | — | — | — | — |
| **judiciary.gov.bd** and district subdomains | Loads (`judiciary.gov.bd` → 200 `Judiciary: Bangladesh`; `dhaka.judiciary.gov.bd` → 200). **No citizen-question content located** in this pass. `nlaso.gov.bd/site/page/legal-aid` → 404. | — | Bangla/English | Static | Not checked per-subdomain | — | — |
| **Cyber Support for Women** | **Could not verify.** `cybersupport.police.gov.bd` failed to connect (curl exit 000, no response). The unit's public presence appears to be its Facebook page — see §(d). `police.gov.bd` itself loads. | cybercrime, women | — | — | — | — | — |
| **mutation.land.gov.bd**, **ldtax.gov.bd** | Both load (200), but are **transactional service apps**, not content. No Q&A. | — | Bangla | JS app | — | — | — |

**Honest summary of (b): the government portals contributed 9 questions, none of them legal.** This category is effectively a dead end for citizen questions.

### (c) NGO legal-aid material

| Source & URL | ~Questions | Domains | Lang | Structure | robots.txt / ToS | PII | Q/A separable |
|---|---|---|---|---|---|---|---|
| **BLAST FAQ**<br>[legal-aid FAQ](https://blast.org.bd/programmes/legal-aid/frequently-asked-questions-faq/) | **0.** The page returns HTTP 200 and 140 KB of HTML, but fetching the page body through the WP REST API (`/wp-json/wp/v2/pages?slug=frequently-asked-questions-faq`) shows the entire content is the placeholder string **`Your Content Goes Here`**. **The FAQ is an empty template.** | — | English | WordPress, open REST API | robots.txt: `User-agent: *` / `Disallow:` (empty ⇒ **everything allowed**), sitemap declared | — | — |
| **BLAST site overall** | 335 posts (`X-WP-Total: 335`). Categories are Press Release (170), See the Latest (160), BLAST in News (94), PIL judgements (9), Land law (1), Violence Against Women (1). **These are press releases and case notes, not citizen questions.** | land, VAW, PIL, adivasi rights | English mostly | WordPress REST | as above | Case notes may name real litigants | N/A |
| **Ain o Salish Kendra**<br>[askbd.org](https://www.askbd.org/ask/) | 4,021 posts (`X-WP-Total: 4021`) — but categories are Acid Violence (109), Border Violence (153), Advocacy (159), Bulletin (64), Annual Report (18)… **human-rights monitoring output, not a Q&A column.** No FAQ/প্রশ্নোত্তর section found. | VAW, human rights | English/Bangla | WordPress REST | robots.txt: `User-agent: *` / `Crawl-Delay: 20` — **20-second crawl delay is the only restriction, and it is binding on any crawl you run.** | Reports name victims | N/A |
| **BRAC HRLS** | **Unreachable at the documented path.** `brac.net/program/hrls/` returns **HTTP 404**. Not located in this pass. | — | — | — | — | — | — |
| **BNWLA, Manusher Jonno Foundation, Bandhu, Nagorik Uddyog** | **Not investigated in this pass** — deprioritised after BLAST and ASK both showed the category yields press releases rather than Q&A. Flagged as an unverified gap. | — | — | — | — | — | — |
| **Law Helpline BD**<br>[lawhelplinebd.com](https://lawhelplinebd.com/) | **77 posts** (`X-WP-Total: 77`). Not citizen questions — plain-Bangla explainer articles. Useful for its **category taxonomy**: অন্যন্য আইন 21, দন্ডবিধি 17, সাক্ষ্য আইন 16, চাকরির খবর 11, নারী-শিশু 10, পারিবারিক 7, ভোক্তা অধিকার 6, মাদক 6, ফৌজদারী 4, **সড়ক আইন 2**. | consumer, road-transport, family, women-children, criminal | Bangla | WordPress, open REST API | robots.txt: `User-agent: *` / `Disallow:` (**allow all**) | None | N/A (articles) |
| **Lawyers Club Bangladesh**<br>[lawyersclubbangladesh.com](https://lawyersclubbangladesh.com/) · category `দৈনন্দিন জীবনে আইন` (id **756**) | **382 posts** in that category (22,981 site-wide). Sampled 100 titles via `/wp-json/wp/v2/posts?categories=756`: **26 end in `?` (26%) ⇒ ~99 question-titled posts.** | **Land is dominant** — নামজারি, খতিয়ান সংশোধন, অগ্রক্রয়/প্রিয়েমশন, বাটোয়ারা, দখল; plus cheque-dishonour, দেনমোহর, নারী ও শিশু নির্যাতন, medical negligence, UD cases | Bangla | WordPress with **open REST API** — `/wp-json/wp/v2/posts?categories=756` returns clean JSON | robots.txt: `User-agent: *` / `Disallow:` (**allow all**), sitemap declared | None (editorial) | N/A — these are **editorial explainers, not citizen letters**. The question-form *titles* are usable as queries; the body is not a citizen's words. Label them distinctly in the dataset. |

### (d) Facebook — legality and feasibility assessment (nothing was scraped)

**Named pages/groups identified** (via web search; follower counts are as reported by search results, **not independently verified by me**, since verifying would require loading Facebook):

| Page/Group | URL | Reported size |
|---|---|---|
| BLAST (official) | https://www.facebook.com/BLASTBangladesh/ | ~44,929 likes |
| Legal Advice BD | https://www.facebook.com/LegalAdviceBangladesh/ | ~4,900 likes |
| Law Help BD | https://www.facebook.com/lawhelpbd/ | not reported |
| Law & Our Rights, The Daily Star | https://www.facebook.com/TDSlawdesk/ | not reported |
| ভূমি সেবা / Land Service (official land.gov.bd page) | https://www.facebook.com/land.gov.bd/ | not reported |
| Cyber Support for Women (Bangladesh Police) | https://www.facebook.com/cybersupport.women | not reported |
| Free Legal Advice Bangladesh (Land+Civil+Criminal…) — **group** | https://m.facebook.com/groups/1909373159375744/ | not reported |
| Legal Jobs In Bangladesh — group | https://www.facebook.com/groups/121565125217880/ | not reported (off-topic: jobs) |

Note that most of these are **organisational broadcast pages**, not places where citizens post questions. The one genuine citizen-question venue found is the *Free Legal Advice Bangladesh* group — and groups are precisely the case the API no longer supports (below).

#### (i) Automated scraping — **prohibited**

Meta's Terms of Service state:

> "You may not access or collect data from our Products using automated means (without our prior permission) or attempt to access data that you do not have permission to access"
> — https://www.facebook.com/terms.php

The dedicated Automated Data Collection Terms are more explicit:

> "You will not engage in Automated Data Collection without first obtaining Meta's express written permission or in any manner that is not explicitly authorized by Meta."

…and "acceptance of these Terms alone does not constitute the required written permission."
— https://www.facebook.com/apps/site_scraping_tos_terms.php

That page defines the prohibited activity broadly:

> "collection of data from Meta Company Products via the use of automated or programmatic tools capable of navigating or indexing the surface-layer of the World Wide Web. These tools may include but are not limited to web scrapers, bots, robots, spiders, crawlers, user-agents."

and restricts even *permitted* collection to two purposes only — search-engine results or URL previews:

> "All other Uses are prohibited, including, but not limited to, transferring, selling, licensing or sublicensing Collected Data and data derived from Collected Data to any third party."

Building an NLP dataset is not one of the two permitted uses. **Verdict: do not scrape Facebook. There is no student-project carve-out.**

#### (ii) Graph API — **closed for groups, gated for pages**

The Graph API v19.0 changelog states that `publish_to_groups`, `groups_access_member_info` and the **Groups API** are deprecated, and:

> "These permissions, features, and abilities will be removed April 22, 2024."
> — https://developers.facebook.com/docs/graph-api/changelog/version19.0/

So reading group posts programmatically is **no longer possible at all**, regardless of permission. For public *Pages*, content access requires the Page Public Content Access feature, which is gated behind App Review and business verification — not realistically obtainable for a student project. (I verified the Groups removal directly; I did not separately verify PPCA's current review requirements, so treat that clause as reported rather than confirmed.)

#### (iii) A human manually reading public posts and rewriting by hand — **the only viable path, with conditions**

This is not automated collection, so the clauses above do not bite. It is ordinary reading. But the ethical constraints in Dhara's own spec still apply, and they bite hard here:

- Facebook legal questions are **the most PII-dense source in this entire report** — real names attached to real accounts, photographs, locations, and frequently descriptions of ongoing family disputes, abuse, or criminal matters. A Prothom Alo letter has been through an editor; a Facebook post has not.
- A hand-rewritten paraphrase that preserves the *register* but discards the identifying narrative is defensible. Verbatim copying is not.
- Volume is inherently low — this is human-speed work, realistically tens of questions, not hundreds.

**Recommendation:** use Facebook only as a *register calibration* exercise — a human reads posts, writes down how citizens actually phrase land and consumer complaints, and those phrasings inform the synthetic question generator. Do not build dataset rows from it directly, and record the decision in `DECISIONS.md`.

### (e) Q&A platforms

| Platform | Verdict | Evidence |
|---|---|---|
| **Quora / bn.quora.com** | **Do not use.** robots.txt opens with an explicit blanket notice: "All crawlers and bots, regardless of whether or not they are specified below, are **strictly prohibited from using Quora platform content for the purposes of training AI models or similar machine learning systems**, except where explicit prior permission has been granted by Quora through a contractual licensing agreement." It further disallows `/*/questions`, `/*/all_questions`, `/*/top_questions`. — https://www.quora.com/robots.txt |
| **Reddit (r/bangladesh, r/dhaka)** | **Do not scrape.** robots.txt is a total block: `User-agent: *` / `Disallow: /`, pointing to Reddit's Public Content Policy. Reddit does run a researcher programme (r/reddit4researchers) which is the only sanctioned route. — https://www.reddit.com/robots.txt |
| **Bissoy.com** | **No longer relevant.** robots.txt is permissive (`Allow: /`) and the site even publishes an `llms.txt`. But that `llms.txt` says: "Bissoy is a Bangladesh **health platform** with verified doctor profiles, online appointments, live patient queue, medicine information, medical Q&A, and MCQ exam preparation." Its listed sections are doctors, hospitals, medicines, health Q&A, MCQs — **no law topic**. Probing `/topic/আইন`, `/t/আইন-কানুন`, `/category/law` all returned 410/404. Bissoy has pivoted away from general Q&A. — https://www.bissoy.com/llms.txt |
| **ask.com.bd** | **Unreachable.** `curl` to `https://ask.com.bd/robots.txt` returned exit code 000 (no response). Could not verify whether the site still exists. |

---

## 3. Verdicts: safe to use / use with care / do not use

### Safe to use

| Source | Reason (with the text I read) |
|---|---|
| **Lawyers Club Bangladesh** | robots.txt is `User-agent: *` / `Disallow:` — an empty Disallow, which means *nothing is disallowed*. Open WP REST API is a deliberate public interface. No PII in editorial articles. |
| **Law Helpline BD** | Identical robots.txt (`Disallow:` empty). Editorial content, no PII. |
| **BLAST** | Identical robots.txt (`Disallow:` empty). But the FAQ is empty, so there is little to take. |
| **`momahadi/bangladesh-legal-qa-dataset`** | **CC-BY-4.0** — explicitly redistributable with attribution. |
| **`seyam2023/bangladesh_law`** | **Apache-2.0.** |
| **`sakhadib/bangladesh-legal-acts-dataset`** (Kaggle) | **CC BY-SA 4.0** — note ShareAlike is viral if you redistribute a derived corpus. |
| **`rubelhossain739/bangladeshi-law-qa`** (Kaggle) | **MIT.** |

### Use with care

| Source | Reason |
|---|---|
| **Prothom Alo** (both columns) | The Terms of Use served at https://www.prothomalo.com/terms-of-use are restrictive. Verbatim: "Site readers/visitors are required to use … services only for lawful means and **for read-only purposes**"; "Users are also prevented from **making any derivative work** from the content"; "users may use available services for **personal, private and non-commercial purposes only**". A research dataset is arguably a derivative work. **Oddity worth recording: this page is written throughout for "Haal Fashion", a Prothom Alo group brand, not for Prothom Alo itself** — so its applicability to prothomalo.com content is genuinely ambiguous. robots.txt does *not* disallow the paths used. Governing law is Bangladesh; disputes go to arbitration in Dhaka. **Conclusion:** academic, non-commercial, fair-dealing-style use is defensible; **do not redistribute the raw article text**. Commit only derived question text with PII stripped, plus source URL and crawl date. Honour a 1.5–2 s delay per the project's own scraping-conduct rule. |
| **Ajker Patrika** | robots.txt permits the paths. I did **not** locate a separate terms-of-use page, so the copyright position is unassessed — assume standard reserved rights and apply the same non-redistribution rule as Prothom Alo. |
| **The Daily Star** | robots.txt permits it. Terms of use not read in this pass — **unverified**. English-only, so lower value for the lexical-gap claim anyway. |
| **Ain o Salish Kendra** | robots.txt sets `Crawl-Delay: 20`. Any crawler **must** sleep 20 s between requests. At 4,021 posts that is ~22 hours — and the content isn't Q&A, so it isn't worth it. |
| **Jagonews24** | robots.txt sets `Content-Signal: search=yes,`**`ai-train=no`**`,use=reference`, framed as "ANY RESTRICTIONS EXPRESSED VIA CONTENT SIGNALS ARE EXPRESS RESERVATIONS OF RIGHTS UNDER ARTICLE 4 OF THE EUROPEAN UNION DIRECTIVE 2019/790". Only 10 questions are at stake. **Given the project fine-tunes models, `ai-train=no` is a direct conflict — recommend simply skipping this source.** |
| **Government portals (nlaso, minlaw, dncrp, dife)** | **No robots.txt exists** on any of them (the URL returns the site's HTML 404 page). Absence of a robots file is not permission. Government works are typically freely usable, but I found no terms page stating so. Moot in practice — they contain no citizen questions. |
| **`bipinsaha/bangla-law-qna` & `AshfakUzzaman/bangla-law-qna`** | **CC-BY-NC-4.0** — non-commercial only, which fits an academic project, but the licence is applied by a third party to *Prothom Alo's* content, which they were arguably not entitled to relicense. Cite it as related work; be cautious about treating the licence as clean title. |
| **`ifathjeba/banglalaw-qa`, `ifathjeba/legalqa`, `nahid002345/bangladesh-law-dataset`** | Kaggle licence field reads **"Unknown"** for all three. No usable licence ⇒ cannot safely redistribute or build on. |

### Do not use

| Source | Reason |
|---|---|
| **Facebook (automated scraping)** | "You may not access or collect data from our Products using automated means (without our prior permission)" (ToS) and "You will not engage in Automated Data Collection without first obtaining Meta's express written permission" (Automated Data Collection Terms). Dataset construction is not among the two permitted uses. |
| **Facebook Groups (Graph API)** | Not a policy choice — technically removed. "These permissions, features, and abilities will be removed April 22, 2024" (Graph API v19.0 changelog). |
| **Quora** | "strictly prohibited from using Quora platform content for the purposes of training AI models or similar machine learning systems" (robots.txt). Unambiguous, and Dhara trains models. |
| **Reddit** | `User-agent: *` / `Disallow: /` (robots.txt). Total crawl prohibition. |
| **`bangla-legal-bot/Fsdyrdyy`** (HF) | **Empty repository** — the only file is `.gitattributes`. No data, no licence. |

---

## 4. Existing datasets

This section matters most: some of this work overlaps Dhara directly.

### Hugging Face

| Dataset | Size | Licence | What it actually contains | Relevance |
|---|---|---|---|---|
| [`momahadi/bangladesh-legal-qa-dataset`](https://huggingface.co/datasets/momahadi/bangladesh-legal-qa-dataset) | **2,165 QA records** (1K–10K); 270 downloads, 2 likes | **CC-BY-4.0** | Bilingual bn/en. Ships a **statutory corpus** (`law-corpus/bangla|english/`) for Penal Code, CrPC, CPC, Evidence Act, Limitation Act, Specific Relief Act + 3 schedules, and **`qa-splits/`** with `single_hop`, `advanced_selection`, `bar_exam_style` in both languages. I downloaded `single_hop_bangla.json`: **493 items**, each with `Question`, `Possible Sections`, **`Relevant Section`** (e.g. `The Penal Code, 1860, Section 323`), `IRAC_Reasoning`, `Keywords`, `Difficulty`. | **This is the closest prior work to Dhara's task** — it is literally question → statutory section. **But the register is different:** questions are LLM-generated and carry a literal `"Scenario-based: "` prefix, e.g. `Scenario-based: প্রতিবেশী আমাকে চড় মেরেছেন। এর জন্য সর্বোচ্চ কতদিন তার জেল হতে পারে?`. Clean, well-formed, textbook Bangla — **not colloquial citizen phrasing**. Dhara's lexical-gap claim survives, and this dataset is the ideal *contrast* condition. Domains are criminal/procedure — **complementary** to Dhara's family/land/labour/consumer. |
| [`momahadi/bangladesh-bar-council-exam-dataset`](https://huggingface.co/datasets/momahadi/bangladesh-bar-council-exam-dataset) | **400 questions** (2022 & 2023 Bar Council exams, bn + en) | **"other"** (carries its own source-rights notice) | `evaluation/bar_exam_{2022,2023}_{bangla,english}.json` | Professional exam register. Useful as an out-of-domain eval, not as citizen questions. |
| [`bipinsaha/bangla-law-qna`](https://huggingface.co/datasets/bipinsaha/bangla-law-qna) | **63 rows** (verified by download) | **CC-BY-NC-4.0** | Single file `Law-QnA-Prothom-Alo-63.csv`, columns **`Problem`, `Solution`**. Verified content: genuine long-form colloquial Prothom Alo reader letters with the lawyer's statutory answer (e.g. a letter about a brother's wife and threats of suicide → answer citing **দণ্ডবিধি ১৮৬০-এর ৩০৬ ধারা** and **সাক্ষ্য আইন ১৮৭২**). | **Someone has already extracted Prothom Alo law Q&A.** 63 rows, and the `Solution` field already names ধারা — meaning gold labels are partly derivable. Small, but it is direct prior art on Dhara's exact source and must be cited. Also a useful cross-check on your own parser. |
| [`AshfakUzzaman/bangla-law-qna`](https://huggingface.co/datasets/AshfakUzzaman/bangla-law-qna) | 63 rows | CC-BY-NC-4.0 | **Same file name and size as the above — a duplicate/fork.** Do not count twice. | — |
| [`Sadatsami/bangladesh-law-professional`](https://huggingface.co/datasets/Sadatsami/bangladesh-law-professional) | Card claims 1K–10K | Apache-2.0 | Card describes an Alpaca-style bn/en instruction dataset. **But the data files are effectively empty**: `benchmark.jsonl` downloads as **26 bytes**, `train.jsonl` as **132 bytes / 3 lines**. | **The README oversells it.** Not usable. Good example of why you check bytes, not cards. |
| [`seyam2023/bangladesh_law`](https://huggingface.co/datasets/seyam2023/bangladesh_law) | 35 downloads | **Apache-2.0** | `bd_laws_translated_05022022.csv`, `bdlaws_ds_formatted.csv/.jsonl` | A bdlaws corpus dump — **corpus-side**, not questions. Possible cross-check for Dhara's own scrape. |
| [`anisafifi/bd-laws`](https://huggingface.co/datasets/anisafifi/bd-laws) | 10K–100K | not declared on the API | `bd_laws_all.jsonl` | Another bdlaws dump. Corpus-side. |
| [`ani4434/explAIN-bangla-legal-vectordb`](https://huggingface.co/datasets/ani4434/explAIN-bangla-legal-vectordb) | <1K | none | A prebuilt Chroma DB + BM25 pickle | Someone else's index, not a dataset. Low value. |
| `bangla-legal-bot/Fsdyrdyy` | — | none | **Empty** (only `.gitattributes`) | Nothing. |

### Kaggle

| Dataset | Size | Licence | Note |
|---|---|---|---|
| [`sakhadib/bangladesh-legal-acts-dataset`](https://www.kaggle.com/datasets/sakhadib/bangladesh-legal-acts-dataset) | 24.5 MB | **CC BY-SA 4.0** | Usability 1.0. Mirrors the [GitHub repo](https://github.com/sakhadib/Bangladesh-Legal-Acts-Dataset), described as "All of Bangladesh's laws and acts published by Bangladesh Govt, structured in JSON format" — reported as **1,484+ acts scraped from the bdlaws portal**. **Directly relevant to Dhara's corpus phase** — worth diffing against your own scrape. GitHub licence field is `NOASSERTION`, so the CC BY-SA on Kaggle is the clearer claim. |
| [`rubelhossain739/bangladeshi-law-qa`](https://www.kaggle.com/datasets/rubelhossain739/bangladeshi-law-qa) | 11.7 MB | **MIT** | "Bangladeshi Law QA LLM fine tune". Cleanest licence of the Kaggle QA sets. Contents not sampled. |
| [`ifathjeba/banglalaw-qa`](https://www.kaggle.com/datasets/ifathjeba/banglalaw-qa) | 9.2 MB | **Unknown** | Updated 2026-08-03. Unusable licence. |
| [`ifathjeba/legalqa`](https://www.kaggle.com/datasets/ifathjeba/legalqa) | 1.1 MB | **Unknown** | Unusable licence. |
| [`nahid002345/bangladesh-law-dataset`](https://www.kaggle.com/datasets/nahid002345/bangladesh-law-dataset) | 16.5 MB | **Unknown** | From 2022. |
| [`abdullaharean/bqad2025`](https://www.kaggle.com/datasets/abdullaharean/bqad2025) | 0.37 MB | **CDLA-Sharing-1.0** | Titled "SOMAJGYAAN". Usability 0.82. Bangla QA, social-knowledge oriented; legal content unconfirmed. |
| [`ikbalnayem/bangladesh-constitution-law`](https://www.kaggle.com/datasets/ikbalnayem/bangladesh-constitution-law) | 0.09 MB | not checked | Constitution text — relevant to Dhara's fifth cross-cutting source. |
| [`afzalhosenmandal/bangladesh-legal`](https://www.kaggle.com/datasets/afzalhosenmandal/bangladesh-legal) | 0.04 MB | not checked | "Bangladesh-legal-service". Tiny. |

### Papers

| Paper | Venue | Relevance |
|---|---|---|
| **Mina: A Multilingual LLM-Powered Legal Assistant Agent for Empowering Access to Justice in Bangladesh** — Wasi, Faisal, Islam, Parvez. [aclanthology.org/2026.findings-acl.1295](https://aclanthology.org/2026.findings-acl.1295/) | **Findings of ACL 2026** | RAG + chain-of-tools Bengali legal assistant, evaluated by law faculty on the 2022/2023 Bar Council exams, scoring 75–80%. **Directly adjacent to Dhara and the strongest related work.** Abstract states the motivation almost identically ("complex legal language, procedural opacity, and high costs"; "Existing AI legal assistants lack Bengali-language support"). **No public dataset is linked from the landing page.** Dhara's differentiators remain: section-level retrieval with citation + abstention, real citizen register, and a measured lexical gap. |
| **Do Small Models Use the Law You Give Them? Context-Injected Fine-Tuning for Legal QA in Bangladesh** — Mahadi et al. [arXiv:2607.23446](http://arxiv.org/abs/2607.23446v1) (2026-07-26) | arXiv | The paper behind `momahadi/bangladesh-legal-qa-dataset`. 2,165 bilingual QA from six Bangladeshi acts; fine-tunes Qwen3.5 at 0.8B/2B/4B; evaluates with **no retrieval, BM25, and FAISS**. Notably reports a **null result at 4B** ("the 4B model has no detectable net gain: Bangla improves while several English conditions regress") — a useful precedent for Dhara's own honest-negative-results stance. |
| **LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval** — Kabir, Sultan, Rahman, Amin, Momen, Mohammed, Rahman. [arXiv:2504.16121](http://arxiv.org/abs/2504.16121v1) (2025-04-19) | arXiv | Bilingual bn/en QA over **Bangladesh Police Gazettes**. Same hybrid-retrieval architecture family as Dhara. Different corpus (gazettes, not acts) and no citizen questions. |
| **BanglaQuAD: A Bengali Open-domain Question Answering Dataset** — [arXiv:2410.10229](https://arxiv.org/pdf/2410.10229) | arXiv | 30,808 QA pairs from Bengali Wikipedia, written by native speakers. **Not legal**, but the only large native-authored Bengali QA set — useful as a general-domain control or for pretraining sanity checks. |
| **A visual search engine for Bangladeshi laws** — [arXiv:1711.05233](https://arxiv.org/pdf/1711.05233) | arXiv (2017) | Early Bangladeshi legal IR. Historical related work. |

### GitHub

| Repo | Note |
|---|---|
| [sakhadib/Bangladesh-Legal-Acts-Dataset](https://github.com/sakhadib/Bangladesh-Legal-Acts-Dataset) | 1,484+ acts from the bdlaws portal as JSON. Licence `NOASSERTION`. 2 stars, last updated 2026-07-11. Corpus-side. |
| [sabbirhossainujjal/Awesome_Bangla_Datasets](https://github.com/sabbirhossainujjal/Awesome_Bangla_Datasets) | Curated index of Bangla datasets — worth a manual pass for anything missed here. |
| [FaishalRudro/Bangladesh-Legal-Advisor-RAG_GROQ](https://github.com/FaishalRudro/Bangladesh-Legal-Advisor-RAG_GROQ), [tonmoyjoy/Bangladesh-Legal-Chatbot](https://github.com/tonmoyjoy/Bangladesh-Legal-Chatbot), [Naawshin/AinAware---A-Bangladeshi-Paralegal-Chatbot](https://github.com/Naawshin/AinAware---A-Bangladeshi-Paralegal-Chatbot) | Three small (1–2 star) student RAG chatbots over Bangladeshi law. No datasets of citizen questions; evidence the application idea is crowded but the *evaluation* is not. |

---

## 5. Gaps

Domain coverage across everything found:

| Domain | Coverage | Best source | Verdict |
|---|---|---|---|
| **Family** | **Strong** | Prothom Alo (dominant), Ajker Patrika | Solved — arguably over-represented |
| **Constitutional / rights** | Adequate | Prothom Alo, Daily Star | OK |
| **Criminal procedure** | **Strong** but from synthetic data | `momahadi` (Penal Code, CrPC, Evidence, Limitation) | OK for contrast; thin on real citizen questions |
| **Cybercrime** | Moderate | Prothom Alo (per its 2023 round-up), Daily Star (cyber-bullying, email hacks) | OK |
| **Women & children** | Moderate | Ajker Patrika, lawhelplinebd (10 posts), LCB (277 posts) | OK |
| **Land** | **Editorial only — no citizen questions** | LCB `দৈনন্দিন জীবনে আইন` (382 posts, land-heavy) | **Gap** |
| **Labour** | **Thin, and English-only** | Daily Star Your Advocate (retrenchment, gratuity, CBA) | **Gap** |
| **Consumer** | **Very thin** | lawhelplinebd ভোক্তা অধিকার (**6 posts**); DNCRP portal has no FAQ | **Gap** |
| **Road transport** | **Almost nothing** | lawhelplinebd সড়ক আইন (**2 posts**) | **Severe gap** |
| **Tenancy** | **Nothing at all** | — | **Severe gap** |

### What I'd suggest instead

1. **Land — mine the LCB titles, then generate.** The 382 `দৈনন্দিন জীবনে আইন` posts are the single best land-language resource found, and 26% are already phrased as plain-Bangla questions. They are editorial, not citizen-written, so **label them as a distinct provenance class** (`source_type: editorial_title`) and keep them out of any claim about "real citizen questions". Their real value is as a **vocabulary seed** for the synthetic generator: নামজারি, খতিয়ান, অগ্রক্রয়, বাটোয়ারা, দাগ, ভোগদখল are exactly the colloquial land terms the generator will otherwise never produce.

2. **Tenancy and road-transport — accept there is no source and say so.** No column, portal or dataset found carries these. Options, in order of honesty: (a) drop them from the reported domain list and state why; (b) generate synthetic questions from the Premises Rent Control Act and the Road Transport Act 2018 and **mark them synthetic in the gold set**, reporting real-question and synthetic-question metrics separately. **Do not silently blend them** — that is exactly the "changing what is measured" failure the project charter forbids.

3. **Consumer — go to the complaint form, not the FAQ.** DNCRP has no FAQ page, but the live host is [dncrp.gov.bd](https://www.dncrp.gov.bd/) (not `dncrp.portal.gov.bd`, which is decommissioned). Its complaint-submission workflow and any published complaint summaries are the most likely source of real consumer phrasing; worth a dedicated follow-up pass.

4. **Labour — DIFE plus the Daily Star.** DIFE has no FAQ, but Daily Star Your Advocate has genuine labour queries. They are English. Translating them into colloquial Bangla changes the register and **would contaminate the lexical-gap measurement** — if you use them, keep them in a separate English-origin split.

5. **Facebook as register calibration only** (see §(d)(iii)) — a human reads the *Free Legal Advice Bangladesh (Land+Civil+Criminal)* group to learn how citizens phrase land and tenancy problems, and writes down the phrasings. This is the only realistic route to authentic land/tenancy register, and it is compatible with Meta's terms because no automated collection occurs.

6. **Don't rebuild what exists.** Before scraping the bdlaws portal, diff against `sakhadib/bangladesh-legal-acts-dataset` (CC BY-SA 4.0, 1,484 acts) and `seyam2023/bangladesh_law` (Apache-2.0). And pull `bipinsaha/bangla-law-qna` (63 Prothom Alo Q&A) early — it is a free correctness check on your own Prothom Alo parser, which is the hardest parsing job in this report.

### Two engineering warnings

- **Prothom Alo Q/A segmentation is the main technical risk in this plan.** The migrated articles have lost their `*` delimiters and render as a single undifferentiated `<p>`. Verified on `lifestyle/পাঠকের-উকিল-90`: 5,299 characters, **zero** `*` markers, question and answer running together with only a signature (`সুমাচট্টগ্রাম`, `নাম প্রকাশে অনিচ্ছুক`) marking the seam. Budget for a signature-based segmenter and manual verification. Ajker Patrika, with its clean `উত্তর:` marker, is the better place to start.
- **PII stripping is non-optional and non-trivial here.** Prothom Alo letters carry real first names and districts inline in running text, not in a separate field. The project's ethics rule (strip names, phone numbers, NID, addresses) requires a Bangla NER pass or manual review, not a regex.

---

## 6. Sources — every URL fetched

**robots.txt / terms**
- https://www.prothomalo.com/robots.txt
- https://www.prothomalo.com/terms-of-use
- https://www.prothomalo.com/privacy-policy
- https://www.jugantor.com/robots.txt
- https://www.kalerkantho.com/robots.txt
- https://www.bd-pratidin.com/robots.txt
- https://www.ittefaq.com.bd/robots.txt
- https://www.samakal.com/robots.txt (301, no body)
- https://www.thedailystar.net/robots.txt
- https://www.dhakatribune.com/robots.txt
- https://www.bdnews24.com/robots.txt (302, no body)
- https://www.jagonews24.com/robots.txt
- https://www.banglanews24.com/robots.txt
- https://www.ajkerpatrika.com/robots.txt
- https://www.askbd.org/robots.txt
- https://www.blast.org.bd/robots.txt
- https://lawhelplinebd.com/robots.txt
- https://lawyersclubbangladesh.com/robots.txt
- https://www.quora.com/robots.txt
- https://bn.quora.com/robots.txt
- https://www.reddit.com/robots.txt
- https://www.bissoy.com/robots.txt
- https://www.bissoy.com/llms.txt
- https://ask.com.bd/robots.txt (no response)
- https://www.land.gov.bd/robots.txt
- https://nlaso.gov.bd/robots.txt (404 HTML)
- https://minlaw.gov.bd/robots.txt (404 HTML)
- https://dife.gov.bd/robots.txt (404 HTML)
- https://dncrp.portal.gov.bd/robots.txt (domain decommissioned)
- https://www.facebook.com/terms.php
- https://www.facebook.com/apps/site_scraping_tos_terms.php
- https://developers.facebook.com/docs/graph-api/changelog/version19.0/

**Prothom Alo**
- https://www.prothomalo.com/lifestyle/পাঠকের-প্রশ্ন-আইন
- https://www.prothomalo.com/lifestyle/পাঠকের-উকিল-36
- https://www.prothomalo.com/api/v1/advanced-search (multiple queries)
- https://www.prothomalo.com/api/v1/stories-by-slug (articles -6, -7, -22, -24, -31, -34, -36, -49, -60, -67, -69, -70, -72, -75, -78, -85, -88, -90, -91)
- https://www.prothomalo.com/api/v1/stories , /api/v1/search , /api/v1/collections/* (probed)

**Other newspapers**
- https://www.ajkerpatrika.com/women/legal-advice (pages 1–6)
- https://www.ajkerpatrika.com/lifestyle/legal-advice
- https://www.ajkerpatrika.com/women/legal-advice/ajpvh8oifylnq
- https://www.thedailystar.net/ds/law-our-rights
- https://www.thedailystar.net/ds/law-our-rights/your-advocate (pages 0–3)
- https://www.thedailystar.net/law-our-rights/your-advocate/news/pandemic-retrenchment-and-labour-law-2209706
- https://archive.thedailystar.net/law/2004/04/01/queriess.htm (no response)
- https://www.jagonews24.com/topic/আইনি-পরামর্শ
- https://www.kalerkantho.com/online/court , /online/law-and-justice
- https://www.jugantor.com/todays-paper/law-and-justice
- https://www.ittefaq.com.bd/law-and-justice (404)
- https://samakal.com/law-and-justice
- https://www.dhakatribune.com/topic/legal-advice (404)
- https://www.bd-pratidin.com/law-and-justice (404)

**Government**
- https://nlaso.gov.bd/ , /faq , /site/view/faq , /site/page/faq , /site/faq , /site/page/legal-aid
- https://minlaw.gov.bd/ (+ FAQ probes)
- https://dife.gov.bd/ (+ FAQ probes)
- https://www.dncrp.gov.bd/ , https://dncrp.gov.bd/ (+ FAQ probes)
- https://land.gov.bd/ , https://land.gov.bd/faq
- https://mutation.land.gov.bd/
- https://ldtax.gov.bd/
- https://judiciary.gov.bd/ , https://dhaka.judiciary.gov.bd/
- https://www.police.gov.bd/ , https://cybersupport.police.gov.bd/ (no response)

**NGO / legal portals**
- https://www.blast.org.bd/
- https://blast.org.bd/programmes/legal-aid/frequently-asked-questions-faq/
- https://blast.org.bd/wp-json/wp/v2/{posts,pages,categories}
- https://www.askbd.org/ask/ , https://www.askbd.org/wp-json/wp/v2/{posts,categories}
- https://www.brac.net/program/hrls/ (404)
- https://lawhelplinebd.com/ , /sitemap_index.xml , /wp-json/wp/v2/{posts,categories}
- https://lawyersclubbangladesh.com/ , /wp-json/wp/v2/{posts,categories}

**Datasets & papers**
- https://huggingface.co/api/datasets?search=… (multiple)
- https://huggingface.co/datasets/momahadi/bangladesh-legal-qa-dataset (+ README, qa-splits/single_hop_bangla.json)
- https://huggingface.co/datasets/momahadi/bangladesh-bar-council-exam-dataset
- https://huggingface.co/datasets/bipinsaha/bangla-law-qna (+ Law-QnA-Prothom-Alo-63.csv, README)
- https://huggingface.co/datasets/AshfakUzzaman/bangla-law-qna
- https://huggingface.co/datasets/Sadatsami/bangladesh-law-professional (+ README, train.jsonl, benchmark.jsonl)
- https://huggingface.co/datasets/seyam2023/bangladesh_law
- https://huggingface.co/datasets/anisafifi/bd-laws
- https://huggingface.co/datasets/ani4434/explAIN-bangla-legal-vectordb
- https://huggingface.co/datasets/sakhadib/Bangladesh-Legal-Acts-Dataset (HTTP 429 — rate-limited, metadata unread)
- https://www.kaggle.com/api/v1/datasets/list?search=… (multiple)
- https://api.github.com/repos/sakhadib/Bangladesh-Legal-Acts-Dataset
- https://api.github.com/search/repositories?q=… (multiple)
- http://export.arxiv.org/api/query (multiple)
- https://aclanthology.org/2026.findings-acl.1295/ and .pdf
- https://arxiv.org/abs/2504.16121 , https://arxiv.org/abs/2607.23446 , https://arxiv.org/pdf/2410.10229 , https://arxiv.org/pdf/1711.05233
- https://aclanthology.org/search/?q=… (JS-rendered; returned no parseable results)

**Facebook (identification only — no content fetched)**
- https://www.facebook.com/BLASTBangladesh/
- https://www.facebook.com/LegalAdviceBangladesh/
- https://www.facebook.com/lawhelpbd/
- https://www.facebook.com/TDSlawdesk/
- https://www.facebook.com/land.gov.bd/
- https://www.facebook.com/cybersupport.women
- https://m.facebook.com/groups/1909373159375744/
- https://www.facebook.com/groups/121565125217880/

---

## Things I could not verify

- **`samakal.com` and `bdnews24.com` robots.txt** — both return Cloudflare 301/302 with an empty body. Their crawl policy is unknown.
- **`ask.com.bd`** — no HTTP response at all (curl exit 000). Unknown whether the site still exists.
- **`cybersupport.police.gov.bd`** — no connection. The Cyber Support for Women unit could not be assessed as a web source.
- **`archive.thedailystar.net`** old `queries.htm` pages — no response. These 2004-era pages appeared in search results and reportedly carry multiple reader queries per page; if reachable later they could add to the Daily Star count.
- **`sakhadib/Bangladesh-Legal-Acts-Dataset` on Hugging Face** — HTTP 429 rate limit; licence and file list read from the Kaggle mirror and GitHub instead.
- **Terms of use for Ajker Patrika and The Daily Star** — no terms page located/read. Copyright position assumed reserved.
- **BNWLA, Manusher Jonno Foundation, Bandhu, Nagorik Uddyog** — not investigated; deprioritised after BLAST and ASK both proved to be press-release archives rather than Q&A sources.
- **Facebook follower/member counts** — taken from search-result snippets, not verified on-platform.
- **Page Public Content Access review requirements** — reported from general knowledge of the gating, not confirmed against a primary Meta page in this pass. The Groups API removal *was* confirmed primarily.
