"""Batch 22: money_recovery domain, Negotiable Instruments Act 1881
(money_recovery_46) - cheque-dishonour cluster and core definitions. This is
the most common citizen money-recovery collision in Bangladesh (section 138
"চেক ডিজঅনার" criminal cases), and this Act is entirely in English (colonial
statute, never translated - see DECISIONS.md 2026-08-22/25 on the
language gap this corpus measures). These 18 chunk_ids were never in
author_queue_v1/v2 at all (worth investigating separately why the earlier
queue-build step skipped them - not chased down in this session), so authored
directly against corpus_v1.jsonl text rather than off the queue file. A
Bangla citizen question against an English-only provision has near-zero
token overlap by construction (disjoint scripts under aggressive()
tokenization), so this batch needed none of the title/body word-avoidance
work the Bangla-Bangla batches required - only genuine accuracy against the
English text and human-band length.
"""

BATCH = [
    ("money_recovery_46_s4", "কাউকে ধার দেওয়া টাকা নির্দিষ্ট সময়ে ফেরত দেওয়ার লিখিত ও শর্তহীন প্রতিশ্রুতি দিয়ে সই করা কাগজকে আইনে কী বলা হয়?"),
    ("money_recovery_46_s5", "একজন আরেকজনকে নির্দিষ্ট তারিখে তৃতীয় কোনো ব্যক্তিকে টাকা দিতে লিখিতভাবে নির্দেশ দিলে, সেই কাগজটাকে ব্যাংকিং ভাষায় কী বলে?"),
    ("money_recovery_46_s6", "একটা ব্যাংকের ওপর দেওয়া, যেকোনো সময় টাকা তোলা যায় এমন লিখিত নির্দেশকে কেন আলাদাভাবে চেক বলা হয়?"),
    ("money_recovery_46_s7", "চেক লিখে দেওয়া ব্যক্তি আর যে ব্যাংক থেকে টাকা দেওয়ার কথা, এই দুই পক্ষকে আইনি ভাষায় কী কী নামে ডাকা হয়?"),
    ("money_recovery_46_s9", "একটা চেক বা প্রতিশ্রুতিপত্র মেয়াদ শেষ হওয়ার আগে সৎ উদ্দেশ্যে টাকা দিয়ে হাতে পাওয়া ব্যক্তিকে আইনে কোন বিশেষ মর্যাদা দেওয়া হয়?"),
    ("money_recovery_46_s13", "প্রতিশ্রুতিপত্র, বিনিময় বিল আর চেক - এই তিন ধরনের কাগজকে একসাথে কোন সাধারণ নামে ডাকা হয় আইনে?"),
    ("money_recovery_46_s14", "একটা চেক বা প্রতিশ্রুতিপত্র হাত বদল হয়ে অন্য কারো কাছে গিয়ে সেই ব্যক্তি তার মালিক হয়ে গেলে, এই প্রক্রিয়াকে কী বলা হয়?"),
    ("money_recovery_46_s15", "চেক বা প্রতিশ্রুতিপত্রের পেছনে বা সাথে লাগানো কাগজে সই করে অন্য কারো কাছে হস্তান্তর করার এই কাজটাকে আইনি ভাষায় কী বলে?"),
    ("money_recovery_46_s22", "একটা প্রতিশ্রুতিপত্র বা বিনিময় বিল যদি নির্দিষ্ট দিনে দিতে হয় বলে লেখা না থাকে, তাহলে সেটা পরিশোধের আসল তারিখ কীভাবে ঠিক হয়?"),
    ("money_recovery_46_s26", "চুক্তি করার আইনি যোগ্যতা যার আছে, শুধু তিনিই কি প্রতিশ্রুতিপত্র বা চেক লিখে নিজেকে দায়বদ্ধ করতে পারেন?"),
    ("money_recovery_46_s123", "চেকের মুখের ওপর দুটো সমান্তরাল আড়াআড়ি রেখা টেনে দিলে সেটাকে সাধারণভাবে ক্রস করা চেক বলার নিয়ম কেন?"),
    ("money_recovery_46_s124", "চেকের ওপর দুটো রেখার মাঝে কোনো নির্দিষ্ট ব্যাংকের নাম লিখে দিলে সেই চেককে বিশেষভাবে ক্রস করা বলা হয় কেন, আর এর মানে কী?"),
    ("money_recovery_46_s130", "কেউ একটা চেক হাতে পেলে, তাতে যদি নট নেগোশিয়েবল লেখা থাকে তাহলে তিনি কি আগের মালিকের চেয়ে বেশি অধিকার পেয়ে যান?"),
    ("money_recovery_46_s138_p0", "কারো দেওয়া চেক ব্যাংকে জমা দিলে টাকা কম থাকার কারণে ফেরত এলে, চেক লেখা ব্যক্তির বিরুদ্ধে ফৌজদারি অপরাধ ধরা হয় কিনা?"),
    ("money_recovery_46_s138_p1", "ব্যাংক থেকে চেক ফেরত আসার খবর পাওয়ার পর, টাকা না পাওয়া ব্যক্তি কতদিনের মধ্যে চেক লেখা ব্যক্তিকে লিখিতভাবে জানাতে হয় আর তারপর কতদিন সময় দিতে হয়?"),
    ("money_recovery_46_s138_p2", "ফৌজদারি মামলার পাশাপাশি, চেকের টাকা পুরোপুরি না পেলে দেওয়ানি আদালতে গিয়েও সেই বাকি টাকা দাবি করা যায় কিনা?"),
    ("money_recovery_46_s138A", "চেক না মেলার মামলায় সাজা হওয়ার পর আপিল করতে গেলে, চেকের টাকার কত শতাংশ আগে থেকে জমা দিতে হয়?"),
    ("money_recovery_46_s141", "চেক না মেলার অপরাধে মামলা নেওয়ার আগে আদালত কি শুধু টাকা পাওনা ব্যক্তির লিখিত অভিযোগের অপেক্ষায় থাকে, আর সেটা কতদিনের মধ্যে দিতে হয়?"),
]
