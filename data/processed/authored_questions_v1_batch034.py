"""Batch 34: local_government domain, Union Parishad Act 2009
(local_government_1027 - a DIFFERENT act_id from the Pourashava Act's
local_government_1024 authored in batches 20/26). This act was wrongly
believed closed earlier in the session due to a Bengali-string literal
comparison bug in an ad-hoc diagnostic (retyping the same Bangla text in
two separate tool calls silently produced two different Unicode
representations) - it actually had 95 provisions remaining, and is
arguably the single most locally-relevant government body for Bangladesh's
majority-rural population (village-level chairman/member elections,
no-confidence, village police, citizen charter, property declaration).
Same institutional-vocabulary pattern as the Pourashava Act - local_
government is already in NO_REGISTER_GAP_DOMAINS, so no special
overlap-avoidance effort spent beyond normal paraphrase.
"""

BATCH = [
    ("local_government_1027_s17", "নদী ভাঙন বা প্রাকৃতিক দুর্যোগে কোনো এলাকা তলিয়ে গেলে, সরকার কি সেই স্থানীয় পরিষদ ভেঙে দিতে বা নতুন করে গঠন করতে পারে?"),
    ("local_government_1027_s18", "একটা নতুন এলাকা ইউনিয়ন ঘোষণা হলে বা পরিষদের মেয়াদ শেষ হলে, নির্বাচিত পরিষদ না হওয়া পর্যন্ত সরকার কি একজন প্রশাসক বসিয়ে দিতে পারে, আর তিনি কতদিন এই দায়িত্ব পালন করতে পারেন?"),
    ("local_government_1027_s19", "গ্রামের একজন বাসিন্দা কোন কোন শর্ত পূরণ করলে ওয়ার্ডের ভোটার তালিকায় নাম তুলতে পারবেন?"),
    ("local_government_1027_s27", "একই ব্যক্তি কি একসাথে চেয়ারম্যান আর সদস্য দুই পদেই নির্বাচনে প্রার্থী হতে পারেন?"),
    ("local_government_1027_s31", "নির্ধারিত সময়ের মধ্যে দায়িত্ব হস্তান্তর না করলে একজন প্রধানের সর্বোচ্চ কত টাকা জরিমানা হতে পারে?"),
    ("local_government_1027_s32", "একজন সদস্য নিজে থেকে দায়িত্ব ছাড়তে চাইলে কার কাছে লিখিতভাবে জানাতে হয়, আর সেটা গৃহীত হওয়ার খবর কতদিনের মধ্যে উপজেলা অফিসারকে জানাতে হয়?"),
    ("local_government_1027_s34_p0", "কারো বিরুদ্ধে ফৌজদারি মামলার অভিযোগপত্র আদালতে গৃহীত হলে বা অপসারণের প্রক্রিয়া শুরু হলে, সরকার কি তাকে সাময়িকভাবে দায়িত্ব থেকে সরিয়ে দিতে পারে?"),
    ("local_government_1027_s35", "শপথ না নেওয়া, পদত্যাগ, মৃত্যু বা অযোগ্যতা - এসবের মধ্যে কোন কোন কারণে একজন প্রধান বা সদস্যের পদ খালি হয়ে যায়?"),
    ("local_government_1027_s39", "নির্দিষ্ট কোনো অভিযোগে একজন প্রধান বা সদস্যের বিরুদ্ধে অনাস্থা আনতে হলে, কতজন সদস্যের স্বাক্ষর লাগে আর সেটা কার কাছে জমা দিতে হয়?"),
    ("local_government_1027_s40", "একজন প্রধান বা সদস্য বছরে সর্বোচ্চ কত মাস বিশ্রামের জন্য অনুমতি পেতে পারেন, আর মহিলা প্রধান বা সদস্যের মাতৃত্বকালীন বিরতির ক্ষেত্রে কী নিয়ম?"),
    ("local_government_1027_s41", "দায়িত্ব নেওয়ার আগে একজন প্রধান বা সদস্যকে নিজের ও পরিবারের সব সম্পদের হিসাব লিখে জমা দিতে হয় কিনা?"),
    ("local_government_1027_s48", "গ্রামের নিরাপত্তার জন্য সরকার কি চাহিদা অনুযায়ী গ্রাম পুলিশ বাহিনী গঠন করতে পারে, আর তাদের নিয়োগ ও প্রশিক্ষণের সিদ্ধান্ত কে নেয়?"),
    ("local_government_1027_s49", "প্রতিটা স্থানীয় পরিষদকে কি নাগরিকদের কোন কোন সেবা কত সময়ের মধ্যে দেওয়া হবে তা লিখে প্রকাশ করতে হয়?"),
    ("local_government_1027_s51", "জমির মতো স্থাবর সম্পদ কেনার বা বিক্রি করার আগে একটা স্থানীয় পরিষদকে কি সরকারের অনুমতি নিতে হয়?"),
    ("local_government_1027_s53", "একটা স্থানীয় পরিষদের নিজস্ব টাকার তহবিলে সরকারি অনুদান ছাড়া আর কোন কোন উৎস থেকে টাকা জমা হতে পারে?"),
    ("local_government_1027_s56", "কর্মচারীদের বেতন-ভাতা বা আদালতের রায়ে দেওয়া টাকা - এগুলো কি পরিষদের তহবিল থেকে বাধ্যতামূলকভাবে দিতেই হয়?"),
    ("local_government_1027_s62", "একটা স্থানীয় পরিষদে একজন সচিব আর একজন হিসাবরক্ষক-কাম-কম্পিউটার অপারেটর থাকা কি বাধ্যতামূলক, আর তাদের নিয়োগ কে দেয়?"),
    ("local_government_1027_s103", "স্থানীয় পরিষদ থেকে কোনো অনুমতি নিলে সেটা মৌখিক না লিখিত আকারে দিতে হয়?"),
    ("local_government_1027_s104", "সরকারি দায়িত্ব পালনের সময় করা কোনো কাজের জন্য একটা পরিষদের বিরুদ্ধে মামলা করতে চাইলে, আগে নোটিশ দিয়ে কতদিন অপেক্ষা করতে হয়?"),
    ("local_government_1027_s105", "কাউকে কিছু করতে বা না করতে বলার প্রয়োজন হলে, সেই নির্দেশ কি নোটিশ আকারে জারি করতে হয়?"),
    ("local_government_1027_s107", "একটা স্থানীয় পরিষদের প্রধান, সদস্য বা কর্মচারীদের ফৌজদারি আইনের চোখে সরকারি কর্মচারী হিসেবেই গণ্য করা হয় কিনা?"),
]
