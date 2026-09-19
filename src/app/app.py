"""Gradio demo: a plain Bangla question in, cited provisions out -- plus two
read-only documentation tabs (law corpus browser, gold Q&A browser) for a
supervisor/committee audience to explore the underlying data.

    python -m src.app.app

Deliberately plain on the search tab, per the supervisor's own guidance recorded
in DECISIONS.md 2026-08-23: "a fancy GUI is not required... there should be a
clear input-output system." The two documentation tabs are new, requested
separately, and are presentation-only -- they never influence ranking.

`Dhara.load()` and the corpus/gold indexes all build once here, at module level,
not inside a callback -- reloading per request is the "demo feels frozen" trap
in CLAUDE.md's known-traps table.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import gradio as gr  # noqa: E402

from dhara.service import Answer, Dhara, cite  # noqa: E402
from app import browse  # noqa: E402

print("loading Dhara (index + encoder, once)...")
dhara = Dhara.load()
print(f"loaded. index={dhara.manifest.get('built_by', dhara.manifest.get('checkpoint'))}, "
      f"{dhara.manifest.get('n_chunks', len(dhara.chunk_ids))} chunks")

print("loading corpus browser (act/provision index)...")
ACTS, ACT_CHOICES = browse.build_act_index()
print(f"loaded {len(ACTS)} acts.")

print("loading gold Q&A browser...")
GOLD_ROWS, GOLD_COLUMNS = browse.build_gold_table()
GOLD_DOMAINS = ["সব ডোমেইন"] + sorted({r[0] for r in GOLD_ROWS if r[0]})
print(f"loaded {len(GOLD_ROWS)} gold rows.")

DISCLAIMER_MD = "> **এটি আইনি পরামর্শ নয় — এটি একটি তথ্য অনুসন্ধান সরঞ্জাম।** " \
                "প্রাপ্ত ফলাফল আইনের মূল পাঠ্য থেকে সরাসরি উদ্ধৃত।"

# 20 real, human-adjudicated citizen questions from the frozen dev/test sets
# (never trained on), live-verified against THIS deployed checkpoint+index --
# not guessed. 11 return an exact Act+section match, 3 land one section off
# in the correct Act, 6 land the correct Act with a different section.
# See docs/PROJECT_REPORT.md's demo section for the full verification table
# and scores. The three placeholder examples originally here (police custody,
# spoken talaq, land dispute) scored 0.57-0.61 -- genuinely fine matches --
# but the abstention threshold that shipped was calibrated for a different,
# older checkpoint and silently discarded them; see configs/abstention.json's
# 2026-09-19 recalibration note and the TIER_LOOKAHEAD fix in service.py.
EXAMPLES_BN = [
    "সরকারি কোনো কর্মকর্তা বা কর্মচারী ফৌজদারি মামলায় সাজা পেলেই কি তাঁর চাকরি চলে যায়?",
    "প্রকাশ্য স্থানে ধূমপান করলে জেল বা অন্য কী শাস্তি হয়? কাউকে এ অপরাধে শাস্তি দিতে পুলিশের সহায়তা চাইলে কীভাবে এগোতে হবে?",
    "ধর্ষণের কারণে যে শিশুর জন্ম হয়, তার ভরণপোষণ আর দেখাশোনার দায় কার?",
    "স্বামীর পক্ষে কি বউয়ের নামে যৌতুক-মামলা দায়ের করা সম্ভব?",
    "বিদেশে থাকলে পাওয়ার অব অ্যাটর্নি করার উপায়টা জেনে নিন।",
    "এই পরিস্থিতিতে এসে নথিপত্রে নিজের নাম সংশোধনের কোনো আইনি প্রক্রিয়া আছে কি?",
    "আমি ছয় বছর ধরে নিজের নকশায় রেজিনের জিনিস বানিয়ে বিক্রি করছি। আমার তৈরি ডিজাইনগুলো কপিরাইটের আওতায় আনতে কী করতে হবে?",
    "আমার আসল জন্ম ১৯৯৪ সালে। এখন ক্লাস নাইনে পড়ি, সামনে রেজিস্ট্রেশন। মায়ের বাড়ি এক জেলায়, বাবার বাড়ি আরেক জেলায়, আর আমরা পুরো পরিবার থাকি অন্য একটা শহরে। মামা আমার জন্মনিবন্ধনটা মায়ের বাড়ির জেলায় করে দিয়েছিলেন। আসল জন্মসাল না জানায় তিনি সেখানে লিখে দেন ১৯৯৭। অথচ প্রাইমারি স্কুল থেকে হাইস্কুলে ভর্তির সময় যে সার্টিফিকেট জমা দিয়েছিলাম, তাতে ১৯৯৪ সালই লেখা আছে। এখন রেজিস্ট্রেশনের সময় আমি ১৯৯৪ লিখব, নাকি জন্মনিবন্ধনের ১৯৯৭ লিখব?",
    "তাহলে কি আমৃত্যু মায়ের এই পেনশন পাওয়ার অধিকার থাকবে?",
    "হঠাৎ দুর্ঘটনা, আত্মহত্যা, নাকি রহস্যজনক মৃত্যু: থানায় 'ইউডি কেস' কেন করা হয়, আর কখন করা হয়?",
    "প্রেমের সম্পর্ক থাকলেও পারিবারিকভাবে ২০১১ সালে আমাদের বিয়ে হয়। মেয়ের স্বভাব ঝগড়াটে ও জেদি হওয়ায় শুরুতে পরিবার একটু অমত করলেও পরে মেনে নিয়েছিল। কিন্তু বিয়ের দেড় মাসের মাথায় সে আমাকে একগাদা ঘুমের ওষুধ শরবতের সাথে গুলিয়ে খাইয়ে দেয়, অনেক চিকিৎসার পর আমি কোনোমতে বেঁচে ফিরি। এই ঘটনা পরিবারের সবাই জানে। তবুও আমি সব ভুলে তাকে নিয়ে সংসার করতে চেয়েছিলাম। কিন্তু সে কাউকে পাত্তা না দিয়ে নিজের মতো চলতে থাকে, যার ফলে প্রতিদিন অশান্তি লেগেই থাকত। শেষ পর্যন্ত ২০১৩ সালের জানুয়ারিতে ঝগড়ার এক পর্যায়ে সে বাড়ি ছেড়ে ঢাকায় চলে যায়, আর কখনোই ফিরে আসেনি। স্থানীয় চেয়ারম্যানের মাধ্যমে বিষয়টি মীমাংসার চেষ্টা করেও কোনো ফল পাইনি, শুধু সময়ক্ষেপণ করেছে। এখন প্রশ্ন হলো: ১) আমি যদি তাকে তালাক দিই তবে সে কি আমার বিরুদ্ধে মামলা করতে পারবে? ২) আমি কি দ্বিতীয় বিয়ে করতে পারব? ৩) এই ঝামেলা থেকে আইনগতভাবে মুক্তি পাওয়ার উপায় কী?",
    "নারী ও শিশু নির্যাতনের মামলা কখন করবেন, কোথায় করবেন, কেন আর কীভাবে করবেন?",
    "উত্তরাধিকারী প্রমাণ করতে বাংলাদেশে কী কী কাগজপত্র লাগবে?",
    "আমরা তিন বোন, ভাই কেউ নেই। বাবা মানসিক অসুস্থতায় ভুগছেন। তিনি উইল করে না গেলে কি আইনের হিসাবে তাঁর পুরো সম্পত্তি আমাদের হবে? বাবার এই অসুস্থতার কথা বিবেচনায় এখন আমাদের কী করতে হবে?",
    "একটি রিয়েল এস্টেট ডেভেলপার কোম্পানির সাথে ঢাকায় আমার আট কাঠা জমিতে বহুতল ভবন নির্মাণের চুক্তি হয়েছিল। চুক্তি মোতাবেক ২০০৬ থেকে ২০০৯ সালের ডিসেম্বর অর্থাৎ ৩৬ মাসের মধ্যে এবং প্রয়োজনে অতিরিক্ত ছয় মাস বাড়িয়ে ২০১০ সালের জুলাইয়ের মধ্যে আমার ভাগের ফ্ল্যাটগুলো বুঝিয়ে দেওয়ার কথা ছিল। কিন্তু গত তিন বছর চার মাসে তারা যেটুকু কাজ করেছে, তাতে আগামী দুই বছরেও কাজ শেষ হওয়ার কোনো লক্ষণ নেই। চুক্তিতে বলা ছিল সময়মতো ফ্ল্যাট দিতে না পারলে উভয় পক্ষ বসে সিদ্ধান্ত নেবে। কিন্তু সম্প্রতি তারা অফিসের ঠিকানা বদলে অন্য কোথাও চলে গেছে এবং ফোনেও তাদের সাথে কোনো যোগাযোগ করা যাচ্ছে না। মনে হচ্ছে প্রতারকের খপ্পরে পড়েছি। তারা দাবি করেছিল যে তারা রিহ্যাবের মেম্বার। এখন রিহ্যাবের মাধ্যমে আমি কি কোনো প্রতিকার পেতে পারি, নাকি সরাসরি আদালতে আইনি মামলা করতে হবে?",
    "অনেক সময় শ্রমিকদের কোনো পাওনা না মিটিয়েই যখন-তখন কারখানা থেকে বের করে দেওয়া হয়। চাকরি শেষ করার নিয়মটা কী, আর শ্রমিকেরা মালিকের কাছ থেকে কখন কী সুবিধা পেতে পারেন?",
    "পত্রপত্রিকায় প্রায়ই খবর আসে, অমুক গ্রামে গ্রাম্য সালিসে কোনো মানুষ বা পরিবারকে এই-সেই শাস্তি দেওয়া হয়েছে। দেশের প্রচলিত আইনে গ্রাম্য সালিস কি বৈধ? বৈধ হলে তার ক্ষমতা কতখানি? গ্রাম্য সালিসে শাস্তি দেওয়া গেলে সর্বোচ্চ কতটা শাস্তি দেওয়া যায়? সমাজ বা গ্রামের কোনো প্রচলিত নিয়ম একেবারে অমান্য করলে, কিংবা গ্রামের শৃঙ্খলা নষ্ট হয় এমন কাজে জড়ানোর অভিযোগে, গ্রামের সবার সিদ্ধান্তে অভিযুক্ত লোককে কি সমাজচ্যুত বা একঘরে করে রাখা যায়? একঘরে করা কি আইনের সঙ্গে মেলে? কোনো কারণে কাউকে সমাজচ্যুত করা কি যায়? বলে রাখি, সমাজচ্যুত করা মানে অভিযুক্ত লোকটার সঙ্গে সব ধরনের লেনদেন, কথাবার্তা আর মেলামেশা পুরোপুরি বন্ধ করে দিয়ে তাকে গ্রামের সিদ্ধান্ত মানতে বাধ্য করা। লোকটা যখন গ্রামের মানুষের সামনে নিজের ভুল স্বীকার করে আর কথা দেয় যে আর এমন করবে না, তখন আবার তাকে স্বাভাবিকভাবে সমাজে ফিরিয়ে নেওয়া হয়। বিষয়টার আইনি দিক নিয়ে বিস্তারিত জানতে চাই।",
    "আমি আঁকাআঁকি করি, নানা রকম ইলাস্ট্রেশনের কাজ করি। বেশির ভাগ কাজ করেছি অনলাইনে বিভিন্ন জায়গায় ছাপা হওয়া গল্পের ছবি আঁকার। কিন্তু আমার আঁকা ছবিগুলোর স্বত্ব নিয়ে প্রায়ই নানা ঝামেলায় পড়তে হয়। আগেও এমন সমস্যায় পড়েছি, প্রতিবারই মিটমাট করে নিতে হয়েছে। এই সমস্যার সমাধান কীভাবে পাব? কাজটা তো আমি ছাড়তে চাই না। তাহলে আইনিভাবে ধাপে ধাপে কী করতে পারি?",
    "কয়েক দিন আগে একটি এলাকায় গণপিটুনিতে ছয় ছাত্রকে হত্যা করতে দেখেছি। আরেক জায়গায় পুলিশের নির্দেশে একজনকে পিটিয়ে মেরে ফেলা হয়েছে। প্রায়ই ছিনতাইকারী সন্দেহে বা অন্য অজুহাতে মানুষকে এভাবে পিটিয়ে মারা হয়। আইন অনুযায়ী গণপিটুনি কতটা অপরাধ? যারা এর জন্য দায়ী, তাদের বিরুদ্ধে কি আইনি ব্যবস্থা নেওয়া যায়? আমাদের প্রচলিত আইন এ বিষয়ে কী বলে? বিস্তারিত জানতে চাই।",
    "তৃতীয় প্রশ্ন: লেখকের প্রথম দিকের কয়েকটি বই এক ব্যক্তির নামে প্রকাশিত হলেও পরের সংস্করণগুলো আরেকজন বা অন্যদের নামে বের হয়েছে। সে ক্ষেত্রে বইগুলোর স্বত্ব কার হবে?",
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&family=Noto+Sans:wght@400;500;600&display=swap');
* { font-family: 'Noto Sans Bengali', 'Noto Sans', sans-serif !important; }
.gradio-container { max-width: 1080px !important; margin: auto; }
#header-title { text-align: center; margin-bottom: 0; }
#header-sub { text-align: center; color: var(--body-text-color-subdued); margin-top: 0.2em; }
.hit-card {
    border: 1px solid var(--border-color-primary); border-radius: 12px;
    padding: 14px 18px; margin-bottom: 12px; background: var(--background-fill-secondary);
}
.hit-card h4 { margin: 0 0 6px 0; }
.hit-score { float: right; font-size: 0.85em; color: var(--body-text-color-subdued); }
.risk-banner {
    border-left: 4px solid #d9534f; background: #fdf2f2; color: #7a1f1f;
    padding: 10px 16px; border-radius: 8px; margin-bottom: 10px;
}
"""


def format_answer(answer: Answer) -> tuple[str, str]:
    """(banner_html, results_html). Renders `hits` whenever present, even when
    `abstained=True` -- the demo shows the raw top-k regardless of confidence
    (see `respond`'s `show_all=True`), with a low-confidence note instead of
    hiding results outright. `service.py`'s real abstention logic (and every
    non-demo caller) is unchanged; this is presentation only."""
    banner = f'<div class="risk-banner">⚠ {answer.notice_bn}</div>' if answer.legal_aid_referral else ""

    if not answer.hits:
        body = f"<p><strong>কোনো ফলাফল পাওয়া যায়নি।</strong></p><p>{answer.message_bn}</p>"
        return banner, body

    body = ""
    if answer.abstained:
        body += (
            '<div style="border-left:4px solid #b8860b; background:#fdf6e3; color:#6b5100; '
            'padding:10px 16px; border-radius:8px; margin-bottom:12px;">'
            f'⚠ কোনো ফলাফলই আত্মবিশ্বাসের থ্রেশহোল্ড অতিক্রম করেনি '
            f'(top score {answer.top_score:.3f} &lt; threshold {answer.threshold:.3f}) — '
            'নিচেরগুলো এখনও সর্বোচ্চ-স্কোরের ফলাফল, নিশ্চিত উত্তর নয়।</div>'
        )

    cards = []
    for i, hit in enumerate(answer.hits, 1):
        cards.append(
            f'<div class="hit-card">'
            f'<span class="hit-score">score {hit.score:.3f} · {hit.domain}</span>'
            f'<h4>{i}. {cite(hit)}</h4>'
            f'<div>{hit.provision_title}</div>'
            f'<p>{hit.text_raw}</p>'
            f'<div style="font-size:0.85em;color:var(--body-text-color-subdued);">'
            f'crawled {hit.crawl_date} · <a href="{hit.source_url}" target="_blank">{hit.source_url}</a></div>'
            f'</div>'
        )
    body += "\n".join(cards)
    return banner, body


def respond(query: str, k: int):
    answer = dhara.search(query, k=int(k), show_all=True)
    banner, results = format_answer(answer)
    return banner, results


def show_act(act_id: str) -> str:
    if not act_id:
        return "*একটি আইন নির্বাচন করুন।*"
    return browse.render_act_markdown(ACTS[act_id])


def filter_gold(domain: str, query: str):
    rows = GOLD_ROWS
    if domain and domain != "সব ডোমেইন":
        rows = [r for r in rows if r[0] == domain]
    if query:
        q = query.strip().lower()
        rows = [r for r in rows if q in r[3].lower() or q in r[4].lower()]
    return rows


with gr.Blocks(title="ধারা (Dhara) — Bangla Legal Retrieval") as demo:
    gr.Markdown("# ধারা · Dhara", elem_id="header-title")
    gr.Markdown("Bangla legal provision retrieval — fine-tuned BGE-m3 bi-encoder", elem_id="header-sub")

    with gr.Tabs():
        with gr.Tab("🔍 অনুসন্ধান  ·  Search"):
            gr.Markdown(DISCLAIMER_MD)
            with gr.Row():
                query_box = gr.Textbox(
                    label="আপনার প্রশ্ন লিখুন (সাধারণ বাংলায়)",
                    placeholder="যেমনঃ পুলিশ আমাকে ধরে নিয়ে গেছে, আমার কী অধিকার?",
                    lines=3, scale=4,
                )
                with gr.Column(scale=1):
                    k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="কয়টি ধারা দেখাবে")
                    submit = gr.Button("খুঁজুন", variant="primary", size="lg")
            banner_out = gr.HTML()
            results_out = gr.HTML()

            submit.click(fn=respond, inputs=[query_box, k_slider], outputs=[banner_out, results_out])
            query_box.submit(fn=respond, inputs=[query_box, k_slider], outputs=[banner_out, results_out])

            gr.Examples(
                examples=[[q] for q in EXAMPLES_BN],
                inputs=[query_box],
                label="২০টি যাচাইকৃত উদাহরণ প্রশ্ন (Gold set থেকে, প্রকৃত সঠিক উত্তরসহ)",
            )

        with gr.Tab("📚 আইন সংকলন  ·  Law Corpus"):
            gr.Markdown(
                f"**{len(ACTS)} আইন, {sum(len(a['provisions']) for a in ACTS.values())} ধারা** — "
                "bdlaws পোর্টাল থেকে সংগ্রহীত সম্পূর্ণ কর্পাস, আইন অনুসারে সাজানো।"
            )
            act_dropdown = gr.Dropdown(
                choices=ACT_CHOICES, label="একটি আইন নির্বাচন করুন",
                filterable=True, value=ACT_CHOICES[0][1] if ACT_CHOICES else None,
            )
            act_display = gr.Markdown()
            act_dropdown.change(fn=show_act, inputs=[act_dropdown], outputs=[act_display])
            demo.load(fn=show_act, inputs=[act_dropdown], outputs=[act_display])

        with gr.Tab("🗂 নমুনা প্রশ্নোত্তর  ·  Gold Q&A"):
            gr.Markdown(
                f"**{len(GOLD_ROWS)} মানব-যাচাইকৃত প্রশ্ন**, দুইজন annotator দ্বারা adjudicated। "
                "প্রতিটি প্রশ্নের সাথে তার সঠিক ধারা যুক্ত করা আছে — এটিই মডেল মূল্যায়নের জন্য ব্যবহৃত gold set।"
            )
            with gr.Row():
                domain_filter = gr.Dropdown(choices=GOLD_DOMAINS, value="সব ডোমেইন", label="ডোমেইন", scale=1)
                search_filter = gr.Textbox(label="প্রশ্ন বা ধারায় খুঁজুন", scale=2)
            gold_table = gr.Dataframe(
                value=GOLD_ROWS, headers=GOLD_COLUMNS, wrap=True,
                interactive=False, row_count=(10, "dynamic"),
            )
            domain_filter.change(fn=filter_gold, inputs=[domain_filter, search_filter], outputs=[gold_table])
            search_filter.change(fn=filter_gold, inputs=[domain_filter, search_filter], outputs=[gold_table])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"), css=CUSTOM_CSS)
