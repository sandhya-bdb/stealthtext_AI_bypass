import streamlit as st
import streamlit.components.v1 as components
import requests
import difflib
import os
from dotenv import load_dotenv

load_dotenv()

# Use BACKEND_URL env var so the frontend works in any deployment environment.
# Defaults to localhost for local development.
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="StealthText | AI Humanizer",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f18; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #12141f !important;
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] * { color: #c8cde8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #b0b7d4 !important; font-size: 0.88rem; line-height: 1.6; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #c850c0 100%);
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.5);
    color: #fff !important;
    border: none !important;
}

/* ── Text Areas ──────────────────────────────────────────── */
.stTextArea textarea {
    background: #1a1d2e !important;
    color: #e8eaf6 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}
.stTextArea textarea:focus { border-color: #6C63FF !important; box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important; }
.stTextArea label { color: #9098c8 !important; font-weight: 600 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Metric Cards ────────────────────────────────────────── */
.m-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}
.m-card.good  { border-top: 3px solid #22c55e; }
.m-card.warn  { border-top: 3px solid #f59e0b; }
.m-card.bad   { border-top: 3px solid #ef4444; }
.m-val { font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0; }
.good .m-val  { color: #22c55e; }
.warn .m-val  { color: #f59e0b; }
.bad  .m-val  { color: #ef4444; }
.m-name { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }
.m-sub  { font-size: 0.8rem; color: #9098c8; margin-top: 0.3rem; }

/* ── Verdict badge ───────────────────────────────────────── */
.badge-human { display:inline-block; background:rgba(34,197,94,0.15); color:#22c55e; border:1px solid #22c55e; padding:0.25rem 0.9rem; border-radius:2rem; font-weight:700; font-size:0.9rem; }
.badge-ai    { display:inline-block; background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid #ef4444; padding:0.25rem 0.9rem; border-radius:2rem; font-weight:700; font-size:0.9rem; }

/* ── Diff box ────────────────────────────────────────────── */
.diff-box { background:#1a1d2e; border:1px solid #2a2d3e; border-radius:12px; padding:1.2rem 1.4rem; line-height:1.85; font-size:0.92rem; color:#c8cde8; }
ins  { background:rgba(34,197,94,0.18); color:#22c55e; text-decoration:none; border-radius:3px; padding:0 2px; }
del  { background:rgba(239,68,68,0.18);  color:#ef4444; border-radius:3px; padding:0 2px; }

/* ── Section titles ──────────────────────────────────────────────────────────────────── */
.sec-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.10em; color:#6b7280; margin-bottom:0.6rem; }
hr.divider { border:none; border-top:1px solid #2a2d3e; margin:1.5rem 0; }

/* ── LangSmith badge ─────────────────────────────────────────────────────────────────── */
.ls-badge-on  { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(34,197,94,0.12); color:#22c55e; border:1px solid #22c55e; padding:0.22rem 0.75rem; border-radius:2rem; font-size:0.78rem; font-weight:700; }
.ls-badge-off { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(107,114,128,0.12); color:#6b7280; border:1px solid #374151; padding:0.22rem 0.75rem; border-radius:2rem; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────
SAMPLE_AI_TEXT = (
    "Artificial intelligence is fundamentally transforming the landscape of modern business operations. "
    "By leveraging advanced machine learning algorithms, organizations can now utilize data-driven insights "
    "to make more informed strategic decisions. Furthermore, the implementation of AI technologies enables "
    "companies to achieve unprecedented levels of operational efficiency. It is crucial for businesses to "
    "embrace these transformative technologies in order to remain competitive in today's rapidly evolving "
    "digital marketplace. Moreover, the integration of AI systems facilitates the automation of repetitive "
    "tasks, thereby allowing human employees to focus on more creative and high-value activities."
)

def ppl_cls(v):
    return "good" if v >= 60 else ("warn" if v >= 35 else "bad")

def burst_cls(v):
    return "good" if v >= 20 else ("warn" if v >= 8 else "bad")

def card(name, val, sub, cls):
    st.markdown(f"""
    <div class="m-card {cls}">
        <div class="m-name">{name}</div>
        <div class="m-val">{val}</div>
        <div class="m-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def word_diff(a, b):
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(" ".join(aw[i1:i2]))
        elif tag == "replace":
            out.append(f'<del>{" ".join(aw[i1:i2])}</del>')
            out.append(f'<ins>{" ".join(bw[j1:j2])}</ins>')
        elif tag == "delete":
            out.append(f'<del>{" ".join(aw[i1:i2])}</del>')
        elif tag == "insert":
            out.append(f'<ins>{" ".join(bw[j1:j2])}</ins>')
    return " ".join(out)

# ─── Sidebar ───────────────────────────────────────────────────────────────
_ls_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
_ls_key_set = bool(os.environ.get("LANGCHAIN_API_KEY", "").strip())
_ls_project = os.environ.get("LANGCHAIN_PROJECT", "StealthText")
_ls_active   = _ls_enabled and _ls_key_set

with st.sidebar:
    st.markdown("## 🕵️ StealthText")
    st.markdown("---")
    st.markdown("🟢 **Backend:** FastAPI running")
    st.markdown("🟢 **LLM:** Groq — Llama 3.3 70B")
    st.markdown("🟢 **Detector:** GPT-2")

    # ── LangSmith tracing status ──────────────────────────
    st.markdown("")
    st.markdown("**🔍 LangSmith Tracing**")
    if _ls_active:
        ls_url = f"https://smith.langchain.com/o/~/projects/p/{_ls_project}"
        st.markdown(
            f'<a href="{ls_url}" target="_blank">'
            f'<span class="ls-badge-on">● Active — {_ls_project}</span></a>',
            unsafe_allow_html=True
        )
        st.caption("Every humanize run is traced. Click badge to open LangSmith.")
    elif _ls_enabled and not _ls_key_set:
        st.markdown(
            '<span class="ls-badge-off">⚠ Key missing</span>',
            unsafe_allow_html=True
        )
        st.caption("Set `LANGCHAIN_API_KEY` in `.env` to activate.")
    else:
        st.markdown(
            '<span class="ls-badge-off">○ Disabled</span>',
            unsafe_allow_html=True
        )
        st.caption("Set `LANGCHAIN_TRACING_V2=true` in `.env` to enable.")

    st.markdown("---")
    st.markdown("**✍️ Writing Tone**")
    tone_option = st.selectbox(
        "Select tone",
        options=["Casual / Creative", "Professional / Academic"],
        index=0,
        label_visibility="collapsed"
    )
    tone = "casual" if tone_option == "Casual / Creative" else "professional"

    st.markdown("---")
    st.markdown("### 📖 Score Guide")
    st.markdown("""
**Perplexity** — unpredictability of words
- ✅ > 60 → Human-like
- ⚠️ 35–60 → Borderline  
- ❌ < 35 → Sounds like AI

**Burstiness** — sentence length variety
- ✅ > 20 → Human-like
- ⚠️ 8–20 → Borderline
- ❌ < 8 → Sounds like AI

**AI Score** — combined probability
- 0% = Definitely human
- 100% = Definitely AI

**Tip:** AI text is uniform and predictable.
Humans write messily — short bursts, then
long complex sentences.
    """)

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("## 🕵️ StealthText — AI Text Humanizer")
st.markdown("Paste AI-generated text → we rewrite it to bypass detectors, then show you **what changed** and **how scores improved**.")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Input / Output ────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="sec-title">📥 Your Input Text</div>', unsafe_allow_html=True)
    if st.button("📋 Load Sample AI Text", use_container_width=False):
        st.session_state['sample_loaded'] = True

    default_val = SAMPLE_AI_TEXT if st.session_state.get('sample_loaded') else ""
    input_text = st.text_area("", value=default_val, height=260,
                               placeholder="Paste AI-generated text here…", key="input_area", label_visibility="collapsed")

    b1, b2 = st.columns(2)
    with b1:
        check_btn = st.button("🔍 Check for AI", use_container_width=True)
    with b2:
        humanize_btn = st.button("✨ Humanize Text", use_container_width=True)

with right:
    st.markdown('<div class="sec-title">📤 Humanized Output</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)  # align with left
    if 'humanized_result' in st.session_state:
        final_text = st.session_state['humanized_result']['final_text']
        st.text_area("", value=final_text,
                     height=260, key="output_area", label_visibility="collapsed")
        # ── Copy to clipboard button ──────────────────────────────────────────
        # Escape backticks/backslashes so the JS template literal is safe
        safe_text = final_text.replace("\\", "\\\\").replace("`", "\\`")
        components.html(
            f"""
            <button id="copy-btn"
              style="
                background: linear-gradient(135deg, #6C63FF 0%, #c850c0 100%);
                color: #fff;
                border: none;
                border-radius: 8px;
                font-weight: 700;
                font-size: 0.88rem;
                padding: 0.45rem 1.1rem;
                cursor: pointer;
                transition: transform 0.15s, box-shadow 0.15s;
                box-shadow: 0 3px 12px rgba(108,99,255,0.35);
                font-family: 'Inter', sans-serif;
                margin-top: 4px;
              "
              onmouseenter="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 20px rgba(108,99,255,0.5)'"
              onmouseleave="this.style.transform='';this.style.boxShadow='0 3px 12px rgba(108,99,255,0.35)'"
              onclick="
                navigator.clipboard.writeText(`{safe_text}`).then(() => {{
                  const btn = document.getElementById('copy-btn');
                  btn.textContent = '✅ Copied!';
                  btn.style.background = 'linear-gradient(135deg,#22c55e,#16a34a)';
                  setTimeout(() => {{
                    btn.textContent = '📋 Copy to Clipboard';
                    btn.style.background = 'linear-gradient(135deg, #6C63FF 0%, #c850c0 100%)';
                  }}, 1500);
                }}).catch(() => {{
                  alert('Copy failed — please select and copy manually.');
                }});
              "
            >📋 Copy to Clipboard</button>
            """,
            height=48,
        )
    else:
        st.text_area("", value="", height=260, disabled=True,
                     placeholder="Humanized text will appear here…", label_visibility="collapsed")

# ─── Button Actions ────────────────────────────────────────────────────────
if check_btn:
    if input_text.strip():
        with st.spinner("Analyzing with GPT-2…"):
            try:
                r = requests.post(f"{API_URL}/analyze", json={"text": input_text}, timeout=60)
                if r.status_code == 200:
                    st.session_state['analysis'] = r.json()
                    st.session_state.pop('humanized_result', None)
                else:
                    st.error(f"Backend error {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Run: `uvicorn backend.api:app --reload`")
    else:
        st.warning("Paste some text first.")

if humanize_btn:
    if input_text.strip():
        with st.spinner("Agent rewriting (up to 3 iterations via Groq)… ~20–40 seconds"):
            try:
                r = requests.post(
                    f"{API_URL}/humanize",
                    json={"text": input_text, "tone": tone},
                    timeout=180
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state['humanized_result'] = data
                    orig = requests.post(f"{API_URL}/analyze", json={"text": input_text}, timeout=60)
                    if orig.status_code == 200:
                        st.session_state['analysis'] = orig.json()
                    st.rerun()
                else:
                    st.error(f"Backend error: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
            except requests.exceptions.Timeout:
                st.error("⏱ Timed out. Try shorter text (< 300 words).")
    else:
        st.warning("Paste some text first.")

# ─── Analysis Section ──────────────────────────────────────────────────────
if 'analysis' in st.session_state:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    d = st.session_state['analysis']
    is_ai = d['ai_score'] > 50
    badge = f'<span class="badge-ai">🤖 {d["verdict"]}</span>' if is_ai else f'<span class="badge-human">🧑 {d["verdict"]}</span>'
    st.markdown(f"### 📊 Original Text Analysis &nbsp;&nbsp; {badge}", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Perplexity", round(d['perplexity'], 1),
             "Target > 60 (unpredictable = human)", ppl_cls(d['perplexity']))
    with c2:
        card("Burstiness", round(d['burstiness'], 1),
             "Target > 20 (varied sentences = human)", burst_cls(d['burstiness']))
    with c3:
        card("AI Score", f"{d['ai_score']}%",
             "0% = human · 100% = AI-generated",
             "bad" if is_ai else "good")

# ─── After Humanizing ──────────────────────────────────────────────────────
if 'humanized_result' in st.session_state:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    h = st.session_state['humanized_result']
    orig = st.session_state.get('analysis', {})

    fp = round(h['final_perplexity'], 1)
    fb = round(h['final_burstiness'], 1)
    op = orig.get('perplexity', fp)
    ob = orig.get('burstiness', fb)
    dp = round(fp - op, 1)
    db = round(fb - ob, 1)

    st.markdown(f"### 🚀 After Humanizing &nbsp;<small style='color:#6b7280;font-size:0.8rem;font-weight:400'>{h['iterations']} iteration(s) by LangGraph agent</small>", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        arrow = f"▲ +{dp}" if dp > 0 else (f"▼ {dp}" if dp < 0 else "→ unchanged")
        card("Final Perplexity", fp, f"{arrow} &nbsp;|&nbsp; was {round(op,1)}", ppl_cls(fp))
    with c2:
        arrow = f"▲ +{db}" if db > 0 else (f"▼ {db}" if db < 0 else "→ unchanged")
        card("Final Burstiness", fb, f"{arrow} &nbsp;|&nbsp; was {round(ob,1)}", burst_cls(fb))
    with c3:
        card("Rewrites", h['iterations'], "Agent stops when scores are good or max 3 reached", "good")

    # ── Score Trend Chart ─────────────────────────────────────────────────
    score_history = h.get('score_history', [])
    if len(score_history) > 1:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📈 Score Trend")
        st.caption("How the text improved across each rewrite iteration by the agent.")
        
        import pandas as pd
        df = pd.DataFrame(score_history)
        # Ensure iteration is an integer and set it as index for the x-axis
        df['iteration'] = df['iteration'].astype(int)
        df = df.set_index('iteration')
        
        st.line_chart(df[['perplexity', 'burstiness']])

    # ── Diff ──────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🔎 What the Agent Changed")

    original_text  = h.get('original_text', input_text or "")
    humanized_text = h.get('final_text', "")

    if original_text.strip() == humanized_text.strip():
        st.warning(
            "⚠️ **The text didn't change.** This happens when your input already looks human "
            "(high perplexity + burstiness). Try the **sample AI text** above — it starts with "
            "typical AI buzzwords ('leveraging', 'crucial', 'moreover') that the agent will rewrite aggressively."
        )
    else:
        st.caption("🟢 Highlighted = words added by the rewriter  •  🔴 Strikethrough = words removed")
        diff_html = word_diff(original_text, humanized_text)
        st.markdown(f'<div class="diff-box">{diff_html}</div>', unsafe_allow_html=True)

    with st.expander("🛠 Debug: Raw API Response"):
        st.json(h)
