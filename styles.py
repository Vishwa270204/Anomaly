"""All dashboard CSS lives here so app.py stays focused on layout and data."""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
html { font-size: 16px; }

:root {
    --primary: #0F3554;
    --primary-dark: #0B2740;
    --warn: #DC6803;
    --danger: #E4463F;
    --border: #E4E9F0;
    --muted: #64748B;
    --surface: #FFFFFF;
    --bg: #F5F7FA;
    --ink: #0F172A;
}

.stApp { background-color: var(--bg); }

/* Streamlit's fixed top toolbar sits above our content; shrink it and give
   the block-container just enough clearance to sit below it. */
header[data-testid="stHeader"] { height: 2.4rem; background: transparent; }
.block-container {
    padding-top: 2.6rem;
    padding-bottom: 0.5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

/* Tighten Streamlit's default vertical rhythm so pages need less scrolling */
[data-testid="stVerticalBlock"] { gap: 0.35rem; }
div[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 2rem !important;
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
}
[data-testid="stWidgetLabel"] p { margin-bottom: 0.1rem !important; }

/* ---------- Metric / KPI cards ---------- */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    transition: box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 4px 10px rgba(15, 23, 42, 0.07); }
[data-testid="stMetricValue"] { font-size: 1.25rem; font-weight: 700; color: var(--ink); }
[data-testid="stMetricLabel"] {
    font-size: 0.74rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.03em;
}

/* ---------- Headings ---------- */
h2, [data-testid="stMarkdownContainer"] h2 {
    color: var(--ink); font-size: 1.55rem !important; font-weight: 700;
    margin-top: 1.1rem !important; margin-bottom: 0.35rem !important;
    padding-bottom: 0.35rem; border-bottom: 1px solid var(--border);
}
h3, [data-testid="stMarkdownContainer"] h3 {
    color: var(--ink); font-size: 1.22rem !important; font-weight: 700;
    margin-top: 0.7rem !important; margin-bottom: 0.25rem !important;
}
h4 {
    color: #1E293B; font-size: 1.05rem !important; font-weight: 600;
    margin-top: 0 !important; margin-bottom: 0.2rem !important;
}
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

[data-testid="stCaptionContainer"], .stCaption { margin-bottom: 0.2rem !important; }

/* ---------- Buttons ---------- */
.stButton > button { border-radius: 8px; font-weight: 600; border: 1px solid var(--border); }
.stButton > button[kind="primary"] { background-color: var(--primary); border-color: var(--primary); }
.stButton > button[kind="primary"]:hover { background-color: var(--primary-dark); border-color: var(--primary-dark); }

/* ---------- Expanders / bordered containers ---------- */
.streamlit-expanderHeader { font-weight: 600; border-radius: 8px; }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px !important; padding: 0.5rem 0.9rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] { gap: 0.3rem; }

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

label, .stSelectbox label, .stDateInput label {
    font-weight: 600 !important; font-size: 0.83rem !important; color: #334155 !important;
}

/* ---------- Header banner ---------- */
.app-header {
    background: linear-gradient(135deg, #0F3554 0%, #164A73 100%);
    padding: 1.05rem 1.3rem; border-radius: 12px; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.85rem;
    box-shadow: 0 4px 14px rgba(15, 53, 84, 0.18);
}
.app-header-icon {
    background: rgba(255,255,255,0.14); color: #FFFFFF; width: 46px; height: 46px;
    border-radius: 11px; display: flex; align-items: center; justify-content: center;
    font-size: 1.45rem; flex-shrink: 0;
}
.app-header-title { color: #FFFFFF; font-size: 1.65rem; font-weight: 700; line-height: 1.2; }
.app-header-subtitle { color: #D6E4F5; font-size: 0.92rem; margin-top: 0.16rem; line-height: 1.35; }

/* ---------- AI Explanation card ---------- */
.ai-title { color: var(--ink); font-size: 1.8rem; font-weight: 700; line-height: 1.3; margin-top: 0.25rem; margin-bottom: 0.15rem; }
.ai-card {
    width: 100%; box-sizing: border-box; background: #FFFFFF; border: 1px solid #D9E1EA;
    border-radius: 12px; padding: 1.35rem 1.5rem; box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05);
}
.ai-row { padding: 0.15rem 0 0.9rem 0; }
.ai-row:last-child { padding-bottom: 0; }
.ai-label { color: var(--primary); font-size: 1rem; font-weight: 700; line-height: 1.35; margin-bottom: 0.28rem; }
.ai-body { color: #1E293B; font-size: 1.06rem; line-height: 1.7; overflow-wrap: anywhere; }

.ai-alert-box {
    display: flex; gap: 0.65rem; background: #FEF3F2; border: 1px solid #FECDCA;
    border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 1rem;
}
.ai-alert-icon { font-size: 1.3rem; line-height: 1.4; flex-shrink: 0; }
.ai-alert-title { font-weight: 700; color: #B42318; font-size: 1.02rem; margin-bottom: 0.15rem; }
.ai-alert-desc { color: #7A271A; font-size: 0.95rem; line-height: 1.5; }

.ai-section { margin-top: 1.1rem; }
.ai-section-header {
    display: flex; align-items: center; gap: 0.5rem; font-weight: 700;
    color: var(--ink); font-size: 1rem; margin-bottom: 0.45rem;
}
.ai-section-icon { font-size: 1.05rem; }
.ai-bullets { margin: 0; padding-left: 1.3rem; }
.ai-bullets li { color: #1E293B; font-size: 0.96rem; line-height: 1.55; margin-bottom: 0.3rem; }

.ai-footer-note {
    display: flex; gap: 0.6rem; background: #F0F6FF; border: 1px solid #D6E4FA;
    border-radius: 10px; padding: 0.75rem 0.9rem; margin-top: 1.1rem;
    font-size: 0.85rem; color: #375273; line-height: 1.5;
}

/* ---------- Event / validation cards ---------- */
.event-summary-card {
    background: #FFFFFF; border: 1px solid #D9E1EA; border-radius: 12px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 4px rgba(15,23,42,0.04); margin-bottom: 0.7rem;
}
.event-summary-title { color: var(--ink); font-size: 1.15rem; font-weight: 700; margin-bottom: 0.7rem; }
.event-badge {
    display: inline-block; padding: 0.22rem 0.55rem; border-radius: 999px;
    background: #FFF1F2; color: #C0262D; border: 1px solid #FECDD3;
    font-size: 0.78rem; font-weight: 600; margin-left: 0.35rem;
}
.validation-card {
    background: #FFFFFF; border: 1px solid #D9E1EA; border-radius: 12px;
    padding: 1rem 1.1rem; margin-top: 0.75rem; box-shadow: 0 1px 4px rgba(15,23,42,0.04);
}
.validation-header { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
.validation-title { color: var(--ink); font-size: 1.15rem; font-weight: 700; }
.validation-subtitle { color: var(--muted); font-size: 0.82rem; line-height: 1.45; margin-bottom: 0.75rem; }
.validation-pill { border-radius: 999px; padding: 0.3rem 0.65rem; font-size: 0.78rem; font-weight: 700; white-space: nowrap; }
.validation-pass { background: #ECFDF3; color: #027A48; border: 1px solid #ABEFC6; }
.validation-warn { background: #FFFAEB; color: #B54708; border: 1px solid #FEDF89; }
.validation-fail { background: #FEF3F2; color: #B42318; border: 1px solid #FECDCA; }

.check-row { display: flex; align-items: flex-start; gap: 0.65rem; padding: 0.55rem 0; border-top: 1px solid #EEF2F6; }
.check-icon {
    width: 1.25rem; height: 1.25rem; border-radius: 50%; display: inline-flex;
    align-items: center; justify-content: center; flex: 0 0 1.25rem; font-size: 0.72rem; font-weight: 700;
}
.check-pass { background: #D1FADF; color: #027A48; }
.check-warn { background: #FEF0C7; color: #B54708; }
.check-fail { background: #FEE4E2; color: #B42318; }
.check-name { color: #1E293B; font-size: 0.88rem; font-weight: 600; line-height: 1.35; }
.check-detail { color: var(--muted); font-size: 0.76rem; line-height: 1.35; margin-top: 0.12rem; }
.validation-note {
    margin-top: 0.65rem; padding: 0.65rem 0.75rem; border-radius: 8px;
    background: #F8FAFC; border: 1px solid #E2E8F0; color: #475569; font-size: 0.78rem; line-height: 1.45;
}
</style>
"""
