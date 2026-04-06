import streamlit as st

st.set_page_config(
    page_title="ML Algorithms - Animated Presentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --navy:    #0F1B2D;
    --blue:    #1A56DB;
    --accent:  #00C2FF;
    --green:   #10B981;
    --orange:  #F59E0B;
    --red:     #EF4444;
    --purple:  #8B5CF6;
    --card-bg: #162032;
    --border:  #1E3A5F;
    --text:    #E2EAF4;
    --muted:   #7A9CC0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A1220 !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0F1B2D 0%, #1A3A5C 50%, #0F1B2D 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,194,255,0.15) 0%, transparent 70%);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(26,86,219,0.2) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    margin: 0 0 1.2rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,194,255,0.12);
    border: 1px solid rgba(0,194,255,0.3);
    color: var(--accent);
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
}

/* Algorithm Cards */
.algo-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.algo-card:hover { border-color: var(--accent); }
.algo-card-accent {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 12px 0 0 12px;
}
.algo-card-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 0.3rem 1rem;
}
.algo-card-desc {
    font-size: 0.88rem;
    color: var(--muted);
    margin: 0 0 0 1rem;
    line-height: 1.6;
}

/* Step boxes */
.step-box {
    background: #0A1220;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.step-num {
    background: var(--accent);
    color: var(--navy);
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.8rem;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text { font-size: 0.9rem; color: var(--text); line-height: 1.5; }

/* Formula box */
.formula-box {
    background: #0A1220;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    color: var(--accent);
    text-align: center;
    margin: 1rem 0;
    letter-spacing: 0.04em;
}

/* Comparison table */
.cmp-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
.cmp-table th {
    background: #1A3A5C;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
}
.cmp-table td {
    padding: 0.65rem 1rem;
    font-size: 0.88rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
}
.cmp-table tr:nth-child(even) td { background: #0A1220; }
.cmp-table tr:hover td { background: #1A2A3F; }
.badge-yes { color: #10B981; font-weight: 700; }
.badge-no  { color: #EF4444; font-weight: 700; }
.badge-med { color: #F59E0B; font-weight: 700; }

/* Metric cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label { font-size: 0.82rem; color: var(--muted); margin-top: 0.2rem; }

/* Section title */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    color: #fff;
    border-left: 3px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar nav item */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 0.3rem;
    font-size: 0.9rem;
    transition: background 0.15s;
}
.nav-item:hover { background: rgba(0,194,255,0.08); }
.nav-item.active { background: rgba(0,194,255,0.12); border-left: 3px solid var(--accent); }

/* Progress bar override */
.stProgress > div > div { background-color: var(--accent) !important; }

/* Streamlit selectbox / radio */
div[data-baseweb="select"] { background: #0A1220 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-family:"Space Mono",monospace; font-size:1.1rem; color:#00C2FF; font-weight:700;'>🧠 ML Algorithms</div>
        <div style='font-size:0.75rem; color:#7A9CC0; margin-top:0.3rem;'>Animated Presentation</div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Home": "home",
        "📐  Linear Regression": "linear",
        "🌳  Decision Trees": "decision",
        "⭕  K-Means Clustering": "kmeans",
        "🔷  Support Vector Machine": "svm",
        "🧠  Neural Networks": "neural",
        "🌲  Random Forest": "forest",
        "📊  Algorithm Comparison": "compare",
    }

    if "page" not in st.session_state:
        st.session_state.page = "home"

    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#7A9CC0; padding:0 0.5rem;'>Built with Streamlit + Python<br>© 2025 ML Project</div>", unsafe_allow_html=True)

# ── Page Router ───────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "home":
    from pages import home; home.render()
elif page == "linear":
    from pages import linear; linear.render()
elif page == "decision":
    from pages import decision; decision.render()
elif page == "kmeans":
    from pages import kmeans; kmeans.render()
elif page == "svm":
    from pages import svm; svm.render()
elif page == "neural":
    from pages import neural; neural.render()
elif page == "forest":
    from pages import forest; forest.render()
elif page == "compare":
    from pages import compare; compare.render()
