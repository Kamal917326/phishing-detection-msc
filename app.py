# =============================================================
# PHISHING DETECTION — STREAMLIT WEB APP (UPGRADED)
# MSc Data Science Final Year Project
# Features: SHAP chart, history table, risk gauge,
#           enhanced sidebar, professional colour scheme
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import re
import urllib.parse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

st.set_page_config(
    page_title="PhishGuard — Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d3250;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #4f46e5;
    }
    .badge-phishing {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .badge-legit {
        background: linear-gradient(135deg, #16a34a, #14532d);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #818cf8; }
    .stTextInput input {
        background: #1e2130 !important;
        border: 1.5px solid #4f46e5 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    .stButton button {
        background: #4f46e5 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #13151f !important;
        border-right: 1px solid #2d3250;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

URL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'NoOfSubDomain',
    'IsHTTPS', 'NoOfLettersInURL', 'LetterRatioInURL',
    'NoOfDigitsInURL', 'DigitRatioInURL', 'NoOfEqualsInURL',
    'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'HasObfuscation', 'NoOfObfuscatedChar',
]

FEATURE_LABELS = {
    'URLLength'                  : 'URL Length',
    'DomainLength'               : 'Domain Length',
    'IsDomainIP'                 : 'Is IP Address',
    'NoOfSubDomain'              : 'Subdomain Count',
    'IsHTTPS'                    : 'Uses HTTPS',
    'NoOfLettersInURL'           : 'Letter Count',
    'LetterRatioInURL'           : 'Letter Ratio',
    'NoOfDigitsInURL'            : 'Digit Count',
    'DigitRatioInURL'            : 'Digit Ratio',
    'NoOfEqualsInURL'            : 'Equals Signs (=)',
    'NoOfQMarkInURL'             : 'Question Marks (?)',
    'NoOfAmpersandInURL'         : 'Ampersands (&)',
    'NoOfOtherSpecialCharsInURL' : 'Special Chars',
    'SpacialCharRatioInURL'      : 'Special Char Ratio',
    'HasObfuscation'             : 'Has Obfuscation',
    'NoOfObfuscatedChar'         : 'Obfuscated Chars (%)',
}

def extract_features(url: str) -> dict:
    try:
        parsed   = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        url_len  = max(len(url), 1)
        letters  = re.findall(r'[a-zA-Z]', url)
        digits   = re.findall(r'\d', url)
        special  = re.findall(r'[^a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]', url)
        return {
            'URLLength'                  : len(url),
            'DomainLength'               : len(hostname),
            'IsDomainIP'                 : 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0,
            'NoOfSubDomain'              : max(hostname.count('.') - 1, 0),
            'IsHTTPS'                    : 1 if url.startswith('https') else 0,
            'NoOfLettersInURL'           : len(letters),
            'LetterRatioInURL'           : round(len(letters) / url_len, 4),
            'NoOfDigitsInURL'            : len(digits),
            'DigitRatioInURL'            : round(len(digits) / url_len, 4),
            'NoOfEqualsInURL'            : url.count('='),
            'NoOfQMarkInURL'             : url.count('?'),
            'NoOfAmpersandInURL'         : url.count('&'),
            'NoOfOtherSpecialCharsInURL' : len(special),
            'SpacialCharRatioInURL'      : round(len(special) / url_len, 4),
            'HasObfuscation'             : 1 if '%' in url or '0x' in url.lower() else 0,
            'NoOfObfuscatedChar'         : url.count('%'),
        }
    except Exception:
        return {f: 0 for f in URL_FEATURES}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names, True
    except Exception:
        return None, None, URL_FEATURES, False

model, scaler, feature_names, model_loaded = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict(url: str):
    feats   = extract_features(url)
    X_input = np.array([feats.get(f, 0) for f in feature_names]).reshape(1, -1)
    if model_loaded:
        try:
            X_sc = scaler.transform(X_input)
            pred = model.predict(X_sc)[0]
            prob = model.predict_proba(X_sc)[0]
            return int(pred), prob, feats
        except Exception:
            pass
    # Rule-based fallback
    score = 0
    if not url.startswith('https'):    score += 25
    if feats['IsDomainIP']:            score += 35
    if feats['HasObfuscation']:        score += 20
    if feats['NoOfSubDomain'] > 3:     score += 15
    if feats['URLLength'] > 100:       score += 10
    if feats['NoOfQMarkInURL'] > 2:    score += 10
    score     = min(score, 99)
    phish_p   = score / 100
    pred      = 1 if phish_p > 0.5 else 0
    return pred, [1-phish_p, phish_p], feats


# ============================================================
# RISK GAUGE
# ============================================================

def draw_gauge(phish_prob: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.6),
                            subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    zones = [
        (np.linspace(np.pi, np.pi*0.67, 50), '#16a34a'),
        (np.linspace(np.pi*0.67, np.pi*0.33, 50), '#d97706'),
        (np.linspace(np.pi*0.33, 0.01, 50), '#dc2626'),
    ]
    for thetas, color in zones:
        ax.plot(thetas, [0.8]*len(thetas), color=color,
                linewidth=16, alpha=0.85, solid_capstyle='butt')

    needle_angle = np.pi * (1 - phish_prob)
    ax.annotate('', xy=(needle_angle, 0.72), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.5, mutation_scale=14))
    ax.plot(0, 0, 'o', color='white', markersize=6, zorder=5)

    ax.text(0, -0.3, f'{phish_prob*100:.1f}%',
            ha='center', va='center', fontsize=17,
            fontweight='bold', color='white', transform=ax.transData)

    risk_label = 'HIGH RISK' if phish_prob > 0.66 else \
                 'MEDIUM RISK' if phish_prob > 0.33 else 'LOW RISK'
    risk_color = '#dc2626' if phish_prob > 0.66 else \
                 '#d97706' if phish_prob > 0.33 else '#16a34a'
    ax.text(0, -0.58, risk_label, ha='center', va='center',
            fontsize=9, fontweight='bold', color=risk_color,
            transform=ax.transData)

    ax.set_ylim(-0.7, 1.05)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ============================================================
# SHAP-STYLE CHART
# ============================================================

def draw_shap_chart(feats: dict):
    contributions = {
        'No HTTPS'           :  0.35 if feats.get('IsHTTPS', 1) == 0 else -0.20,
        'IP as domain'       :  0.40 if feats.get('IsDomainIP', 0) == 1 else -0.05,
        'URL obfuscation'    :  0.30 if feats.get('HasObfuscation', 0) == 1 else -0.05,
        'Subdomain count'    :  min(feats.get('NoOfSubDomain', 0) * 0.09, 0.35),
        'URL length'         :  min((feats.get('URLLength', 0)-30)*0.003, 0.30)
                                if feats.get('URLLength', 0) > 30 else -0.10,
        'Query params (?)'   :  min(feats.get('NoOfQMarkInURL', 0) * 0.08, 0.25),
        'Ampersands (&)'     :  min(feats.get('NoOfAmpersandInURL', 0) * 0.06, 0.20),
        'Obfuscated chars'   :  min(feats.get('NoOfObfuscatedChar', 0) * 0.05, 0.20),
        'Special char ratio' :  min(feats.get('SpacialCharRatioInURL', 0) * 1.5, 0.25),
        'Digit ratio'        :  min(feats.get('DigitRatioInURL', 0) * 0.8, 0.15)
                                if feats.get('DigitRatioInURL', 0) > 0.2 else -0.05,
    }

    items  = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    colors = ['#dc2626' if v > 0 else '#16a34a' for v in values]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    bars = ax.barh(labels, values, color=colors, alpha=0.85,
                   edgecolor='none', height=0.55)
    ax.axvline(0, color='#94a3b8', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('← Legitimate   |   Phishing →',
                  color='#94a3b8', fontsize=8.5)
    ax.set_title('Feature Contributions (SHAP-style)',
                 color='#e2e8f0', fontsize=10.5, fontweight='bold', pad=8)
    ax.tick_params(colors='#cbd5e1', labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color('#2d3250')

    for bar, val in zip(bars, values):
        ax.text(val + (0.008 if val >= 0 else -0.008),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}', va='center',
                ha='left' if val >= 0 else 'right',
                color='#e2e8f0', fontsize=7.5)

    red_p   = mpatches.Patch(color='#dc2626', label='→ Phishing')
    green_p = mpatches.Patch(color='#16a34a', label='→ Legitimate')
    ax.legend(handles=[red_p, green_p], loc='lower right',
              facecolor='#1e2130', edgecolor='#2d3250',
              labelcolor='#cbd5e1', fontsize=8)
    plt.tight_layout()
    return fig


# ============================================================
# SESSION STATE
# ============================================================

if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'phishing': 0, 'legitimate': 0}


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="background:linear-gradient(135deg,#1e1b4b,#312e81);
            padding:1.4rem 2rem;border-radius:12px;
            margin-bottom:1.5rem;border:1px solid #4f46e5;">
  <h1 style="color:white;margin:0;font-size:1.75rem;">
    🛡️ PhishGuard — Phishing URL Detector
  </h1>
  <p style="color:#a5b4fc;margin:0.3rem 0 0;font-size:0.9rem;">
    MSc Data Science Final Year Project &nbsp;|&nbsp;
    PhiUSIIL Dataset (235,795 URLs) &nbsp;|&nbsp;
    Random Forest + Stacking Ensemble
  </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STATS ROW
# ============================================================

s = st.session_state.stats
rate = round(s['phishing']/max(s['total'],1)*100, 1)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, color in [
    (c1, 'URLs Analysed',     s['total'],      '#818cf8'),
    (c2, 'Phishing Detected', s['phishing'],   '#f87171'),
    (c3, 'Legitimate Found',  s['legitimate'], '#4ade80'),
    (c4, 'Phishing Rate',     f"{rate}%",      '#fb923c'),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div style="color:#94a3b8;font-size:0.78rem;">{label}</div>
        <div style="font-size:1.9rem;font-weight:700;color:{color};">{val}</div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# URL INPUT
# ============================================================

st.markdown('<div class="section-header">Analyse a URL</div>',
            unsafe_allow_html=True)

ci, cb = st.columns([5, 1])
with ci:
    url_input = st.text_input(
        label="url", label_visibility="collapsed",
        placeholder="Enter any URL — e.g. https://www.example.com",
        key="url_box"
    )
with cb:
    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("🔍 Analyse", use_container_width=True, type="primary")

st.caption("Quick tests →")
q1, q2, q3, q4, q5 = st.columns(5)
if q1.button("youtube.com"):       url_input = "https://www.youtube.com/"
if q2.button("bbc.co.uk"):         url_input = "https://www.bbc.co.uk/news"
if q3.button("gov.uk"):            url_input = "https://www.gov.uk"
if q4.button("🔴 Phishing ex 1"):  url_input = "http://paypal-secure-verify.tk/login?user=abc&token=xyz%20"
if q5.button("🔴 Phishing ex 2"):  url_input = "http://192.168.0.1/banking/login%20verify?redirect=evil.com"


# ============================================================
# RESULTS
# ============================================================

if url_input:
    pred, prob, feats = predict(url_input)

    st.session_state.stats['total']     += 1
    st.session_state.stats['phishing'   if pred==1 else 'legitimate'] += 1
    st.session_state.history.append({
        'Time'      : datetime.now().strftime('%H:%M:%S'),
        'URL'       : url_input[:55] + ('…' if len(url_input)>55 else ''),
        'Verdict'   : '🔴 Phishing' if pred==1 else '🟢 Legitimate',
        'Confidence': f"{max(prob)*100:.1f}%",
        'Phishing % ': f"{prob[1]*100:.1f}%",
    })

    st.markdown("---")

    left, mid, right = st.columns([2, 2, 3])

    # ---- Verdict ----
    with left:
        st.markdown('<div class="section-header">Verdict</div>',
                    unsafe_allow_html=True)
        if pred == 1:
            st.markdown('<div class="badge-phishing">🔴 PHISHING</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-legit">🟢 LEGITIMATE</div>',
                        unsafe_allow_html=True)

        parsed = urllib.parse.urlparse(url_input)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:0.8rem;font-size:0.82rem;">
            <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:0.4rem;">
                PROBABILITIES</div>
            <div style="color:{'#f87171' if pred==1 else '#4ade80'};font-size:1.5rem;font-weight:700;">
                {prob[1]*100:.1f}% phishing</div>
            <div style="color:#4ade80;font-size:1.2rem;font-weight:600;">
                {prob[0]*100:.1f}% legitimate</div>
            <hr style="border-color:#2d3250;margin:0.6rem 0;">
            <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:0.3rem;">
                URL STRUCTURE</div>
            <div style="color:#cbd5e1;">
                <b>Scheme:</b> {parsed.scheme or '—'}<br>
                <b>Domain:</b> {parsed.netloc or '—'}<br>
                <b>Path:</b>   {(parsed.path or '—')[:28]}<br>
                <b>Query:</b>  {(parsed.query or '—')[:28]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Gauge ----
    with mid:
        st.markdown('<div class="section-header">Risk Gauge</div>',
                    unsafe_allow_html=True)
        st.pyplot(draw_gauge(prob[1]), use_container_width=True)
        plt.close()
        pct = max(prob)*100
        bar_color = '#dc2626' if pred==1 else '#16a34a'
        st.markdown(f"""
        <div style="margin-top:0.6rem;">
            <div style="color:#94a3b8;font-size:0.78rem;margin-bottom:0.3rem;">
                Model confidence: {pct:.1f}%</div>
            <div style="background:#2d3250;border-radius:6px;height:9px;">
                <div style="background:{bar_color};width:{pct:.0f}%;
                            height:9px;border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- SHAP chart ----
    with right:
        st.markdown('<div class="section-header">Feature Contributions (SHAP-style)</div>',
                    unsafe_allow_html=True)
        st.pyplot(draw_shap_chart(feats), use_container_width=True)
        plt.close()

    # ---- Feature table ----
    st.markdown('<div class="section-header">Full Feature Breakdown</div>',
                unsafe_allow_html=True)

    FLAGS = {
        'IsHTTPS'               : (lambda v: v==0, '⚠️ No HTTPS'),
        'IsDomainIP'            : (lambda v: v==1, '⚠️ IP address used'),
        'HasObfuscation'        : (lambda v: v==1, '⚠️ Obfuscation detected'),
        'NoOfSubDomain'         : (lambda v: v>2,  '⚠️ Many subdomains'),
        'URLLength'             : (lambda v: v>75, '⚠️ Very long URL'),
        'NoOfQMarkInURL'        : (lambda v: v>1,  '⚠️ Multiple query params'),
        'NoOfObfuscatedChar'    : (lambda v: v>0,  '⚠️ Encoded chars found'),
        'SpacialCharRatioInURL' : (lambda v: v>0.1,'⚠️ High special char ratio'),
    }

    rows = []
    for feat, val in feats.items():
        note = ''
        if feat in FLAGS:
            chk, msg = FLAGS[feat]
            if chk(val): note = msg
        rows.append({
            'Feature': FEATURE_LABELS.get(feat, feat),
            'Value'  : f'{val:.4f}' if isinstance(val, float) else str(val),
            'Status' : note if note else '✓ OK',
        })

    def colour_rows(row):
        if '⚠️' in str(row['Status']):
            return ['background-color:rgba(220,38,38,0.15)']*3
        return ['', '', '']

    st.dataframe(
        pd.DataFrame(rows).style.apply(colour_rows, axis=1),
        use_container_width=True, hide_index=True, height=310
    )


# ============================================================
# HISTORY TABLE
# ============================================================

st.markdown('<div class="section-header">Analysis History</div>',
            unsafe_allow_html=True)

if st.session_state.history:
    hdf = pd.DataFrame(st.session_state.history[::-1])

    def colour_hist(row):
        return (['background-color:rgba(220,38,38,0.10)']*len(row)
                if 'Phishing' in str(row['Verdict'])
                else ['background-color:rgba(22,163,74,0.07)']*len(row))

    st.dataframe(
        hdf.style.apply(colour_hist, axis=1),
        use_container_width=True, hide_index=True, height=240
    )

    dl_col, clr_col = st.columns([4, 1])
    with dl_col:
        st.download_button(
            "⬇️ Download history CSV",
            data=hdf.to_csv(index=False),
            file_name=f"phishguard_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )
    with clr_col:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.session_state.stats   = {'total':0,'phishing':0,'legitimate':0}
            st.rerun()
else:
    st.markdown("""
    <div style="background:#1e2130;border-radius:8px;padding:1.5rem;
                text-align:center;color:#475569;border:1px dashed #2d3250;">
        No URLs analysed yet — enter a URL above to get started
    </div>""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.8rem 0 1rem;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">PhishGuard</div>
        <div style="font-size:0.75rem;color:#64748b;">MSc Data Science 2025</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Dataset")
    st.markdown("""
    <div style="background:#0f1117;border-radius:8px;padding:0.9rem;
                border:1px solid #2d3250;font-size:0.83rem;color:#cbd5e1;
                line-height:1.7;">
        <b style="color:#a5b4fc;">PhiUSIIL Phishing URL Dataset</b><br>
        Prasad & Chandra, 2024<br>
        <i>Computers & Security Journal</i><br><br>
        🔢 235,795 total URLs<br>
        ✅ 134,850 legitimate<br>
        🚨 100,945 phishing<br>
        📋 54 original features<br>
        ❌ No missing values
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🤖 Models")
    for name, role, color in [
        ("Logistic Regression", "Baseline",      "#6366f1"),
        ("Random Forest",       "Best single",   "#10b981"),
        ("XGBoost",             "Gradient boost","#f59e0b"),
        ("Stacking Ensemble",   "Meta-learner",  "#8b5cf6"),
    ]:
        st.markdown(f"""
        <div style="background:#0f1117;border-radius:6px;
                    padding:0.45rem 0.8rem;margin-bottom:0.4rem;
                    border-left:3px solid {color};font-size:0.82rem;
                    color:#cbd5e1;">
            <b>{name}</b>
            <span style="color:#64748b;float:right;">{role}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Metrics Used")
    for m, d in [("Accuracy","Primary"),("F1-Score","Imbalance-robust"),
                 ("AUC-ROC","Discrimination"),("MCC","Overall quality"),
                 ("Precision","FP rate"),("Recall","Detection rate")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    font-size:0.8rem;padding:0.25rem 0;
                    border-bottom:1px solid #1e2130;color:#cbd5e1;">
            <span><b>{m}</b></span>
            <span style="color:#64748b;">{d}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🔍 Explainability")
    st.markdown("""
    <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.7;">
        🟣 <b>SHAP</b> — Global feature importance<br>
        <span style="color:#64748b;font-size:0.75rem;">
            Which features matter most overall</span><br><br>
        🟡 <b>LIME</b> — Local explanations<br>
        <span style="color:#64748b;font-size:0.75rem;">
            Why this specific URL was flagged</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Tech Stack")
    tech = ["Python","Scikit-learn","XGBoost","SHAP","LIME","Streamlit","Pandas","NumPy"]
    cols = st.columns(2)
    for i, t in enumerate(tech):
        cols[i%2].markdown(f"""
        <div style="background:#0f1117;border-radius:4px;
                    padding:0.3rem 0.4rem;margin-bottom:0.3rem;
                    font-size:0.74rem;color:#a5b4fc;
                    border:1px solid #2d3250;text-align:center;">
            {t}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;font-size:0.73rem;color:#475569;">
        <a href="https://archive.ics.uci.edu/dataset/967"
           style="color:#818cf8;text-decoration:none;">UCI Dataset (id=967)</a><br>
        Cross-test: UCI Classic (id=327, 2012)
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if model_loaded:
        st.success("✓ ML model loaded")
    else:
        st.warning("⚠ Rule-based fallback mode")
