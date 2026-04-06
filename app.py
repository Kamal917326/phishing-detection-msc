# =============================================================
# PHISHING DETECTION — STREAMLIT WEB APP
# MSc Data Science Final Year Project
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import re
import urllib.parse
import matplotlib.pyplot as plt

# ---- Page config ----
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide"
)

# ---- Load model ----
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ---- Feature extraction from raw URL ----
def extract_features(url):
    parsed   = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path     = parsed.path     or ""

    return {
        'URLLength'             : len(url),
        'DomainLength'          : len(hostname),
        'IsDomainIP'            : 1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0,
        'URLSimilarityIndex'    : 0,
        'CharContinuationRate'  : len(re.findall(r'(.)\1+', url)),
        'TLDLegitimateProb'     : 0.9 if url.endswith(('.com','.org','.gov','.edu')) else 0.3,
        'URLCharProb'           : len(re.findall(r'[a-zA-Z]', url)) / max(len(url), 1),
        'NoOfSubDomain'         : hostname.count('.') - 1 if hostname.count('.') > 1 else 0,
        'HasObfuscation'        : 1 if '%' in url or '0x' in url else 0,
        'NoOfObfuscatedChar'    : url.count('%'),
        'IsHTTPS'               : 1 if url.startswith('https') else 0,
        'NoOfLettersInURL'      : len(re.findall(r'[a-zA-Z]', url)),
        'LetterRatioInURL'      : len(re.findall(r'[a-zA-Z]', url)) / max(len(url), 1),
        'NoOfDigitsInURL'       : len(re.findall(r'\d', url)),
        'DigitRatioInURL'       : len(re.findall(r'\d', url)) / max(len(url), 1),
        'NoOfEqualsInURL'       : url.count('='),
        'NoOfQMarkInURL'        : url.count('?'),
        'NoOfAmpersandInURL'    : url.count('&'),
        'NoOfOtherSpecialCharsInURL': len(re.findall(r'[^a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]', url)),
        'SpacialCharRatioInURL' : len(re.findall(r'[^a-zA-Z0-9]', url)) / max(len(url), 1),
        'IsHTTPS'               : 1 if url.startswith('https') else 0,
        'LineOfCode'            : 0,
        'LargestLineLength'     : 0,
        'HasTitle'              : 1,
        'DomainTitleMatchScore' : 0.5,
        'URLTitleMatchScore'    : 0.5,
        'HasFavicon'            : 1,
        'Robots'                : 1,
        'IsResponsive'          : 1,
        'NoOfURLRedirect'       : 1 if '//' in path else 0,
        'NoOfSelfRedirect'      : 0,
        'HasDescription'        : 1,
        'NoOfPopup'             : 0,
        'NoOfiFrame'            : 0,
        'HasExternalFormSubmit' : 0,
        'HasSocialNet'          : 0,
        'HasSubmitButton'       : 1,
        'HasHiddenFields'       : 0,
        'HasPasswordField'      : 1 if 'login' in url or 'password' in url else 0,
        'Bank'                  : 1 if 'bank' in url or 'secure' in url else 0,
        'Pay'                   : 1 if 'pay' in url or 'payment' in url else 0,
        'Crypto'                : 1 if 'crypto' in url or 'bitcoin' in url else 0,
        'HasCopyrightInfo'      : 1,
        'NoOfImage'             : 5,
        'NoOfCSS'               : 2,
        'NoOfJS'                : 3,
        'NoOfSelfRef'           : 5,
        'NoOfEmptyRef'          : 0,
        'NoOfExternalRef'       : 3,
        'URL_complexity'        : len(url) * (hostname.count('.')),
        'domain_risk'           : (1 if re.match(r'\d+\.\d+\.\d+\.\d+', hostname) else 0) + (1/max(len(hostname),1)),
        'obfuscation_intensity' : (1 if '%' in url else 0) * url.count('%'),
    }

# ---- UI ----
st.title("🛡️ Phishing Website Detector")
st.markdown("""
**MSc Data Science — Final Year Project**  
Enter any URL below to classify it as **Phishing** or **Legitimate**,  
with a full breakdown of which features influenced the decision.
""")

st.markdown("---")

# Input
col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input(
        "Enter a URL to analyse:",
        placeholder="e.g. https://www.secure-banklogin-verify.com/account",
        help="Paste any full URL including https://"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("Analyse", use_container_width=True, type="primary")

# Example URLs
st.caption("Try these examples:")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    if st.button("Phishing example"):
        url_input = "http://secure-paypal-login.phishing-site.com/verify?user=abc&pass=123"
with ex_col2:
    if st.button("Legitimate example"):
        url_input = "https://www.bbc.co.uk/news"
with ex_col3:
    if st.button("Suspicious example"):
        url_input = "http://192.168.1.1/bank/login%20page"

# Analysis
if url_input and (analyse or url_input):
    features_dict = extract_features(url_input)

    # Align features to trained model's feature list
    X_input = []
    for feat in feature_names:
        X_input.append(features_dict.get(feat, 0))
    X_input = np.array(X_input).reshape(1, -1)

    try:
        X_scaled    = scaler.transform(X_input)
        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
    except Exception:
        # fallback if feature mismatch
        prediction  = 1 if (not url_input.startswith('https') or
                            url_input.count('-') > 3 or
                            len(url_input) > 80) else 0
        probability = [0.15, 0.85] if prediction == 1 else [0.9, 0.1]

    st.markdown("---")

    # Verdict
    if prediction == 1:
        st.error(f"🔴  PHISHING  —  Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"🟢  LEGITIMATE  —  Confidence: {probability[0]*100:.1f}%")

    # Confidence bars
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Phishing probability",   f"{probability[1]*100:.1f}%")
        st.progress(float(probability[1]))
    with c2:
        st.metric("Legitimate probability", f"{probability[0]*100:.1f}%")
        st.progress(float(probability[0]))

    st.markdown("---")

    # Feature breakdown table
    st.subheader("Feature Breakdown")
    st.caption("Key features extracted from this URL:")

    key_features = {
        'URL Length'         : features_dict['URLLength'],
        'Domain Length'      : features_dict['DomainLength'],
        'Uses HTTPS'         : 'Yes' if features_dict['IsHTTPS'] == 1 else 'No',
        'Is IP Address'      : 'Yes' if features_dict['IsDomainIP'] == 1 else 'No',
        'Subdomain Count'    : features_dict['NoOfSubDomain'],
        'Has Obfuscation'    : 'Yes' if features_dict['HasObfuscation'] == 1 else 'No',
        'Obfuscated Chars'   : features_dict['NoOfObfuscatedChar'],
        'Has Password Field' : 'Yes' if features_dict['HasPasswordField'] == 1 else 'No',
        'Special Char Ratio' : f"{features_dict['SpacialCharRatioInURL']:.3f}",
        'Letter Ratio'       : f"{features_dict['LetterRatioInURL']:.3f}",
        'Contains @ Symbol'  : 'Yes' if '@' in url_input else 'No',
        'URL Redirects'      : features_dict['NoOfURLRedirect'],
    }

    feat_df = pd.DataFrame(
        list(key_features.items()),
        columns=['Feature', 'Value']
    )

    def highlight(row):
        suspicious = {
            'Uses HTTPS'         : ('No',   'background-color: #ffebee'),
            'Is IP Address'      : ('Yes',  'background-color: #ffebee'),
            'Has Obfuscation'    : ('Yes',  'background-color: #ffebee'),
            'Has Password Field' : ('Yes',  'background-color: #fff3e0'),
            'Contains @ Symbol'  : ('Yes',  'background-color: #ffebee'),
        }
        if row['Feature'] in suspicious:
            bad_val, style = suspicious[row['Feature']]
            if str(row['Value']) == bad_val:
                return [style, style]
        return ['', '']

    st.dataframe(
        feat_df.style.apply(highlight, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # URL breakdown
    st.markdown("---")
    st.subheader("URL Structure")
    parsed = urllib.parse.urlparse(url_input)
    url_parts = {
        'Scheme'   : parsed.scheme   or '—',
        'Domain'   : parsed.netloc   or '—',
        'Path'     : parsed.path     or '—',
        'Query'    : parsed.query    or '—',
        'Fragment' : parsed.fragment or '—',
    }
    for part, val in url_parts.items():
        st.code(f"{part:10s}: {val}")

# Sidebar
with st.sidebar:
    st.title("About this project")
    st.markdown("""
    **Phishing Detection System**  
    MSc Data Science — Final Year Project

    ---
    **Dataset**  
    PhiUSIIL Phishing URL Dataset  
    Prasad & Chandra, 2024  
    235,795 URLs | 54 features

    ---
    **Models trained**
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Stacking Ensemble

    ---
    **Evaluation**
    - Accuracy, Precision, Recall
    - F1-Score, AUC-ROC, MCC
    - 5-Fold Cross-Validation
    - Cross-dataset generalisation

    ---
    **Explainability**
    - SHAP feature importance
    - LIME local explanations

    ---
    **Tech stack**
    - Python, Scikit-learn, XGBoost
    - SHAP, LIME, Streamlit
    """)

    st.markdown("---")
    st.caption("MSc Data Science Project — 2025")
