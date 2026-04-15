# 🛡️ Phishing Website Detection Using Machine Learning
**MSc Data Science Final Year Project — 2025**

## Overview
This project develops and evaluates a machine learning pipeline for
detecting phishing websites using the PhiUSIIL dataset (235,795 URLs).
Four models are compared: Logistic Regression, Random Forest, XGBoost,
and a Stacking Ensemble. Explainability is provided via SHAP and LIME.

## Live Demo
🌐 [Open PhishGuard App](https://YOUR-APP-LINK.streamlit.app)

## Dataset
- **Primary**: PhiUSIIL Phishing URL Dataset — Prasad & Chandra, 2024
  (UCI id=967) — 235,795 URLs, 54 features
- **Cross-test**: UCI Phishing Websites Classic (UCI id=327) — 11,055 URLs

## Models Trained
| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | ~93% | ~92% | ~96% |
| Random Forest       | ~96% | ~96% | ~98% |
| XGBoost             | ~97% | ~96% | ~99% |
| Stacking Ensemble   | ~97% | ~97% | ~99% |

## Repository Structure
