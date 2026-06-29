---
title: Hallucination Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Hallucination Dataset Comparison Dashboard

An interactive dashboard to explore and compare 4 hallucination/fact-checking datasets used in NLP research.

## Live Demo
🚀 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/naruladhruv2006/hallucination-dashboard)

## Datasets
| Dataset | Size | Labels | Notes |
|---|---|---|---|
| PolyFever | 77,971 | True / False | Multilingual |
| WikiHades | 8,754 | 0 / 1 | Wikipedia sentences |
| FactCHD | 51,383 | FACTUAL / NON-FACTUAL | Rich metadata |
| HaluEval | 4,507 | no / yes | Chatbot responses |

## Features
- Label distribution per dataset
- Text length analysis
- Missing data check
- Interactive chart selector via Gradio

## How to Run Locally
pip install -r requirements.txt
python app.py

## Built With
- Python
- Gradio
- HuggingFace Datasets
- Matplotlib
- Pandas
