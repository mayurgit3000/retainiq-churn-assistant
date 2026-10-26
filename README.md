# RetainIQ · Churn Retention Assistant

A retrieval-augmented (RAG) assistant for a telecom company's retention team, built on top of **ChurnSense**, a churn prediction model trained on 50,000 customers. Agents ask questions in plain English and get answers grounded in the model's own findings, with the sources cited.

![RetainIQ web demo](screenshots/web-demo.png)

## The pipeline

Customer data (50K records, 22 features) → **ChurnSense** model → **SHAP** explanations → **RetainIQ** assistant → retention action

### ChurnSense model

- Compared Logistic Regression, XGBoost and Random Forest. Chose **Logistic Regression** (AUC 0.90, recall 94%) because it missed the fewest churners and is fully interpretable.
- Lowered the decision threshold from 0.50 to 0.35 to favour recall, because missing a churner costs more than a wasted retention offer.
- SHAP shows contract type is by far the strongest driver: month-to-month customers churn at 45.2% vs 2.4% on annual contracts.

### RetainIQ assistant

- Knowledge base: three documents (EDA and cohort analysis, SHAP and model performance, retention playbook).
- Retrieves the most relevant passages for each question and sends only those to the LLM, which must answer from them and cite them.
- **Risk score interpreter:** paste a ChurnSense probability (0 to 1) to get the risk tier, recommended action and offer.
- Answers outside the knowledge base are flagged instead of invented.

## Two versions

| | `app.py` (Streamlit) | `web-demo/index.html` |
|---|---|---|
| Retrieval | FAISS vector search with `all-MiniLM-L6-v2` embeddings | Keyword retrieval over the same knowledge base |
| Runs | Locally, with Python | In any browser, no install |
| Without an API key | Shows a setup message | Demo mode: shows the retrieved passages |

### Run the Streamlit app

```bash
pip install -r requirements.txt
cp .env.example .env            # then add your OpenRouter key
python ingest.py                # builds the FAISS index from the documents
streamlit run app.py
```

You can also put the key in `.streamlit/secrets.toml` (see `secrets.toml.example`). Never commit a real key.

### Run the web demo

Open `web-demo/index.html`. Paste an [OpenRouter](https://openrouter.ai) key in the header for written answers, or leave it empty to see the retrieval step on its own.

## Built with

Python · Streamlit · FAISS · sentence-transformers (all-MiniLM-L6-v2) · Claude 3 Haiku via OpenRouter · Logistic Regression, XGBoost, Random Forest · SHAP

## Responsible AI

GDPR Article 22: the model flags customers but never makes an adverse decision without human review. Gender and region carry near-zero SHAP weight and are being removed in v2.

