# Fraud Detection Pipeline

An end-to-end fraud detection system: raw transaction data is cleaned and
loaded into Postgres, a classifier is trained to handle severe class
imbalance, and predictions are served through a live dashboard.

**Live dashboard:** _TODO: add Streamlit Community Cloud link after deploying_

## Architecture

```
data/creditcard.csv
        |
        v
  src/ingest.py  ---->  Postgres (Supabase)  ---->  src/train_model.py  ---->  models/*.joblib
        (clean)              (transactions)              (SMOTE + RF)                |
                                    |                                                 v
                                    +----------------------------------------> src/app.py (Streamlit)
```

- **Ingest** (`src/ingest.py`) — loads the raw CSV, drops duplicate/null rows,
  and writes a clean `transactions` table to Postgres.
- **Train** (`src/train_model.py`) — pulls from Postgres, splits train/test
  *before* any resampling, balances the training set with SMOTE, trains a
  Random Forest classifier, and saves the model + scaler.
- **Serve** (`src/app.py`) — a Streamlit dashboard that scores a live sample
  of transactions from Postgres and lets you explore the precision/recall
  tradeoff at different decision thresholds.

## Stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), SQLAlchemy +
psycopg2, Postgres (Supabase free tier), Streamlit, joblib.

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) —
284,807 transactions, 492 fraudulent (0.17%). Features `V1`–`V28` are PCA
components of the original (undisclosed, for confidentiality) transaction
features; `Time` and `Amount` are raw.

## Key decisions

**Why SMOTE, and why only on the training set.** With fraud at 0.17% of
transactions, a model can hit 99.8% accuracy by never predicting fraud at
all — accuracy is useless here. SMOTE (Synthetic Minority Oversampling
Technique) generates synthetic fraud examples by interpolating between real
fraud cases and their nearest neighbors, giving the classifier a balanced
training signal instead of one dominated by the majority class. Critically,
resampling happens **after** the train/test split and only touches the
training set — resampling before the split lets synthetic points derived
from a test-set example leak into training (or vice versa), which inflates
evaluation metrics with numbers that won't hold up on real transactions.

**Why PR-AUC over ROC-AUC / accuracy.** ROC-AUC weighs true negatives
heavily, which on a 99.8%-legit dataset makes almost any model look good.
Precision-Recall AUC only scores how well the model ranks the minority
(fraud) class, which is the actual question a bank cares about: of the
transactions we flag, how many are really fraud (precision), and of the
real fraud, how much did we catch (recall)?

**Why the threshold is a dashboard control, not a hardcoded constant.**
Precision and recall trade off against each other, and where you want to
sit on that curve is a business decision, not a modeling one — a bank
willing to tolerate more false alerts to catch more fraud will pick a lower
threshold than one trying to minimize customer friction. The dashboard
exposes this as a slider so the tradeoff is explorable rather than baked in.

## Setup

1. **Create a Supabase project** (free tier) and grab the connection string
   from Project Settings → Database → Connection string (URI, "Session
   pooler" or direct connection both work).

2. **Clone and install:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```

3. **Configure credentials:**
   ```bash
   copy .env.example .env
   # edit .env and paste in your Supabase DATABASE_URL
   ```

4. **Download the dataset** from Kaggle and place `creditcard.csv` in `data/`.

5. **Run the pipeline:**
   ```bash
   cd src
   python ingest.py          # loads + cleans data, writes to Postgres
   python train_model.py     # trains model, prints metrics, saves to models/
   streamlit run app.py      # launches the dashboard
   ```

## Results

_TODO: fill in after running train_model.py — Random Forest test-set
precision/recall/PR-AUC._

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud),
pointed at this repo with `DATABASE_URL` set as a secret.
