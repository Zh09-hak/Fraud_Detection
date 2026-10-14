import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# ==========================================
# PAGE CONFIG — must be the FIRST Streamlit command
# ==========================================
st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

# ==========================================
# CACHED RESOURCES (loaded once, reused on every rerun)
# ==========================================

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    """Model, threshold and SHAP explainer are not serializable -> cache_resource."""
    model = joblib.load('artifacts/xgb_model.pkl')
    threshold = joblib.load('artifacts/optimal_threshold.pkl')
    explainer = shap.TreeExplainer(model)
    return model, threshold, explainer


@st.cache_data(show_spinner=False)
def load_test_data():
    """Test set for the demo picker and for the metrics block."""
    X_test = pd.read_pickle('artifacts/X_test.pkl')
    y_test = pd.read_pickle('artifacts/y_test.pkl')
    return X_test, y_test


@st.cache_data(show_spinner=False)
def get_test_predictions():
    """Cache batch predictions on the full test set (56 962 rows).
    Without this, Streamlit recomputes them on EVERY interaction."""
    model, _, _ = load_resources()
    X_test, _ = load_test_data()
    return model.predict_proba(X_test)[:, 1]


model, threshold, explainer = load_resources()
X_test, y_test = load_test_data()

FEATURES = list(X_test.columns)   # exact training column order

# ==========================================
# HEADER
# ==========================================
st.title("🛡️ FraudGuard — Credit Card Fraud Detection with XAI")
st.caption(
    f"XGBoost model · Decision threshold: **{threshold:.4f}** "
    f"(tuned on the PR-curve for recall ≥ 0.80)"
)

# ==========================================
# HUMAN-READABLE LABELS
# V1..V28 are PCA-anonymized by design -> no business names exist.
# ==========================================
FEATURE_LABELS = {
    'Hour': 'Time of day',
    'Log_Amount': 'Transaction amount',
}


def label_for(f):
    return FEATURE_LABELS.get(f, f"Anonymized pattern {f}")


def amount_to_dollars(log_amount):
    """Model works with log1p internally; humans need dollars."""
    return np.expm1(float(log_amount))


# ==========================================
# SIDEBAR: INPUT
# ==========================================
st.sidebar.header("Choose a transaction")

mode = st.sidebar.radio(
    "Mode",
    ["📂 Real examples from dataset", "🎛️ Custom scenario"],
    help="Pick a real transaction, or build your own scenario."
)

if mode == "📂 Real examples from dataset":
    proba_all = get_test_predictions()

    # 8 confident frauds + 8 confident legits = best cases for a live demo
    fraud_pool = X_test.index[(y_test == 1) & (proba_all > 0.9)][:8].tolist()
    legit_pool = X_test.index[(y_test == 0) & (proba_all < 0.05)][:8].tolist()
    pool = fraud_pool + legit_pool

    if not pool:   # safety net: never crash on an empty pool
        st.sidebar.error("No confident examples found in the test set.")
        st.stop()

    # Select by position, not by label text (labels can collide)
    pos = st.sidebar.selectbox(
        "Transaction",
        range(len(pool)),
        format_func=lambda i: (
            f"{'🚨 FRAUD' if y_test[pool[i]] == 1 else '✅ LEGIT'} · row {pool[i]} · "
            f"${amount_to_dollars(X_test.loc[pool[i], 'Log_Amount']):,.2f} · "
            f"{int(X_test.loc[pool[i], 'Hour'])}:00"
        ),
    )
    row_idx = pool[pos]
    row = X_test.loc[row_idx].copy()
    st.sidebar.caption(f"Ground truth: **{'FRAUD' if y_test[row_idx] == 1 else 'LEGIT'}**")

else:
    # What-if mode: start from a REAL transaction (fraud or legit), change what a human understands
    st.sidebar.caption("Pick a base transaction, then change amount and time.")

    proba_all = get_test_predictions()
    whatif_pool = (
        X_test.index[(y_test == 1) & (proba_all > 0.9)][:20].tolist()
        + X_test.index[(y_test == 0) & (proba_all < 0.05)][:20].tolist()
    )

    if not whatif_pool:
        st.sidebar.error("No suitable base transactions found.")
        st.stop()

    base_pos = st.sidebar.selectbox(
        "Base transaction",
        range(len(whatif_pool)),
        format_func=lambda i: (
            f"{'🚨 FRAUD' if y_test[whatif_pool[i]] == 1 else '✅ LEGIT'} · "
            f"row {whatif_pool[i]} · ${amount_to_dollars(X_test.loc[whatif_pool[i], 'Log_Amount']):,.2f}"
        ),
    )
    row = X_test.loc[whatif_pool[base_pos]].copy()
    base_truth = int(y_test[whatif_pool[base_pos]])
    st.sidebar.caption(f"Base ground truth: **{'FRAUD' if base_truth == 1 else 'LEGIT'}**")

    amount = st.sidebar.number_input("Amount, $", min_value=0.0,
                                     value=float(amount_to_dollars(row['Log_Amount'])), step=25.0)
    hour = st.sidebar.slider("Hour of day", 0, 23, int(row['Hour']))

    row['Log_Amount'] = np.log1p(amount)   # model works with log internally
    row['Hour'] = hour

    with st.sidebar.expander("⚙️ Advanced: PCA features V1–V28"):
        st.caption("Anonymized behavioural signals. Edit only if you know what you are doing.")
        for f in [c for c in FEATURES if c.startswith('V')]:
            # key includes base_pos so widgets RESET when base changes
            row[f] = st.number_input(f, value=float(row[f]), format="%.4f",
                                     key=f"adv_{base_pos}_{f}")


# ==========================================
# EXPLAIN IN PLAIN LANGUAGE
# ==========================================
def explain_in_words(sv_row, row, top_n=4):
    """Turn SHAP values into sentences a non-ML person can read."""
    # base_values can be a scalar (single row) or an array (batch) -> normalize
    base = float(np.atleast_1d(sv_row.base_values).ravel()[0])
    values = np.atleast_2d(sv_row.values)[0]   # ensure shape (n_features,)

    contrib = sorted(
        zip(FEATURES, values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    lines = []
    for feat, shap_val in contrib:
        direction = "⬆️ raised the risk" if shap_val > 0 else "⬇️ lowered the risk"

        if feat == 'Log_Amount':
            detail = f"${amount_to_dollars(row['Log_Amount']):,.2f}"
        elif feat == 'Hour':
            detail = f"{int(row['Hour'])}:00"
        else:
            detail = f"{row[feat]:.2f} std"

        lines.append(f"**{label_for(feat)}** = {detail} → {direction} by `{abs(shap_val):.3f}`")

    return lines, base

# ==========================================
# MAIN: PREDICTION + EXPLANATION
# ==========================================
if st.sidebar.button("🔍 Run fraud check", type="primary"):
    df_row = pd.DataFrame([row])[FEATURES]      # enforce training column order
    proba = float(model.predict_proba(df_row)[0, 1])
    is_fraud = proba >= threshold

    if is_fraud:
        st.error(
            f"🚨 **BLOCK THIS TRANSACTION** — fraud probability **{proba:.2%}** "
            f"(threshold {threshold:.2%})"
        )
    else:
        st.success(
            f"✅ **APPROVE** — fraud probability **{proba:.2%}** "
            f"(threshold {threshold:.2%})"
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"${amount_to_dollars(row['Log_Amount']):,.2f}")
    c2.metric("Time", f"{int(row['Hour'])}:00")
    c3.metric("Fraud probability", f"{proba:.2%}")

    st.subheader("Why did the model decide this?")
    sv = explainer(df_row)

    # Single-row Explanation: use sv[0] for waterfall
    lines, base = explain_in_words(sv[0], row)
    st.markdown(f"Baseline risk for any transaction: **{base:.2%}**")
    for line in lines:
        st.markdown(f"- {line}")
    st.markdown(f"**→ Final risk: {proba:.2%}**")

    with st.expander("📈 Full technical explanation (SHAP waterfall)"):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(plt.gcf())
        plt.close()


# ==========================================
# MODEL METRICS IN BUSINESS LANGUAGE
# ==========================================
with st.expander(f"📊 How good is this model? (test set, {len(X_test):,} transactions)"):
    y_proba = get_test_predictions()            # cached, not recomputed
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    total_fraud = int((y_test == 1).sum())

    # Guard against ZeroDivisionError if the model flags nothing
    precision_str = f"{tp / (tp + fp):.0%}" if (tp + fp) > 0 else "n/a"
    recall_str = f"{tp / total_fraud:.0%}" if total_fraud > 0 else "n/a"

    st.markdown(f"""
**In plain words:**
- Out of **{total_fraud} real frauds**, the model catches **{tp}** and misses **{fn}**.
- Out of **{tp + fp} transactions it flags**, **{tp} are real fraud** and **{fp} are false alarms**.
- That means **{precision_str} of alerts are correct**, and we catch **{recall_str} of all fraud**.

**Why not accuracy?** Only 0.17% of transactions are fraud.
A model that says "everything is fine" would score **99.83% accuracy** and catch **zero** fraud.
That is why we optimize **recall** and use **PR-AUC**, not accuracy.
""")

    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
    col2.metric("PR-AUC", f"{average_precision_score(y_test, y_proba):.4f}")

    st.text(classification_report(y_test, y_pred))