import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Anomaly Detection Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Anomaly Detection & Classification")
st.caption("Machine Learning project dashboard • Explore data, compare models, and predict anomaly status")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])
    st.markdown("---")
    st.caption("Expected target column: `anomaly` (0 = Normal, 1 = Anomaly)")
    st.caption("Optional split column: `train` (1 = train, 0 = test)")

if uploaded is None:
    st.info("Upload your dataset CSV in the sidebar to get started.")
    st.markdown("### Dashboard features")
    st.write("• Dataset summary and quality checks")
    st.write("• Anomaly distribution and feature exploration")
    st.write("• Logistic Regression, Random Forest, and Decision Tree comparison")
    st.write("• Confusion matrix and classification report")
    st.write("• Batch predictions and downloadable output")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if "anomaly" not in df.columns:
    st.error("Your uploaded file must contain an 'anomaly' target column for model training.")
    st.stop()

# Dataset overview
st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

tab_eda, tab_models, tab_predict, tab_data = st.tabs(
    ["📊 Exploratory Analysis", "🤖 Model Comparison", "🔍 Predictions", "🗃️ Data Quality"]
)

# Target values, convert safely for charts/models
eda_df = df.copy()
target_counts = eda_df["anomaly"].value_counts(dropna=False)
with tab_eda:
    st.subheader("Normal vs Anomaly")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        target_counts.sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Class (0 = Normal, 1 = Anomaly)")
        ax.set_ylabel("Record count")
        ax.set_title("Target Distribution")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.write("**Class counts**")
        st.dataframe(target_counts.rename("Count").to_frame(), use_container_width=True)

    numeric_all = eda_df.select_dtypes(include=np.number)
    if "anomaly" in numeric_all.columns and numeric_all.shape[1] > 1:
        st.subheader("Correlation Heatmap")
        corr = numeric_all.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature distribution by class")
    numeric_features = [c for c in numeric_all.columns if c != "anomaly"]
    chosen_feature = st.selectbox("Choose a numeric feature", numeric_features) if numeric_features else None
    if chosen_feature:
        fig, ax = plt.subplots()
        sns.boxplot(data=eda_df, x="anomaly", y=chosen_feature, ax=ax)
        ax.set_title(f"{chosen_feature} by anomaly class")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Numeric feature summaries")
    st.dataframe(eda_df.describe(include="all").T, use_container_width=True)

# Prepare feature data (drop identifiers and split/target columns)
drop_cols = [c for c in ["anomaly", "train", "segment"] if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")
y = pd.to_numeric(df["anomaly"], errors="coerce")
valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# Determine train/test based on the notebook's train flag when available
if "train" in df.columns:
    train_mask = pd.to_numeric(df.loc[valid, "train"], errors="coerce").fillna(0).astype(int) == 1
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

# Feature types and preprocessing
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
transformers = []
if num_cols:
    transformers.append(("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols))
if cat_cols:
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols))
preprocessor = ColumnTransformer(transformers, remainder="drop")

model_defs = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
trained_models = {}
metrics_rows = []
for name, model in model_defs.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    trained_models[name] = pipeline
    metrics_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Anomaly Recall": classification_report(y_test, pred, output_dict=True, zero_division=0).get("1", {}).get("recall", 0),
        "Anomaly F1": classification_report(y_test, pred, output_dict=True, zero_division=0).get("1", {}).get("f1-score", 0)
    })
metrics = pd.DataFrame(metrics_rows).sort_values("Accuracy", ascending=False)
best_model_name = metrics.iloc[0]["Model"]

with tab_models:
    st.subheader("Model Performance Comparison")
    m1, m2, m3 = st.columns(3)
    m1.metric("Training records", f"{len(X_train):,}")
    m2.metric("Testing records", f"{len(X_test):,}")
    m3.metric("Best model (accuracy)", best_model_name)
    st.dataframe(metrics.style.format({"Accuracy": "{:.2%}", "Anomaly Recall": "{:.2%}", "Anomaly F1": "{:.2%}"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics.set_index("Model")[["Accuracy", "Anomaly Recall", "Anomaly F1"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model metrics")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    selected_model = st.selectbox("Inspect model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
    inspect_pred = trained_models[selected_model].predict(X_test)
    st.write("**Classification report**")
    report = classification_report(y_test, inspect_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.write("**Confusion matrix**")
    cm = confusion_matrix(y_test, inspect_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Normal", "Anomaly"] if len(cm) == 2 else ax.get_xticklabels())
    ax.set_yticklabels(["Normal", "Anomaly"] if len(cm) == 2 else ax.get_yticklabels())
    st.pyplot(fig)
    plt.close(fig)

with tab_predict:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV containing the feature columns used in the training dataset. The target column `anomaly` is optional.")
    pred_file = st.file_uploader("Upload records for prediction", type=["csv"], key="prediction_csv")
    model_choice = st.selectbox("Prediction model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
    if pred_file:
        try:
            pred_df = pd.read_csv(pred_file)
            required = list(X.columns)
            missing = [col for col in required if col not in pred_df.columns]
            if missing:
                st.error("Missing feature columns: " + ", ".join(missing))
            else:
                # Remove optional label/split columns if present, retain feature columns in expected order
                pred_input = pred_df[required].copy()
                predicted = trained_models[model_choice].predict(pred_input)
                output = pred_df.copy()
                output["Predicted anomaly"] = predicted
                output["Prediction label"] = np.where(predicted == 1, "Anomaly", "Normal")
                st.dataframe(output, use_container_width=True)
                st.download_button("Download predictions CSV", output.to_csv(index=False), "anomaly_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Upload a CSV above to generate predictions.")

with tab_data:
    st.subheader("Data Quality")
    st.write("Missing values per column")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.write("Duplicate records:", int(df.duplicated().sum()))
    st.write("Data types")
    st.dataframe(df.dtypes.astype(str).rename("Data type").to_frame(), use_container_width=True)
    st.write("Raw dataset")
    st.dataframe(df, use_container_width=True)

st.caption("Models are trained in-session from the uploaded dataset. This app is a demonstration and should be validated before operational use.")
