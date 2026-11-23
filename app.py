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
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

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

tab_eda, tab_models, tab_predict, tab_raw, tab_data = st.tabs(
    ["📊 Dashboard / EDA", "🤖 Model Comparison", "✍️ Single Segment & Batch", "📡 Raw Signal", "🗃️ Data Quality"]
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
    st.subheader("Manual Signal Prediction")
    st.write("Enter the feature values for one new signal. The app uses the selected trained model to predict whether it is Normal or Anomaly.")
    model_choice = st.selectbox("Prediction model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name), key="manual_model_choice")

    with st.form("manual_signal_form"):
        st.markdown("#### Signal features")
        manual_values = {}
        feature_cols = list(X.columns)
        feature_layout = st.columns(2)
        for i, col in enumerate(feature_cols):
            with feature_layout[i % 2]:
                if col in num_cols:
                    median_value = pd.to_numeric(X_train[col], errors="coerce").median()
                    default_value = float(median_value) if pd.notna(median_value) else 0.0
                    manual_values[col] = st.number_input(
                        col, value=default_value, key=f"manual_{col}", help=f"Enter a value for {col}"
                    )
                else:
                    options = X_train[col].dropna().astype(str).unique().tolist()
                    options = sorted(options)
                    if len(options) <= 50 and options:
                        manual_values[col] = st.selectbox(col, options, key=f"manual_{col}")
                    else:
                        common_value = X_train[col].dropna().astype(str).mode()
                        default_text = str(common_value.iloc[0]) if not common_value.empty else ""
                        manual_values[col] = st.text_input(col, value=default_text, key=f"manual_{col}")
        submitted_manual = st.form_submit_button("🔍 Predict Signal", use_container_width=True)

    if submitted_manual:
        try:
            manual_input = pd.DataFrame([manual_values], columns=feature_cols)
            manual_prediction = trained_models[model_choice].predict(manual_input)[0]
            if hasattr(trained_models[model_choice], "predict_proba"):
                probabilities = trained_models[model_choice].predict_proba(manual_input)[0]
                classes = trained_models[model_choice].named_steps["model"].classes_
                confidence = float(probabilities[list(classes).index(manual_prediction)])
            else:
                confidence = None
            if int(manual_prediction) == 1:
                st.error("🚨 Prediction: ANOMALY")
            else:
                st.success("✅ Prediction: NORMAL")
            if confidence is not None:
                st.metric("Model confidence for predicted class", f"{confidence:.1%}")
            with st.expander("Review submitted feature values"):
                st.dataframe(manual_input, use_container_width=True)
        except Exception as e:
            st.error(f"Manual prediction error: {e}")

    st.divider()
    st.subheader("Batch Prediction")
    st.write("Upload a CSV containing the feature columns used in the training dataset. The target column `anomaly` is optional.")
    pred_file = st.file_uploader("Upload records for prediction", type=["csv"], key="prediction_csv")
    if pred_file:
        try:
            pred_df = pd.read_csv(pred_file)
            required = list(X.columns)
            missing = [col for col in required if col not in pred_df.columns]
            if missing:
                st.error("Missing feature columns: " + ", ".join(missing))
            else:
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
        st.info("Upload a CSV above to generate batch predictions.")

with tab_raw:
    st.subheader("Predict from a raw signal")
    st.warning(
        "Raw-signal features are calculated in this app using NumPy and SciPy. "
        "Because the original feature-engineering script used to create the training dataset was not available, "
        "peak-based features may differ from the original pipeline. Treat raw-signal predictions as indicative."
    )

    raw_left, raw_right = st.columns(2)
    with raw_left:
        raw_channel = st.selectbox(
            "Channel", sorted(X_train["channel"].dropna().astype(str).unique()) if "channel" in X_train.columns else ["CADC0872"],
            key="raw_channel"
        )
    with raw_right:
        raw_sampling = st.number_input("Sampling interval", min_value=0.000001, value=1.0, step=1.0, key="raw_sampling")

    use_demo = st.checkbox("Use a synthetic demo signal (not real telemetry)", value=True)
    upload_col, paste_col = st.columns(2)
    with upload_col:
        raw_file = st.file_uploader("Upload a CSV with one column of readings", type=["csv"], key="raw_signal_csv")
    with paste_col:
        pasted = st.text_area("Or paste readings (comma, space or newline separated)", placeholder="2.1e-06, 3.4e-06, 1.9e-06, ...", height=100)

    def extract_raw_features(signal, sampling_interval):
        x = np.asarray(signal, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            raise ValueError("Please provide at least 5 valid numeric readings.")
        d1 = np.diff(x)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        smooth10 = pd.Series(x).rolling(10, min_periods=1, center=True).mean().to_numpy()
        smooth20 = pd.Series(x).rolling(20, min_periods=1, center=True).mean().to_numpy()
        p, _ = find_peaks(x)
        p10, _ = find_peaks(smooth10)
        p20, _ = find_peaks(smooth20)
        dp, _ = find_peaks(d1) if len(d1) > 2 else (np.array([]), {})
        d2p, _ = find_peaks(d2) if len(d2) > 2 else (np.array([]), {})
        duration = max((len(x) - 1) * float(sampling_interval), 1.0)
        variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
        features = {
            "sampling": float(sampling_interval), "duration": duration, "len": len(x),
            "mean": float(np.mean(x)), "var": variance,
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "kurtosis": float(kurtosis(x, bias=True)) if np.std(x) > 0 else 0.0,
            "skew": float(skew(x, bias=True)) if np.std(x) > 0 else 0.0,
            "n_peaks": int(len(p)), "smooth10_n_peaks": int(len(p10)),
            "smooth20_n_peaks": int(len(p20)), "diff_peaks": int(len(dp)),
            "diff2_peaks": int(len(d2p)),
            "diff_var": float(np.var(d1, ddof=1)) if len(d1) > 1 else 0.0,
            "diff2_var": float(np.var(d2, ddof=1)) if len(d2) > 1 else 0.0,
            "gaps_squared": float(duration), "len_weighted": float(len(x)),
            "var_div_duration": float(variance / duration),
            "var_div_len": float(variance / max(len(x), 1))
        }
        return features, x

    signal = None
    if raw_file is not None:
        try:
            raw_df = pd.read_csv(raw_file)
            numeric_candidates = raw_df.select_dtypes(include=np.number)
            if numeric_candidates.empty:
                st.error("The uploaded CSV must contain at least one numeric readings column.")
            else:
                selected_reading_col = st.selectbox("Select readings column", numeric_candidates.columns.tolist())
                signal = numeric_candidates[selected_reading_col].dropna().to_numpy(dtype=float)
        except Exception as e:
            st.error(f"Could not read signal CSV: {e}")
    elif pasted.strip():
        try:
            signal = np.fromstring(pasted.replace(",", " ").replace(";", " "), sep=" ")
        except Exception:
            st.error("Could not parse the pasted readings. Use commas or spaces between numbers.")
    elif use_demo:
        t = np.linspace(0, 2 * np.pi, 200)
        signal = 2.0e-5 * np.sin(t) + 2.0e-6 * np.sin(5 * t)

    if signal is not None and len(signal) > 0:
        try:
            raw_features, clean_signal = extract_raw_features(signal, raw_sampling)
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(np.arange(len(clean_signal)) * raw_sampling, clean_signal, linewidth=1.2)
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal amplitude")
            ax.set_title("Signal preview")
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if st.button("🔍 Predict raw signal", type="primary", key="predict_raw_signal"):
                raw_row = {}
                for col in X.columns:
                    if col == "channel":
                        raw_row[col] = raw_channel
                    elif col in raw_features:
                        raw_row[col] = raw_features[col]
                    else:
                        # Preserve compatibility with any additional training feature.
                        med = pd.to_numeric(X_train[col], errors="coerce").median() if col in X_train.columns else 0
                        raw_row[col] = float(med) if pd.notna(med) else 0.0
                raw_input = pd.DataFrame([raw_row], columns=X.columns)
                raw_pred = trained_models[best_model_name].predict(raw_input)[0]
                probs = trained_models[best_model_name].predict_proba(raw_input)[0] if hasattr(trained_models[best_model_name], "predict_proba") else None
                anomaly_prob = float(probs[list(trained_models[best_model_name].named_steps["model"].classes_).index(1)]) if probs is not None and 1 in trained_models[best_model_name].named_steps["model"].classes_ else None
                st.divider()
                if int(raw_pred) == 1:
                    st.error("🚨 Satellite Anomaly Detected")
                else:
                    st.success("✅ Signal classified as Normal")
                if anomaly_prob is not None:
                    k1, k2 = st.columns(2)
                    k1.metric("Anomaly probability", f"{anomaly_prob:.1%}")
                    k2.metric("Normal probability", f"{1-anomaly_prob:.1%}")
                    st.progress(min(max(anomaly_prob, 0.0), 1.0), text=f"Anomaly risk: {anomaly_prob:.1%}")
                with st.expander("Extracted features"):
                    st.dataframe(pd.DataFrame([raw_features]).T.rename(columns={0: "Value"}), use_container_width=True)
        except Exception as e:
            st.error(f"Raw signal processing error: {e}")
    else:
        st.info("Upload a one-column signal CSV, paste readings, or select the synthetic demo signal.")

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
