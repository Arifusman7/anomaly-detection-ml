# 🛡️ Anomaly Detection in Network Traffic

A machine learning project designed to classify data records as **Normal** or **Anomaly** using data preprocessing, exploratory data analysis, and classification algorithms.

## 📌 Project Overview

This project explores anomaly detection using supervised machine learning. It analyzes the dataset, preprocesses numerical and categorical features, trains multiple classification models, and evaluates their performance to identify anomalous records.

The application provides an interactive dashboard built with Streamlit, allowing users to explore data, compare model performance, and generate predictions from uploaded CSV files.

## 🚀 Features

- Interactive data exploration
- Normal vs Anomaly distribution visualization
- Numerical feature analysis
- Correlation heatmap
- Missing value and duplicate analysis
- Automated numerical and categorical preprocessing
- Machine learning model comparison
- Confusion matrix and classification report
- Batch anomaly predictions
- Downloadable prediction results

## 🤖 Machine Learning Models

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## ⚙️ Data Preprocessing

- Numerical missing values handled using median imputation
- Numerical features standardized using StandardScaler
- Categorical missing values handled using most-frequent imputation
- Categorical features encoded using OneHotEncoder
- Train/test separation supported through the dataset's train column when available

## 📊 Model Evaluation

The models are evaluated using:

- Accuracy
- Anomaly Recall
- Anomaly F1-score
- Confusion Matrix
- Classification Report

## 🌐 Web Application

The Streamlit application allows users to upload a dataset, explore the data, compare models, and generate predictions without directly running notebook cells.

## ▶️ Run Locally

```bash
git clone YOUR_REPOSITORY_URL
cd anomaly-detection-ml
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the entry point.

## 👨‍💻 Project

Machine Learning | Data Analysis | Classification | Anomaly Detection


