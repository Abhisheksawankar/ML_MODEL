# 🏦 Loan Approval Prediction using Logistic Regression

## 📌 Overview

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** using **Logistic Regression**.

The model analyzes applicant information such as income, credit history, loan amount, education, employment status, and property area.

## 🔄 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Imputation
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Interactive Prediction
```

## 🛠️ Technologies

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* Streamlit
* Kaggle Dataset

## 🧹 Data Preprocessing

* Removed `Loan_ID` because it is only an identifier.
* Handled missing values using median and mode imputation.
* Encoded categorical variables.
* Applied feature scaling to improve Logistic Regression convergence.

### Initial Issue

The initial model generated a convergence warning because the features had different numerical scales.

### Improvement

Feature scaling using `StandardScaler` was applied to improve model optimization and convergence.

## 📊 Model Performance

The initial Logistic Regression model achieved approximately **79% accuracy**.

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Because the dataset contains more approved than rejected loans, metrics beyond accuracy were also considered.

## 🖥️ Interactive Prediction

Users can enter applicant details such as:

* Income
* Credit History
* Loan Amount
* Loan Term
* Education
* Employment Status
* Property Area

The application provides an instant prediction:

```text
✅ Loan Approved
❌ Loan Rejected
```

It can also display the predicted approval probability.

## 🚀 Future Improvements

* Use `Pipeline` and `ColumnTransformer`
* Replace `LabelEncoder` with `OneHotEncoder`
* Compare Logistic Regression with Random Forest and XGBoost
* Add cross-validation and hyperparameter tuning
* Add model explainability using SHAP

## ⚠️ Disclaimer

This project is for educational purposes and should not be used as the sole basis for real-world financial decisions.
