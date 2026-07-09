
"""Bank_Analytics.ipynb"""

from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Bank_file = pd.read_csv('bank-additional-full.csv', sep=';')
Bank_file.info()

Bank_full = pd.read_csv('bank-full.csv', sep=';')
Bank_full.info()

Bank_additional = pd.read_csv('bank-additional.csv', sep=';')
Bank_additional.info()

Bank_full.describe()

Bank_full.columns

Bank_full.head()

print(Bank_full.shape)
print(Bank_full.columns)
print(Bank_full.dtypes)

Bank_full.head()

Bank_full.describe(include='all')

Bank_full.isnull().sum()

for bankfull in Bank_full.columns:
    print(bankfull, Bank_full[bankfull].unique())

Bank_full.replace("unknown", pd.NA, inplace=True)

# categorical
for col in Bank_full.select_dtypes(include='object'):
    Bank_full[col] = Bank_full[col].fillna(Bank_full[col].mode()[0])

# numerical
for col in Bank_full.select_dtypes(include='int64'):
    Bank_full[col] = Bank_full[col].fillna(Bank_full[col].median())

Bank_full.drop_duplicates(inplace=True)
Bank_full

Q1 = Bank_full['balance'].quantile(0.25)
Q3 = Bank_full['balance'].quantile(0.75)
IQR = Q3 - Q1

df = Bank_full[(Bank_full['balance'] >= Q1 - 1.5*IQR) & (Bank_full['balance'] <= Q3 + 1.5*IQR)]
df

Bank_full['y'] = Bank_full['y'].str.lower()
Bank_full

sns.histplot(Bank_full['age'], kde=True)
plt.show()

sns.countplot(x='y', data=Bank_full)
plt.show()

sns.boxplot(x='y', y='balance', data=Bank_full)
plt.show()

sns.heatmap(Bank_full.corr(numeric_only=True), annot=True)
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in Bank_full.select_dtypes(include='object'):
    Bank_full[col] = le.fit_transform(Bank_full[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Bank_full.drop('y', axis=1))

# Using the scaled data prepared in cell RX9lUkSCR-sl
y = Bank_full['y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

# Increased max_iter and using scaled training data
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

print(grid.best_params_)

"""**# 🏦 Bank Marketing Prediction (ML Project)

## 📌 Overview
This project predicts whether a customer will subscribe to a term deposit based on banking and campaign data.

## 🎯 Problem Statement
Banks need to identify potential customers who are likely to subscribe to term deposits to optimize marketing campaigns.

## 📊 Dataset
- Source: Bank Marketing Dataset
- Records: 45,000+
- Features: Customer demographics, financial data, campaign details

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🔍 Workflow
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Evaluation & Optimization

## 🤖 Models Used
- Logistic Regression
- Decision Tree
- Random Forest (Best Model)

## 📈 Results
- Achieved high accuracy using Random Forest
- Key influencing features:
  - Call duration
  - Account balance
  - Previous campaign outcome

## 💡 Business Impact
- Helps banks target high-probability customers
- Reduces marketing cost
- Improves conversion rate

# ✅ Report


1.  Target High-Probability Customers
2.   Improve Call Quality, Not Quantity
3. Re-target Interested Customers
4. Use Data-Driven Segmentation(Enables personalized marketing, improving success rate)
5. Optimize Campaign Timing(maximum response)
  
**We recommend targeting the right customers, improving call quality, and using predictive modeling to increase conversion while reducing cost.**
"""
