# 🧠 Loan Default Prediction using Bayesian Network

## 📌 Overview

A probabilistic machine learning project that predicts the probability of **Loan Default** based on a customer's **Credit Score** and **Payment Delay** behavior.

The project uses a **Discrete Bayesian Network** implemented with `pgmpy`.

## 📊 Dataset

A small categorical dataset containing:

* `CreditScore` → High, Medium, Low
* `PaymentDelay` → Yes or No
* `LoanDefault` → Yes or No

## 🔄 Workflow

id="a1d0b3"
Dataset
   ↓
Define Bayesian Network
   ↓
Set Variable Relationships
   ↓
Learn Probabilities from Data
   ↓
Variable Elimination
   ↓
Probability-Based Prediction


## 🔗 Bayesian Network Structure
id="zj50hy"
CreditScore ───────┐
                   ↓
              LoanDefault
                   ↑
PaymentDelay ──────┘


The model assumes that both **Credit Score** and **Payment Delay** influence the probability of a loan default.

## ⚙️ Model Training

The model learns conditional probabilities from the dataset using:


MaximumLikelihoodEstimator


This allows the Bayesian Network to estimate the probability of each possible outcome based on observed data.

## 🔍 Inference

For the following evidence:

id="0x4n3j"
CreditScore = Low
PaymentDelay = Yes


The model predicted:

 id="1m6t0q"
LoanDefault = Yes → 100%
LoanDefault = No  → 0%


This means that, based on the available training data, the combination of a **Low Credit Score** and **Payment Delay** strongly indicates a high probability of loan default.

## 🛠️ Technologies

* Python
* Pandas
* pgmpy
* Discrete Bayesian Network
* Maximum Likelihood Estimation
* Variable Elimination
* Google Colab

## 🚀 Key Learning

This project demonstrates how Bayesian Networks can model relationships between categorical variables and perform probabilistic predictions using evidence.

Unlike traditional classification models, Bayesian Networks provide a probability distribution for possible outcomes rather than only a single prediction.

## 🔮 Future Improvements

* Use a larger real-world loan dataset
* Add more variables such as income and debt-to-income ratio
* Learn network structure automatically
* Compare Bayesian Network results with Logistic Regression
* Build an interactive prediction application using Streamlit


