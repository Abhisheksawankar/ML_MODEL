# 💧 Water Quality Classification using Random Forest

## 📌 Overview

A machine learning classification project that predicts whether water is **Safe** or **Not Safe** based on chemical and biological water-quality parameters.

The project uses **Random Forest Classifier** to analyze 20 water-quality features and classify the water quality.

## 📊 Dataset

* **Dataset:** Water Quality Dataset
* **Records:** 7,999 initially
* **Features:** 20 water-quality parameters
* **Target:** is_safe

  * `1` → Safe
  * `0` → Not Safe

### Key Features

* Aluminium
* Ammonia
* Arsenic
* Barium
* Cadmium
* Chloramine
* Chromium
* Copper
* Bacteria
* Viruses
* Lead
* Nitrates
* Mercury
* Perchlorate
* Radium
* Selenium
* Silver
* Uranium

## 🔄 Workflow


Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Handle Invalid Values
   ↓
Feature Analysis
   ↓
Handle Class Imbalance
   ↓
Train-Test Split
   ↓
Random Forest Classifier
   ↓
Model Evaluation


## 🧹 Data Cleaning

The dataset contained invalid `#NUM!` values in the `ammonia` and `is_safe` columns.

These invalid records were removed, and the columns were converted into numeric format.


Initial Records: 7,999
After Cleaning: 7,996


## ⚖️ Class Imbalance

The original dataset was imbalanced:


Not Safe: 7,084
Safe:       912


To balance the classes, **RandomOverSampler** was used.

After oversampling:


Not Safe: 7,084
Safe:     7,084


This helped the model learn both classes more effectively.

## 🤖 Model

**Random Forest Classifier** was used because it:

* Handles nonlinear relationships
* Works well with multiple features
* Combines multiple decision trees
* Provides strong classification performance

## 📈 Model Performance

The model achieved approximately:


Accuracy: 99%


### Classification Metrics

* Precision: ~99%
* Recall: ~99%
* F1-Score: ~99%

The model was evaluated using a **Classification Report**.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Random Forest
* Jupyter Notebook

## 🚀 Key Learning

This project demonstrates an end-to-end machine learning classification workflow, including data cleaning, exploratory analysis, class imbalance handling, model training, and evaluation.

## 🔮 Future Improvements

* Use a Pipeline to prevent data leakage
* Apply cross-validation
* Tune Random Forest hyperparameters
* Compare XGBoost and Gradient Boosting
* Add feature importance visualization
* Deploy the model using Streamlit

