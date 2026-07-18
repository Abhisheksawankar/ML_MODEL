# 🧬 Breast Cancer Classification with AdaBoost

-> A machine learning project that classifies breast tumors as **Benign** or **Malignant** using the AdaBoost algorithm.

---

## 📊 Model Performance

| Metric              |      Score |
| ------------------- | ---------: |
| 🎯 Accuracy         | **97.37%** |
| 🟢 Benign Recall    |    **99%** |
| 🔴 Malignant Recall |    **95%** |
| ⭐ Weighted F1-Score |    **97%** |

---

## 🧠 What This Project Does

The model analyzes **30 tumor-related numerical features** and predicts:

B → Benign
M → Malignant


The project also identifies the **Top 10 most important features** used by the model for classification.

---

## 🔄 Machine Learning Pipeline

📥 Dataset
    ↓
🧹 Data Preprocessing
    ↓
🎯 Feature Selection
    ↓
✂️ Train-Test Split
    ↓
🤖 AdaBoost Training
    ↓
🔮 Prediction
    ↓
📈 Model Evaluation
    ↓
🔍 Feature Importance


---

## 🤖 Why AdaBoost?

**AdaBoost (Adaptive Boosting)** builds a strong classifier by combining multiple weak learners.

Each new learner focuses on samples that were incorrectly classified by previous learners, improving the overall performance step by step.

---

## 📊 Dataset

| Property        | Details         |
| --------------- | --------------- |
| Dataset Size    | **569 records** |
| Input Features  | **30**          |
| Target Variable | `diagnosis`     |
| Benign          | `B`             |
| Malignant       | `M`             |

### Key Features

`radius` • `texture` • `perimeter` • `area` • `smoothness` • `compactness` • `concavity` • `symmetry`

---

## 🧪 Evaluation

The model was evaluated using:

* ✅ Accuracy Score
* 📋 Classification Report
* 🔲 Confusion Matrix
* 🔍 Feature Importance

### Confusion Matrix

```text
[[70   1]
 [ 2  41]]
```

The model correctly classified the majority of both benign and malignant cases.

---

## 🛠️ Tech Stack


🐍 Python
📊 Pandas
🔢 NumPy
📈 Matplotlib
🤖 Scikit-learn
🧠 AdaBoost
📓 Jupyter Notebook / Google Colab


---

## 📁 Project Structure


ML_MODEL/
│
├── Adaboost_algo.ipynb
└── README.md


---

## 🚀 Future Improvements

* 🔧 Hyperparameter tuning
* 🔁 Cross-validation
* 📊 Model comparison with Random Forest and XGBoost
* 🔍 Advanced model explainability
* 🌐 Interactive Streamlit application



