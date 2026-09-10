<div align="center">

# 🩺 Diabetes Risk Classification

**A machine learning pipeline that predicts whether a patient is likely diabetic based on diagnostic measurements.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/Purpose-Educational-blue.svg)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models](#-models)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Interactive Interface](#-interactive-interface)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

This project predicts whether a patient is **diabetic** or **non-diabetic** using 8 diagnostic measurements (glucose level, blood pressure, BMI, age, etc.). The target `Outcome` is binary:

| Value | Class |
|---|---|
| `0` | Non-Diabetic |
| `1` | Diabetic |

Four classification models were trained and compared, combined into a soft-voting ensemble, and the best-performing model was selected on held-out test accuracy.

---

## 🗂 Dataset

- **Source:** `pima-indians-diabetes.csv` — Pima Indians Diabetes Database
- **Size:** 768 rows × 9 columns
- **Duplicates:** none found
- **Missing values (NaN):** none — but several columns contain **medically invalid zeros** (see below)

**Features (8):**
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

**Target:** `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

### Invalid zero values (before imputation)

A value of `0` is not physiologically possible for several of these columns, so it was treated as a missing-value marker rather than a true reading:

| Column | Zero Count | Zero % |
|---|---|---|
| `Insulin` | 374 | 48.70% |
| `SkinThickness` | 227 | 29.56% |
| `Pregnancies`* | 111 | 14.45% |
| `BloodPressure` | 35 | 4.56% |
| `BMI` | 11 | 1.43% |
| `Glucose` | 5 | 0.65% |

*`Pregnancies = 0` is a valid value, not a data quality issue, so it was left untouched.

Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were replaced with the **median of the non-zero training values** (fit on train, applied to both train and test — no leakage).

### Class distribution (target imbalance)

| Class | Count | % |
|---|---|---|
| Non-Diabetic | 500 | 65.10% |
| Diabetic | 268 | 34.90% |

The dataset is moderately imbalanced toward `Non-Diabetic`, which affects recall on the diabetic class (see [Results](#-results)).

---

## ⚙️ Project Workflow

1. **Data Loading** — Load the raw CSV with Pandas.
2. **Data Inspection** — Shape, dtypes, missing values, duplicates.
3. **Data Quality Audit** — Identify medically invalid zeros across features.
4. **Train/Test Split** — Stratified 80/20 split (614 train / 154 test) — done *before* imputation to avoid leakage.
5. **Zero Imputation** — Median imputation (fit on train, applied to train + test) for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
6. **Skewness Check** — Compared feature skew before/after cleaning.
7. **Outlier Handling** — IQR-based capping (1.5×IQR bounds), fit on train, applied to both sets.
8. **EDA**
   - Class distribution breakdown.
   - Correlation of each feature with `Outcome`.
9. **Scaling** — `StandardScaler`, fit on train only.
10. **Modeling** — Train 4 classifiers + 1 soft-voting ensemble (see below).
11. **Evaluation** — Accuracy, precision, recall, F1-score, classification report per model.
12. **Final Model Selection** — Best model chosen by test accuracy.

---

## 🧠 Models

| Model | Key Settings |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Random Forest | `n_estimators=200`, `max_depth=6`, `min_samples_leaf=5` |
| SVM | `kernel='linear'`, `C=1`, `probability=True` |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.05`, `max_depth=5` |
| Voting Classifier (soft) | Combines all 4 models above |

**Best model: Gradient Boosting** — highest test accuracy among the models compared.

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [`pandas`](https://pandas.pydata.org/) | Data loading, cleaning, and manipulation |
| [`numpy`](https://numpy.org/) | Numerical operations |
| [`matplotlib`](https://matplotlib.org/) | Plotting |
| [`seaborn`](https://seaborn.pydata.org/) | Statistical visualizations |
| [`scikit-learn`](https://scikit-learn.org/) | Core ML library — preprocessing, models, metrics |
| &nbsp;&nbsp;↳ `train_test_split` | Stratified train/test splitting |
| &nbsp;&nbsp;↳ `StandardScaler` | Feature scaling |
| &nbsp;&nbsp;↳ `LogisticRegression`, `RandomForestClassifier`, `SVC`, `GradientBoostingClassifier`, `VotingClassifier` | Classification models |
| &nbsp;&nbsp;↳ `classification_report` | Evaluation metrics |
| [`Streamlit`](https://streamlit.io/) | Interactive web app for live predictions |

---

## 📁 Project Structure

```
diabetes-risk-classification/
│
├── diabetes_classification.ipynb   # Main notebook (full pipeline)
├── data/
│   └── pima-indians-diabetes.csv   # Raw dataset
├── models/
│   ├── best_model.pkl              # Saved best model
│   └── scaler.pkl                  # Saved fitted scaler
├── visualization/                  # Saved EDA/evaluation plots
├── app.py                          # Streamlit app
├── requirements.txt                # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/diabetes-risk-classification.git
cd diabetes-risk-classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `pima-indians-diabetes.csv` inside a `data/` folder, and update the loading path in the notebook:
```python
df = pd.read_csv("data/pima-indians-diabetes.csv")
```

### 5. Run the notebook
```bash
jupyter notebook diabetes_classification.ipynb
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
jupyter
```

---

## 📊 Results

All models were evaluated on the same 154-row held-out test set.

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.788 | 0.714 |
| SVM (linear) | 0.788 | 0.701 |
| Random Forest | 0.849 | 0.740 |
| Voting Classifier | 0.888 | 0.734 |
| **Gradient Boosting** | **0.998** | **0.747** |

### Final classification report (Gradient Boosting, test set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Non-Diabetic (0) | 0.79 | 0.83 | 0.81 | 100 |
| Diabetic (1) | 0.65 | 0.59 | 0.62 | 54 |
| **Accuracy** | | | **0.75** | 154 |

Gradient Boosting delivers the highest overall accuracy of the models compared, with solid precision and recall on the majority Non-Diabetic class.

---

## 🖥 Interactive Interface

A Streamlit app (`app.py`) provides a form-based UI for the 8 diagnostic values:

- Two-column input form (`Pregnancies`, `Glucose`, `Blood Pressure`, `Skin Thickness`, `Insulin`, `BMI`, `Diabetes Pedigree`, `Age`), each with a help tooltip explaining the metric.
- Inputs are batched behind a `Run Prediction` submit button (via `st.form`) instead of triggering a rerun on every keystroke.
- On submit, a spinner shows while the prediction runs, then the result is shown as a styled "High Risk" / "Low Risk" banner with a recommendation to consult a healthcare professional on a positive result.

---

## 👥 Team

- **Anas Elhalwagy** — [GitHub](https://github.com/<username>)
- **Walaa Atef** — [GitHub](https://github.com/<username>)
- **Ali Elhassan** — [GitHub](https://github.com/<username>)
- **Kareem Mohamed** — [GitHub](https://github.com/<username>)
- **Jana Mohamed** — [GitHub](https://github.com/<username>)

---

## 📄 License

This project is for educational purposes only and is not intended for real medical diagnosis or clinical use.
