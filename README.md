# 🎮 Steam Games Sales Prediction & Classification

This project focuses on analyzing and predicting the commercial performance of video games on the Steam platform. Using a dataset of game metadata, features, and sales estimates, we approach the problem from two angles:

- **Regression**: Predicting the number of copies sold.
- **Classification**: Categorizing games into sales tiers (e.g., Low, Medium, High).

---

## 📁 Table of Contents

- [Dataset](#dataset)
- [Project Goals](#project-goals)
- [Technologies Used](#technologies-used)
- [Regression Pipeline](#regression-pipeline)
- [Classification Pipeline](#classification-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
---

## 📊 Dataset

The dataset was sourced from Steam and Gamalytic, and includes:

- **Basic Info**: `AppID`, name, release date, supported platforms.
- **Game Features**: Steam Achievements, Trading Cards, Workshop support.
- **Metadata**: Price, genres, publisher class, review scores, Metacritic.
- **Target Variables**:
  - `copiesSold` (for regression)
  - `salesClass` (for classification — binned from `copiesSold`)

---

## 🧠 Project Goals

- Build regression models to predict `copiesSold` based on game features.
- Create classification models to categorize games into sales performance groups.
- Analyze feature importance and patterns across successful titles.

---

## 🛠 Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn** – ML models and pipelines
- **XGBoost** – Advanced classification modeling
- **Matplotlib, Seaborn** – Visualization
---

## 🔁 Regression Pipeline

📍 _File: `Regression.ipynb`_

Steps:
1. Load and clean game metadata.
2. Parse list-type columns (e.g., genres).
3. Binarize multi-label features.
4. Remove outliers using IQR/Z-score.
5. Train models:
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Evaluate using RMSE and R².

---

## 🧮 Classification Pipeline

📍 _File: `Classification.ipynb`_

Steps:
1. Prepare dataset as in regression.
2. Convert target `copiesSold` into classes.
3. Train models:
   - Logistic Regression
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Support Vector Classifier (SVC)
   - XGBoost Classifier
4. Evaluate using Accuracy, F1 Score, Confusion Matrix.

---

## 📈 Evaluation Metrics

- **Regression**: 
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Classification**:
  - Accuracy
  - F1 Score
  - Confusion Matrix
  - Classification Report


