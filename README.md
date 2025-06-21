<h1 align="center">
  🥋 UFC Fight Predictor Model
  <img src="img/ufc_logo.png" width="70" style="vertical-align: middle; margin-left: 10px;" />
</h1>

---

## 🎯 Objective

This project aims to build a robust **binary classification model** to predict the winner of a UFC fight. The model estimates whether **Fighter Red** or **Fighter Blue** is more likely to win based on differences in physical attributes, fighting styles, and recent performances.

By transforming fighter-level data into **relative feature vectors**, the model learns from historical outcomes and generalizes effectively to future matchups.

---

## 📊 Dataset Description

The dataset includes detailed information on historical UFC fights. Each row represents a single bout with features combining:

- 🧍‍♂️ **Numerical attributes** (e.g., height, reach, age)
- 🎯 **Categorical encodings** (e.g., stance, fighting style)
- 📈 **Performance indicators** (e.g., striking accuracy, win streaks)

All features are encoded *relatively*:
$$x = fighter_{blue} - fighter_{red}$$

### Key Feature Groups

- **Fighter Attributes**: Height, reach, weight, stance, age  
- **Style & Stance**: One-hot encoded during preprocessing  
- **Performance Metrics**: Strikes per minute, accuracy, takedown success  
- **Recent Form**: Win/loss streaks, time since last fight, fight activity

### 🎯 Target Variable:
- **0** → Fighter Red wins  
- **1** → Fighter Blue wins  

---

## 🛠️ Modeling Approach

The modeling pipeline consists of three stages:

1. **Feature Engineering**
   - Raw fighter stats are converted into relative differences and standardized.
   - Categorical features are one-hot encoded.
   - The final feature vector captures quantitative and qualitative aspects of both fighters.

2. **Model Training**
   - Multiple ML models (both classical and deep learning) are trained on the dataset.
   - The task is framed as a symmetric binary classification problem.

3. **Evaluation**
   - Evaluation metrics include **Accuracy**, **F1-score**, **ROC-AUC**, and **Confusion Matrix**.

---

## 🤖 Models Implemented

The following classifiers have been integrated and tuned:

- ✅ **K-Nearest Neighbors (KNN)**: Classifies by proximity to neighbors in feature space.
- ✅ **Support Vector Machine (SVM)**: Effective in high-dimensional and binary tasks.
- ✅ **Logistic Regression**: Linear classifier with probabilistic output.
- ✅ **Random Forest**: Ensemble of decision trees with high robustness.
- ✅ **Neural Networks (PyTorch)**: Learns non-linear patterns from complex inputs.
- ✅ **AdaBoost**: Combines weak learners in a sequential boosting framework.
- ✅ **Naive Bayes**: Probabilistic model ideal for high-dimensional feature spaces.
- ✅ **Quadratic Discriminant Analysis (QDA)**: Assumes Gaussian class distributions.
- ✅ **Extra Trees**: Randomized ensemble variant of Random Forests.
- ✅ **Gradient Boosting**: Sequential additive model minimizing prediction errors.
- ✅ **XGBoost**: Optimized gradient boosting framework with built-in regularization and parallelism.

---

## 🧪 Project Structure

```bash
ufc-predictor/
├── data/
│   ├── raw/                        # Original fight data
│   └── processed/                  # Cleaned and transformed datasets
├── notebooks/
│   ├── 01-etl.ipynb                  # Data extraction and cleaning
│   ├── 02-eda.ipynb                  # Exploratory Data Analysis
│   ├── 03-feature_engineering.ipynb  # Feature engineering using UFCData
│   ├── 04-training.ipynb             # Model training using training set
│   └── 05-model_experiments.ipynb    # Model comparison and results analysis
├── src/
│   ├── models/
│   │   ├── model_factory.py       # Central model selection logic
│   │   ├── nn_model.py            # PyTorch-based neural network class
│   │   ├── config.py              # Model hyperparameters and registry
│   │   └── ufc_model.py           # Wrapper class for saving, loading, and evaluating models
│   ├── utils/
│   │   ├── helpers.py             # Utility and preprocessing functions
│   │   ├── metrics.py             # Evaluation metrics and plots
│   │   ├── io_models.py           # Save/load model objects from disk
│   │   └── ufc_data.py            # UFCData class: manages data splits and transformations
├── docs/                          # Markdown documentation per model
├── img/                           # Images for plots, logos, and visuals
└── requirements.txt               # Project dependencies

```

---

## 🚀 Getting Started

To run the pipeline locally:

1. **Clone the repository**

```bash
git clone https://github.com/mfourier/ufc-predictor.git
cd ufc-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the notebooks** Start from `notebooks/01-etl.ipynb` and proceed step by step through to `05-model_experiments.ipynb`.

---

## 📚 Documentation

Each model is documented under the `docs/` folder, including:

- Overview and mathematical formulation
- Key assumptions and limitations
- Hyperparameter grids used with `GridSearchCV`
- Integration details with the UFC pipeline

---

## 👥 Contributors

- **Maximiliano Lioi** — M.Sc. in Applied Mathematics @ University of Chile
- **Rocío Yáñez** — M.Sc. in Applied Mathematics @ University of Chile

---

## 🙏 Acknowledgements

We thank [shortlikeafox](https://github.com/shortlikeafox/ultimate_ufc_dataset) for their excellent work compiling the UFC dataset used as the foundation of this project. Their contribution made it possible to train and evaluate predictive models on historical fight outcomes.

---
