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
- 🎯 **Categorical encodings** (fighting style: ortodox, southpaw, switch, fight stance: open, closed, weight classes.)
- 📈 **Performance indicators** (e.g., striking landed per minute, average takedown attempts)

All features are encoded *relatively*:
$$x = fighter_{blue} - fighter_{red}$$

### Key Feature Groups

- **Fighter Attributes**: Height, reach, weight class, stance, age.  
- **Style & Stance**: One-hot encoded during preprocessing.  
- **Performance Metrics**: Strikes per minute, accuracy, takedown attempts.  
- **Recent Form**: Win/loss streaks, odds. 

### 🎯 Target Variable:
- **0** → Fighter Red wins  
- **1** → Fighter Blue wins  

---

## 🛠️ Modeling Approach

The modeling pipeline is organized into three interconnected stages:

1. **Feature Engineering**
   - Fighter data is transformed into **relative differences** between Blue and Red fighters, covering height, reach, age, striking stats, grappling stats, and win streaks.
   - Categorical variables (e.g., stance, fighting style, weight class) are one-hot encoded, using binary encoding for two-class categories and full dummies for multiclass features.
   - Numerical features are standardized using scalers fitted exclusively on the training set, ensuring no data leakage.
   - Additional features are engineered to capture recent activity, such as experience-per-age difference (total rounds fought divided by age), win-by-decision rate difference, and win-by-finish rate difference.
   - Feature selection is guided by correlation analysis, aiming to minimize inter-feature correlation while preserving predictive signal.
   - A synthetic random noise feature (`Random_Noise`) is introduced as a baseline for feature importance: different combinations were explored until the random column gained prominence, guiding the final selection. This iterative approach led to a feature set that maximizes predictive power without overfitting.

2. **Model Training**
   - A diverse suite of machine learning models is trained, combining **classical algorithms**, **boosted ensemble methods**, and **deep learning architectures**.
   - The task is framed as a binary classification problem, with a baseline distribution of approximately 58% red corner wins, reflecting historical outcome imbalance.
   - Hyperparameter tuning is conducted in the notebook `04-training.ipynb`, where detailed parameter grids are defined for each model using `GridSearchCV`. This systematic exploration includes models such as XGBoost, SVM, Random Forest, AdaBoost, and Neural Networks, optimizing performance across algorithmic families.

3. **Evaluation**
   - Model evaluation leverages a comprehensive set of metrics, computed via the modular `metrics.py` implementation:
     - **Accuracy** (0–1, higher is better): Overall proportion of correct predictions.
     - **Precision** (0–1, higher is better): Fraction of positive predictions that are actually correct.
     - **Recall** (0–1, higher is better): Fraction of true positives correctly identified.
     - **F1 Score** (0–1, higher is better): Harmonic mean of precision and recall.
     - **ROC-AUC** (0.5–1, higher is better): Probability the model ranks a random positive higher than a random negative.
     - **Brier Score** (0–1, lower is better): Mean squared error between predicted probabilities and actual outcomes, reflecting calibration.
   - Confusion matrices visualize classification performance across true/false positives and negatives.
   - The framework supports automated multi-model comparison, identifying the top-performing model per metric, enabling robust benchmarking.

---

## 🤖 Models Implemented

The following classifiers have been integrated and carefully tuned, all coordinated through the modular `model_factory.py` pipeline, enabling systematic benchmarking and performance optimization:

- 🔹 **Classical Models**
  - ✅ **K-Nearest Neighbors (KNN)**: Classifies based on proximity to neighboring points in feature space.
  - ✅ **Support Vector Machine (SVM)**: Effective in high-dimensional, binary classification tasks.
  - ✅ **Logistic Regression**: Linear classifier with probabilistic outputs.
  - ✅ **Naive Bayes**: Probabilistic model suited for high-dimensional feature spaces.
  - ✅ **Quadratic Discriminant Analysis (QDA)**: Assumes Gaussian class-conditional distributions.

- 🔹 **Ensemble Methods**
  - ✅ **Random Forest**: Bagging ensemble of decision trees, providing robustness and low variance.
  - ✅ **Extra Trees**: Randomized ensemble variant of Random Forest, enhancing variance reduction.

- 🔹 **Boosted Ensemble Models**
  - ✅ **AdaBoost**: Sequentially combines weak learners to focus on difficult samples.
  - ✅ **Gradient Boosting**: Iteratively builds additive models to minimize prediction error.
  - ✅ **XGBoost**: Highly optimized gradient boosting with regularization, parallelism, and advanced hyperparameter tuning.

- 🔹 **Deep Learning**
  - ✅ **Neural Networks (MLP)**: Multi-layer perceptron capable of capturing complex, non-linear relationships.

---

## 🧪 Project Structure

```bash
ufc-predictor/
├── data/
│   ├── raw/                         # Original fight data
│   ├── processed/                   # Cleaned and transformed datasets
│   └── results/                     # Evaluation logs, metrics, model reports
├── notebooks/
│   ├── 01-etl.ipynb                 # Data extraction and cleaning
│   ├── 02-eda.ipynb                 # Exploratory Data Analysis
│   ├── 03-feature_engineering.ipynb # Feature engineering using UFCData
│   ├── 04-training.ipynb            # Model training using training set
│   └── 05-model_experiments.ipynb   # Model comparison and results analysis
├── src/
│   ├── config.py                    # Model hyperparameters and registry
│   ├── data.py                      # UFCData class: manages data splits and transformations
│   ├── helpers.py                   # Utility and preprocessing functions
│   ├── io_model.py                  # Save/load model objects from disk
│   ├── metrics.py                   # Evaluation metrics and plots
│   ├── model.py                     # Wrapper class for saving, loading, and evaluating models
│   ├── model_factory.py             # Central model selection logic
├── docs/                            # Markdown documentation per model
├── img/                             # Images for plots, logos, and visuals
└── requirements.txt                 # Project dependencies


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

Comprehensive project documentation is available in the `docs/` folder, covering:

- **Model overviews and mathematical formulations**: Detailed descriptions of each algorithm, including underlying principles and expected behavior.
- **Key assumptions and limitations**: Insights into when and why each model performs best, as well as potential pitfalls.
- **Hyperparameter grids**: Full parameter configurations used for tuning with `GridSearchCV`, enabling reproducibility and extension.
- **Usage guides**: Step-by-step instructions on running the notebooks, customizing experiments, and interpreting results.

---

## 👥 Contributors

- **Maximiliano Lioi** — M.Sc. in Applied Mathematics @ University of Chile
- **Rocío Yáñez** — M.Sc. in Applied Mathematics @ University of Chile

---

## 🙏 Acknowledgements

We thank [shortlikeafox](https://github.com/shortlikeafox/ultimate_ufc_dataset) for their excellent work compiling the UFC dataset used as the foundation of this project. Their contribution made it possible to train and evaluate predictive models on historical fight outcomes.

---
