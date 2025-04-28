# 🥋 UFC Fight Predictor Model

<p align="center">
  <img src="img/ufc_logo.png" width="400">
</p>

## 🎯 Objective

This project aims to build a **binary classification model** that predicts the winner of a UFC fight. The model evaluates differences in physical attributes, fighting styles, and recent performances to estimate whether **Fighter Red** or **Fighter Blue** is more likely to win.

By transforming raw fighter data into **relative feature vectors**, the model learns from historical outcomes and generalizes to future matchups.

---

## 📊 Dataset Description

The dataset contains detailed information for each UFC fight. Each sample represents a single bout, with features that combine:

- **Numerical Features** between fighters (e.g., height, reach, weight).
- **Categorical encodings** of fighter characteristics (e.g., stance, fighting style).
- **Recent performance indicators** (e.g., win streaks, ring rust).

While numerical features are often expressed as fighter-to-fighter differences:
$$x = fighter_{blue} - fighter_{red}$$

### 🧠 Key Features

- **Fighter Attributes**: Height, reach, weight, stance, age  
- **Fighting Style & Stance**: One-hot encoded as part of preprocessing  
- **Performance Metrics**: Striking accuracy, strikes per minute, takedown accuracy  
- **Win History**: Current win/loss streaks, decision types (e.g., split/majority/unanimous)  
- **Activity Level**: Recent fight frequency, layoff duration

### Target Variable:
- **0**: Fighter Red wins
- **1**: Fighter Blue wins

The target variable represents the winner of the fight, where **0** indicates that Fighter Red wins, and **1** indicates that Fighter Blue wins.

## 🛠️ Approach

The modeling pipeline follows these core steps:

1. **Feature Engineering**  
   - Numerical features (e.g., height, reach) are first transformed into relative differences between fighters, and then standardized to have zero mean and unit variance for consistent model input.
   - Categorical features (e.g., stance, style) are **encoded and combined** to represent fighter characteristics  
   - Final feature vectors capture both **quantitative and qualitative** aspects of the fighters

2. **Model Training**  
   - Multiple machine learning models are trained on the engineered dataset, including both traditional models and deep learning approaches.
   - The task is framed as a binary classification problem with balanced classes, since the matchups between Fighter Red and Fighter Blue are treated symmetrically—making Fighter Red vs. Fighter Blue and Fighter Blue vs. Fighter Red equivalent scenarios.

3. **Evaluation**  
   - Model performance is assessed using **accuracy**, **F1-score**, **ROC-AUC**, and **confusion matrices**

---

## Models Implemented
The following machine learning models have been implemented to predict UFC fight outcomes:

- ✅**K-Nearest Neighbors (KNN)**: A non-parametric method used for classification based on the proximity of data points.
- ✅**Support Vector Machine (SVM)**: A supervised learning model that works well in high-dimensional spaces and is effective for binary classification.
- ✅**Logistic Regression**: A linear model used for binary classification, commonly used for probabilistic predictions.
- ✅**Random Forest**: An ensemble learning method based on decision trees that improves model accuracy by combining multiple trees.
- ✅**Neural Networks (using PyTorch)**: A deep learning approach that can learn complex patterns in large datasets.
- ✅**AdaBoost**: An ensemble technique that combines weak classifiers to create a strong classifier.
- ✅**Naive Bayes**: A probabilistic classifier based on Bayes' theorem, useful for large feature sets.

## 🧪 Project Structure
```bash
ufc-predictor/
├── data/
│   ├── raw/                       # Original fight data
│   └── processed/                 # Cleaned & transformed datasets
├── notebooks/
│   ├── etl.ipynb                  # Data extraction and cleaning
│   ├── feature_engineering.ipynb  # Feature Engineering
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── training.ipynb             # Model training
│   └── model_experiments.ipynb    # Models metrics comparisons, results analysis and experimentation
├── src/
│   ├── models/
│   │   ├── model_factory.py       # Central model selection and training
│   │   ├── nn_model.py            # PyTorch neural network implementation
│   │   └── config.py              # Model-related configuration settings
│   ├── utils/
│   │   ├── helpers.py             # Data preparation and utility functions
│   │   ├── metrics.py             # Evaluation and plotting metrics functions
│   │   └── io_models.py           # Saving/loading models to/from disk
├── docs/                          # Model documentation in Markdown format
```

## 🚀 Installation & Usage
To run the model, follow these steps:

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/mfourier/ufc-predictor.git

2. **Install the required dependencies**:  
   ```bash
   pip install -r requirements.txt

3. **Run the notebooks**
Start with notebooks/etl.ipynb and proceed through the pipeline.

## 👥 Contributors:
* Maximiliano Lioi, **M.Sc.** in Applied Mathematics @ University of Chile, Departament of Mathematical Engineering  
* Rocío Yañez, **M.Sc.** in Applied Mathematics @ University of Chile, Department of Mathematical Engineering

## 🙏 Acknowledgements
We gratefully acknowledge the work of **shortlikeafox**, whose repository can be found at https://github.com/shortlikeafox/ultimate_ufc_dataset. This dataset contains valuable historical data on UFC fights, which is crucial for training and evaluating the predictive models in this project.


