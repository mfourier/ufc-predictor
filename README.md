# UFC Fight Predictor Model

## Objective
The objective of this project is to build a binary classification model to predict the winner of a UFC fight. The model utilizes key features of both fighters to predict whether Fighter Red or Fighter Blue will emerge victorious. The prediction is made by analyzing various factors, including age, height, reach, weight, recent performance streaks, and more.

## Dataset Description
The dataset contains detailed fight data where each entry represents a UFC fight, with features representing the differences between the two fighters. For each fight, the feature vector is constructed as the difference in attributes between Fighter Red and Fighter Blue:

$$x = \text{features-fighter-red} - \text{features-fighter-blue}$$

### Key Features:
- **Fighter Attributes**: Stance, Height, Reach, Weight, Current Win/Loss Streaks, etc.
- **Striking Accuracies**: Average strikes landed per minute, etc.
- **Win by Decision Splits**: Breakdown of decision outcomes in previous fights.
- **Fight History**: Previous win/loss streaks and performance consistency.

### Target Variable:
- **0**: Fighter Red wins
- **1**: Fighter Blue wins

The target variable represents the winner of the fight, where **0** indicates that Fighter Red wins, and **1** indicates that Fighter Blue wins.

## Approach
The model aims to predict the outcome of UFC fights by analyzing the feature differences between two fighters. Each fight's feature vector $ x $ is the difference between the attributes of Fighter Red and Fighter Blue, capturing the relative strengths and weaknesses of the fighters.

The machine learning models trained on this feature difference aim to predict whether Fighter Red or Fighter Blue is more likely to win based on historical data and their performance characteristics.

### Feature Engineering:
- **Feature Vector Construction**: The feature vector for each fight is represented as:
  $$x = \text{features-fighter-red} - \text{features-fighter-blue}$$
  This transformation highlights the relative advantages or disadvantages of each fighter in various aspects, such as physical attributes, fighting style, and recent performance trends.

### Classification Task:
- The task is framed as a binary classification problem, where the model predicts either **0** (Fighter Red wins) or **1** (Fighter Blue wins).
- Various machine learning algorithms are used to train models on this data, including both traditional models and deep learning approaches.

## Models Implemented
The following machine learning models have been implemented to predict UFC fight outcomes:

- **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification based on the proximity of data points.
- **Support Vector Machine (SVM)**: A supervised learning model that works well in high-dimensional spaces and is effective for binary classification.
- **Logistic Regression**: A linear model used for binary classification, commonly used for probabilistic predictions.
- **Random Forest**: An ensemble learning method based on decision trees that improves model accuracy by combining multiple trees.
- **Neural Networks (using PyTorch)**: A deep learning approach that can learn complex patterns in large datasets.
- **AdaBoost**: An ensemble technique that combines weak classifiers to create a strong classifier.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, useful for large feature sets.

## Hyperparameter Tuning
To optimize model performance, **GridSearchCV** is used for hyperparameter tuning. This method exhaustively searches through a specified parameter grid to find the optimal set of hyperparameters for each model. Some of the hyperparameters tuned include:

- **K-Nearest Neighbors**: `n_neighbors`, `weights`, `metric`
- **SVM**: `C`, `kernel`, `gamma`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **Neural Networks**: Number of hidden units, learning rate, batch size, epochs

GridSearchCV is implemented with **5-fold cross-validation** to ensure robust performance evaluation across different data splits.

## Installation & Usage
To run the model, follow these steps:

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/mfourier/ufc-predictor.git

## Installation & Usage
To run the model, follow these steps:

1. Clone the repository:  
   `git clone https://github.com/mfourier/ufc-predictor.git`

2. Install the required dependencies:  
   `pip install -r requirements.txt`

3. Run the model or execute the provided Jupyter notebooks to start making predictions.

## Contributors:
* Maximiliano Lioi, **M.Sc.** in Applied Mathematics @ University of Chile  
* Rocío Yañez, **M.Sc.** Candidate in Applied Mathematics @ University of Chile

## Acknowledgements
We thank the dataset provided by the user **shortlikeafox**, whose repository can be found at https://github.com/shortlikeafox/ultimate_ufc_dataset. This dataset contains valuable historical data on UFC fights, which is crucial for training and evaluating the predictive models in this project.


