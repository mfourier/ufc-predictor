# UFC Fight Predictor Model

## Objective
The goal of this project is to predict the winner of a UFC fight using a binary classification approach. By analyzing key features of both fighters, the model determines whether Fighter A or Fighter B will emerge victorious. The prediction is based on various factors, including but not limited to age, height, reach, and recent performance streaks.

## Dataset Description
The dataset contains detailed fight data, where each fight is represented by the difference in features between two fighters. Each fight, denoted as **x**, is the feature difference between Fighter A and Fighter B. 

Key features include:
- **Stance, Height, Reach, Weight, Current Lose Streak, Current Win Streak, etc.**
- **Striking Accuracies** (average strikes landed per minute, etc.)
- **Win by Decision Splits**
- and more...

For each fight, the feature vector is modeled as:
$$x = \text{features-fighter-red} - \text{features-fighter-blue}$$

Where $$x \in \mathbb{R}^n$$ represents the vector containing the differences in the respective features between Fighter A and Fighter B.

The target variable is a binary classification indicating the winner:
- **0**: Fighter Red wins
- **1**: Fighter Blue wins

## Approach
The model predicts the outcome of a UFC fight by analyzing the difference in features between the two fighters. For each fight, the feature vector $$x$$ is calculated as:
$$x = \text{features-fighter-red} - \text{features-fighter-blue}$$

The goal is to build a machine learning model capable of accurately predicting the fight winner based on these feature differences.

## Models Implemented
The following machine learning models have been implemented to predict fight outcomes:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **Neural Networks** (using PyTorch)
- **AdaBoost**
- **Naive Bayes**

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
We thank the dataset provided by the user **shortlikeafox**, whose repository can be found at [https://github.com/shortlikeafox/ultimate_ufc_dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset).


