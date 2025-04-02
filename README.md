# UFC Fight Prediction Model

## Objective
The goal of this model is to predict the winner of a UFC fight using a binary classification approach. By analyzing key features of both fighters, the model determines whether Fighter A or Fighter B will emerge victorious. This prediction is based on factors such as age, height, reach, and other relevant characteristics.

## Dataset Description
The dataset consists of fight data, where each fight is represented by the difference in features between two fighters. Each fight, denoted as **x**, is the feature difference between Fighter A and Fighter B. 

The features considered includes:
- **Stance, Height, Reach, Weight, Current Lose Streak, Current Win Streak, etc...**
- **Striking Accuracies** (average strikes landed per minute, etc.)
- **Win by decision splits** 
- and more...

The features for each fight are modeled as:
$$x = \text{features-fighter-red} - \text{features-fighter-blue}$$

Where $$x \in \mathbb{R}^n$$ is the vector representing the fight, containing the differences in the respective features between Fighter A and Fighter B.

The target variable is a binary classification indicating the winner:
- **0**: Fighter Red wins
- **1**: Fighter Blue wins

## Approach
The model predicts the outcome of a UFC fight by using the difference in features between the two fighters as input, this is, for each fight, the feature vector $x$  is calculated as
$$x = \text{features-fighter-red} - \text{features-fighter-blue}.$$ 
The goal is to create a machine learning model that can accurately predict the winner based on these feature differences.

## Models Implemented
The following machine learning models have been implemented to predict the fight outcomes:
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

