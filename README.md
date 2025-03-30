# UFC Fight Prediction Model

## Objective
The goal of this model is to predict the winner of a UFC fight using a binary classification approach. By analyzing key features of both fighters, the model determines whether Fighter A or Fighter B will emerge victorious. This prediction is based on factors such as age, height, reach, and other relevant characteristics.

## Dataset Description
The dataset consists of fight data, where each fight is represented by the difference in features between two fighters. Each fight, denoted as **x**, is the feature difference between Fighter A and Fighter B. 

The features considered include:
- **Age**
- **Height**
- **Reach**
- **Weight** (optional)
- **Striking Accuracies** (average strikes landed per minute, etc.)
- **Takedown Accuracy** (optional)
- and more...

The features for each fight are modeled as:
$$
x = \text{{features\_fighter\_A}} - \text{{features\_fighter\_B}}
$$
Where \( x \in \mathbb{R}^n \) is the vector representing the fight, containing the differences in the respective features between Fighter A and Fighter B.

The target variable is a binary classification indicating the winner:
- **1**: Fighter A wins
- **0**: Fighter B wins

## Approach
The model predicts the outcome of a UFC fight by using the difference in features between the two fighters as input. The goal is to create a machine learning model that can accurately predict the winner based on these feature differences.

For each fight, the feature vector \( x \) is calculated as 
$$ 
x = \text{{features\_fighter\_A}} - \text{{features\_fighter\_B}} 
$$
The model will be trained on this data to learn the relationship between feature differences and fight outcomes.

## Models Implemented
The following machine learning models have been implemented to predict the fight outcomes:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **Neural Networks** (using PyTorch)

## Installation & Usage
To run the model, follow these steps:
1. Clone the repository:  
   `git clone <repository_url>`
2. Install the required dependencies:  
   `pip install -r requirements.txt`

