# Medical Diagnosis Classification Using Random Forest and K-Nearest Neighbors

## Overview

This project applies two popular machine learning algorithms, Random Forest (RF) and K-Nearest Neighbors (KNN), to classify data from a medical diagnosis dataset (`diagnose2.csv`). We use various Python libraries for data manipulation, preprocessing, model training, evaluation, and visualization.

### Dataset

The dataset used in this project is `diagnose2.csv`. It contains medical diagnostic data, and the goal is to classify the data into relevant categories using machine learning models.

### Machine Learning Algorithms

- **Random Forest (RF):** A powerful ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
- **K-Nearest Neighbors (KNN):** A simple, instance-based learning algorithm that classifies data based on the closest training examples in the feature space.

### Libraries Used

- **Pandas:** Data manipulation and analysis.
- **NumPy:** Array processing for numerical data.
- **Scikit-learn:** Machine learning library for data preprocessing, model training, and evaluation.
- **Matplotlib:** Plotting library for visualizing data.
- **Seaborn:** Statistical data visualization.
  

## Steps in the Notebook

1. **Data Import and Exploration:**
   - Load the dataset using Pandas and explore its structure.
  
2. **Data Preprocessing:**
   - Handle missing values, feature scaling, and encoding categorical variables.
  
3. **Model Building and Training:**
   - Apply Random Forest and K-Nearest Neighbors for classification.
   - Split the data into training and testing sets using `train_test_split`.
  
4. **Model Evaluation:**
   - Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrix.
   - Use cross-validation to validate model performance.
  
5. **Visualization:**
   - Use Seaborn and Matplotlib to visualize feature relationships and model performance.

## Getting Started

### Prerequisites

To run this notebook, you need to have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
