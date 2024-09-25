
# A Framework for Combining Machine Learning with Data-Balancing Methods for Credit Scoring

## Overview

This project, titled **"A Framework for Combining Machine Learning with Data-Balancing Methods for Credit Scoring"**, was completed during the winter of 2023 and is published now. It serves as a continuation of the previous project conducted in the fall of 2022 titled **"Combining Machine Learning with Data-balancing Methods for Credit Scoring"**. The goal of this research is to expand upon the original framework and implement additional machine learning models, feature selection techniques, and advanced data-balancing methods to enhance the performance of credit scoring systems.

Credit scoring is critical for financial institutions to assess the creditworthiness of clients and estimate the likelihood of loan default. This project focuses on leveraging advanced machine learning models in combination with data-balancing techniques to handle imbalanced datasets, which are prevalent in credit scoring problems.

## Key Objectives

- **Improve the performance of credit scoring models** by combining different machine learning algorithms with advanced data-balancing methods.
- **Handle data imbalance** through techniques such as SMOTE and its variants, which correct for the skewed distribution of credit default datasets.
- **Experiment with different feature selection methods** to reduce dataset dimensionality and improve model performance.
- **Evaluate the performance** of multiple machine learning models based on key metrics: Accuracy, Sensitivity, Specificity, AUC-ROC score, and F1 score.

## Data

This project utilizes two datasets:

1. **UCI Credit Card Dataset**: Contains 30,000 samples with 24 features, which include demographic and financial information about credit card clients.
2. **Default Dataset**: Contains 10,000 samples with 4 features, which simulate credit card default scenarios with a high imbalance ratio.

## Methods

### Machine Learning Models:
1. **Random Forest**: An ensemble of decision trees used for classification.
2. **Neural Network**: A multi-layer perceptron model for nonlinear classification.
3. **Gradient Boosting (XGBoost)**: A powerful boosting algorithm that constructs models in a sequential manner.
4. **Logistic Regression**: A simple and interpretable model used for binary classification.
5. **Heterogeneous Ensemble**: Combines the strengths of different machine learning models (Random Forest, XGBoost, Logistic Regression).

### Feature Selection Techniques:
- **L1-based Feature Selection (Support Vector Machine)**: Uses linear SVM to eliminate irrelevant features.
- **Random Forest Recursive Feature Elimination (RF-RFE)**: Recursively eliminates less important features based on their ranking by Random Forest.

### Data Balancing Techniques:
- **Random Oversampling**: Duplicates minority class samples to balance the dataset.
- **SMOTE (Synthetic Minority Oversampling Technique)**: Creates synthetic samples of the minority class by interpolating between existing instances.
- **SMOTE Variants**: Includes methods such as SMOTE-Tomek, SMOTE-OSS, and SMOTE-ENN, which enhance dataset balancing by removing noise and overlapping samples.

## Experiments

Experiments were conducted on both datasets by splitting them into training and testing sets. Two feature selection methods (L1-based and RF-RFE) were applied to the datasets, followed by different data-balancing methods. The following machine learning models were trained on the balanced datasets:
- Random Forest
- Neural Network
- Gradient Boosting (XGBoost)
- Logistic Regression
- Heterogeneous Ensemble

Each model was evaluated based on:
- **Accuracy**: The proportion of correct predictions.
- **Sensitivity**: The ability to correctly identify positive instances (defaulters).
- **Specificity**: The ability to correctly identify negative instances (non-defaulters).
- **AUC-ROC**: Measures the model's ability to distinguish between classes.
- **F1 Score**: The harmonic mean of precision and recall.

## Results

- **Best Model on UCI Credit Card Dataset**: Neural Network with Random Oversampling achieved the highest AUC score of **0.749**.
- **Best Model on Default Dataset**: Neural Network with SMOTE and RF-RFE showed strong performance with an AUC score of **0.948**.
- **Logistic Regression** struggled with highly imbalanced data, showing declining performance as data imbalance increased.

## Conclusion

This project successfully demonstrates that combining machine learning models with advanced data-balancing techniques significantly improves credit scoring performance, especially when dealing with imbalanced datasets. The best-performing models were Neural Networks and Random Forest classifiers, especially when paired with Random Forest Recursive Feature Elimination (RF-RFE) and data-balancing techniques such as SMOTE.

The framework developed here is flexible and can be extended to other classification problems, particularly in the context of imbalanced datasets. By incorporating ensemble methods, feature selection, and data-balancing techniques, this framework provides a robust solution to improving credit scoring accuracy and reliability.
