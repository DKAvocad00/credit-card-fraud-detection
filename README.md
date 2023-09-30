# Credit Card Fraud Detection

## Overview

This project focuses on detecting credit card fraud using machine learning techniques. The dataset exhibits a significant class imbalance, with the positive class (fraudulent transactions) being sparsely represented. The analysis involves exploring different strategies, such as oversampling using SMOTE and undersampling using RandomUnderSampler, to address this class imbalance.

## Files

1. **credit_card_fraud_detection.ipynb**: Jupyter Notebook containing the main code for the project.
2. **utils.py**: Python script containing utility functions for data preprocessing, model evaluation, and visualization.

## Dataset Analysis

During the dataset analysis, the following key observations were made:

- Strong class imbalance: The positive class (fraudulent transactions) is underrepresented.
- Bimodal distribution of the Time feature: Addressed by applying logarithmic transformation.

## Data Preprocessing

Two main approaches were employed to tackle class imbalance:

1. **Oversampling (SMOTE):** Synthetic Minority Over-sampling Technique was used to oversample the minority class.
2. **Undersampling (RandomUnderSampler):** Random instances from the majority class were removed to balance the class distribution.

## Models Explored

The following classification models were explored:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree

Each model was trained on both undersampled and oversampled datasets, and the best hyperparameters were determined through cross-validation.

## Model Evaluation

The evaluation metric used is recall, given the importance of identifying fraudulent transactions. The recall scores for the best models are as follows:

- Logistic Regression: 0.891892
- Decision Tree Classifier: 0.851351
- SVC: 0.932432
- K-Nearest Neighbors: 0.851351

## Final Model Selection

Logistic Regression and SVC emerged as the top-performing models. The final evaluation was conducted using Logistic Regression due to its superior performance and efficiency.

## Final Metrics

The final performance metrics for the Logistic Regression model are as follows:

- ROC AUC: 0.927941
- Accuracy: 0.909157
- Precision: 0.016018
- Recall: 0.851351
- F1-score: 0.031445

These metrics were obtained using cross-validation with SMOTE during each iteration.

## Recommendations

- In the case of pronounced class imbalance, consider sampling techniques such as oversampling (preferably) or undersampling.
- Conduct sampling during cross-validation to obtain more reliable model performance metrics.
- Logistic Regression and SVC demonstrated superior performance in this credit card fraud detection task.

## Conclusion

This project highlights the importance of addressing class imbalance in credit card fraud detection. The chosen models, Logistic Regression and SVC, provide a solid foundation for building a robust fraud detection system. Adjustments to sampling techniques and model parameters may be explored further for continuous improvement.
