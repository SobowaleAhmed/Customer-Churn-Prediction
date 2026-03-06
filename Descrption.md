# Project Description

## What this project does

This notebook trains a neural network to predict customer churn for a telecom company. Given a customer's account details and service usage, the model outputs a probability that they will cancel their subscription.

The project was built to demonstrate how neural networks learn through forward and backward propagation, using a real-world imbalanced classification problem as the vehicle.

---

## Problem statement

Customer churn is expensive — acquiring a new customer costs significantly more than retaining an existing one. By identifying customers who are likely to churn before they do, a business can intervene with targeted retention offers. This project builds a classifier that flags high-risk customers from structured account data alone.

---

## Technical approach

The data goes through a standard supervised learning pipeline:

**Data cleaning** — `TotalCharges` is stored as a string in the raw file (some entries are blank spaces). These are coerced to numeric and missing values are imputed with the column median.

**Feature encoding** — All 15 categorical columns are label-encoded. The target column `Churn` is mapped to a binary integer (1 = churned, 0 = stayed).

**Train / test split** — 80% of the data is used for training, 20% is held out for evaluation. The split is stratified to preserve the class ratio (~26.5% churn) in both sets.

**Feature scaling** — `StandardScaler` is fit exclusively on the training set and then applied to both sets. This prevents data leakage and ensures all features are on a comparable scale before being fed to the network.

**Neural network** — A three-layer Keras Sequential model with ReLU activations in the hidden layers and Sigmoid at the output. Dropout is applied after each hidden layer to reduce overfitting.

**Training** — Up to 50 epochs with a batch size of 32. A 15% validation split is carved from the training data. `EarlyStopping` monitors validation loss with a patience of 10 epochs and restores the best weights automatically.

**Evaluation** — The held-out test set is used to report Loss, Accuracy, AUC, and a full classification report including per-class precision, recall, and F1-score.

---

## Key findings

- The model achieves **79% accuracy** and **0.84 AUC** on unseen data.
- The network is better at identifying non-churners (F1 = 0.86) than churners (F1 = 0.58), which reflects the class imbalance in the dataset.
- Early stopping triggered at epoch 23 and restored weights from epoch 13, preventing overfitting without manual tuning of the epoch count.

---

## Limitations and potential improvements

- **Class imbalance** is not explicitly addressed. Techniques like class weighting (`class_weight` in `model.fit`), oversampling (SMOTE), or threshold tuning could improve recall on the minority churn class.
- **Label encoding** for multi-class categorical features (e.g. `Contract`, `InternetService`) introduces an arbitrary ordinal relationship. One-hot encoding may be more appropriate for some features.
- **Hyperparameter search** (number of layers, units, dropout rates, learning rate) was not performed. A grid search or Bayesian optimisation could yield a stronger model.
- The notebook is designed for Google Colab and reads data directly from Google Drive. A more portable version would include a local data loading fallback.
