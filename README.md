# Customer Churn Prediction with Neural Networks

A binary classification project that predicts whether a telecom customer will churn using a multi-layer neural network built with TensorFlow/Keras. The project covers the full machine learning pipeline — data loading, exploratory analysis, preprocessing, model training with early stopping, and evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [How the Network Learns](#how-the-network-learns)

---

## Project Overview

| Property           | Detail                                            |
|--------------------|---------------------------------------------------|
| **Task**           | Binary Classification — Churn (Yes / No)          |
| **Dataset**        | Telecom Customer Churn — 7,043 rows × 21 columns  |
| **Framework**      | TensorFlow 2.x / Keras                            |
| **Optimizer**      | Adam (lr = 0.001)                                 |
| **Loss**           | Binary Cross-Entropy                              |
| **Regularisation** | Dropout + Early Stopping                          |
| **Test Accuracy**  | 79.06%                                            |
| **Test AUC**       | 0.8388                                            |

---

## Dataset

**File:** `Customer-Churn.xls` (CSV format despite the `.xls` extension — load with `pd.read_csv`)

**Key columns:**

| Column           | Type        | Description                                                  |
|------------------|-------------|--------------------------------------------------------------|
| `customerID`     | string      | Unique ID — dropped during preprocessing                     |
| `gender`         | categorical | Male / Female                                                |
| `SeniorCitizen`  | binary      | 1 = senior citizen                                           |
| `tenure`         | numeric     | Months as a customer                                         |
| `Contract`       | categorical | Month-to-month / One year / Two year                         |
| `MonthlyCharges` | numeric     | Monthly charge amount                                        |
| `TotalCharges`   | numeric     | Total charged (stored as string — fixed in preprocessing)    |
| `Churn`          | binary      | **Target** — Yes (churned) / No (stayed)                     |

**Class distribution:** ~73.5% No Churn · ~26.5% Churn

---

## Project Structure

```
.
├── Untitled1.ipynb      # Main notebook — complete pipeline
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── Customer-Churn.xls   # Dataset (upload to Google Drive before running)
```

---

## Setup & Installation

### Option A — Google Colab (recommended)

1. Upload `Untitled1.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload `Customer-Churn.xls` to your Google Drive.
3. Update the file path in Cell 2 to match your Drive location:

```python
data = pd.read_csv('/content/drive/MyDrive/<YOUR_FOLDER>/Customer-Churn.xls')
```

4. Run **Runtime → Run all**.

### Option B — Local environment

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Untitled1.ipynb
```

---

## Pipeline Walkthrough

| Step | Notebook Cell | What happens |
|------|---------------|--------------|
| 1. Imports       | Cell 1  | Load NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow |
| 2. Load data     | Cell 2  | `pd.read_csv` — outputs `describe()` summary |
| 3. Null check    | Cell 3  | `isnull().sum()` — confirms zero missing values before cleaning |
| 4. EDA           | Cell 4–5 | Bar + pie chart of churn distribution; histograms of numeric features split by churn label |
| 5. Preprocessing | Cell 6  | Drop `customerID`; fix `TotalCharges`; encode target & all categorical columns |
| 6. Split & scale | Cell 7  | 80/20 stratified train/test split; `StandardScaler` fit on train only |
| 7. Build model   | Cell 8  | Keras Sequential — Input → Dense(64) → Dropout → Dense(32) → Dropout → Dense(1) |
| 8. Compile       | Cell 9  | Adam optimizer, binary cross-entropy loss, accuracy + AUC metrics |
| 9. Train         | Cell 10 | 50 epochs max, batch size 32, 15% validation split, EarlyStopping (patience=10) |
| 10. Evaluate     | Cell 11 | Loss, accuracy, AUC on test set; full classification report |

---

## Model Architecture

```
Input Layer       →  19 features (scaled)
        ↓
Dense(64, ReLU)   →  Hidden Layer 1
Dropout(0.30)     →  Regularisation
        ↓
Dense(32, ReLU)   →  Hidden Layer 2
Dropout(0.20)     →  Regularisation
        ↓
Dense(1, Sigmoid) →  Output — churn probability in [0, 1]
```

**Total trainable parameters: 3,393**

---

## Results

```
=============================================
 TEST SET EVALUATION
=============================================

  Test Loss     : 0.4237
  Test Accuracy : 79.06%
  Test AUC      : 0.8388

--- Classification Report ---
              precision    recall  f1-score   support

    No Churn       0.84      0.88      0.86      1035
       Churn       0.62      0.53      0.58       374

    accuracy                           0.79      1409
   macro avg       0.73      0.71      0.72      1409
weighted avg       0.78      0.79      0.79      1409
```

Training used early stopping — stopped at epoch 23, best weights restored from epoch 13.

---

## How the Network Learns

**Forward Propagation**
Input features pass through each layer. At every Dense layer, the network computes a weighted sum of its inputs plus a bias, then applies an activation function (ReLU in hidden layers, Sigmoid at the output). The final neuron produces a churn probability between 0 and 1.

**Loss Computation**
The predicted probability is compared to the true label using Binary Cross-Entropy:

```
L = -(1/N) * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
```

**Backpropagation**
The chain rule propagates the gradient of the loss backwards through every layer, computing how much each weight contributed to the error.

**Weight Update — Adam Optimizer**
Weights are adjusted in the direction that reduces loss:

```
w = w - lr * (dL/dw)
```

Adam maintains adaptive per-weight learning rates for faster, more stable convergence. This cycle — forward pass → loss → backward pass → update — repeats for every mini-batch of 32 samples across all training epochs.
