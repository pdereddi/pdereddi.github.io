# Employee Salary Level Classification

A scikit-learn notebook that predicts an employee's salary level (`low`, `medium`, `high`) from HR attributes and compares six classifiers using 10-fold stratified cross-validation.

## Dataset

`employees.csv`: 14,999 records, 10 columns.

| Type | Columns |
|---|---|
| Numeric | `satisfactoryLevel`, `lastEvaluation`, `numberOfProjects`, `avgMonthlyHours`, `timeSpent.company` |
| Binary | `workAccident`, `left`, `promotionInLast5years` |
| Categorical | `dept` |
| Target | `salary` (`low` 48.8%, `medium` 43.0%, `high` 8.2%) |

About 20% of rows (3,008) are exact duplicates. The notebook has a `DROP_DUPLICATES` flag to remove them.

## What It Does

- Loads and inspects the data
- Plots boxplots of numeric features by salary level
- Builds preprocessing pipelines: `StandardScaler` for numeric columns and `OneHotEncoder` for `dept`
- Evaluates Logistic Regression, Naive Bayes, K-NN, SVM, Decision Tree and a Neural Network (MLP) with accuracy, weighted precision, weighted recall, weighted F1 and macro F1

## Results (10-fold CV, duplicates kept)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Decision Tree | 0.613 | 0.614 |
| Neural Network | 0.515 | 0.501 |
| SVM | 0.513 | 0.478 |
| K-NN | 0.510 | 0.504 |
| Logistic Regression | 0.499 | 0.462 |
| Naive Bayes | 0.484 | 0.410 |

Salary level is hard to predict from these features. Most models only slightly beat guessing the majority class (about 49%). Decision Tree and K-NN scores may be inflated by the duplicate rows, so compare with `DROP_DUPLICATES = True`.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install -U threadpoolctl
```

Keep `threadpoolctl` up to date. An old version crashed K-NN on Windows/Anaconda with `AttributeError: 'NoneType' object has no attribute 'split'`.

## Run

1. Put `employees.csv` in the same folder as `employee_analysis.ipynb`.
2. Open the notebook in VS Code or Jupyter and run all cells.

SVM and the Neural Network are the slowest models and may take a few minutes.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn
