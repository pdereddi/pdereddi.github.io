# Crypto Address Type Classification

A scikit-learn study that classifies short text snippets by the type of cryptocurrency address they contain (Bitcoin legacy/SegWit, Ethereum, Litecoin, Dogecoin, Monero, Ripple, Cardano), or labels them `none` / `ambiguous`. It compares eight classifiers and two custom experiments on a synthetic research dataset.

## Problem

Wallet addresses appear in free text such as messages, emails and posts. Detecting them and identifying the blockchain they belong to is useful for fraud and scam detection and for compliance monitoring. This project evaluates how well classical ML models do this on engineered text features, and what each costs in time and memory.

## Dataset

`crypto_detection_research_dataset.csv`: 25,000 synthetic records, 52 columns (text length and character ratios, per-chain regex pattern counts, keyword features). The target is `crypto_type`, which is heavily imbalanced: `none` is 50% of rows, while Ripple and Cardano are about 0.2% each.

## Tech Stack

Python, Jupyter, pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## What's Implemented

- Descriptive statistics and class-distribution plots
- Label encoding and a stratified 70/30 train/validation split
- Pipelines for SVM, Logistic Regression, Decision Tree, K-NN, Naive Bayes, MLP, AdaBoost and Random Forest, benchmarked on accuracy, precision, recall, F1, peak memory and training time
- Random Forest with a custom cuckoo hash table for storing and retrieving feature rows
- Ensemble K-NN: 10 learners on random feature and sample subsets with majority voting (code included; not yet run in the saved notebook)

## Results

| Model | Accuracy | Weighted F1 | Train time |
|---|---|---|---|
| SVM | 97.4% | 0.964 | 1.8 s |
| Logistic Regression | 97.4% | 0.964 | 0.7 s |
| K-NN | 97.3% | 0.964 | 0.03 s |
| Random Forest | 97.3% | 0.963 | 2.8 s |
| MLP | 97.0% | 0.962 | 61.8 s |
| AdaBoost | 94.8% | 0.936 | 1.6 s |
| Decision Tree | 94.1% | 0.945 | 0.2 s |
| Naive Bayes | 73.7% | 0.725 | 0.05 s |

Logistic Regression and K-NN give the best accuracy-to-cost trade-off. The cuckoo-hash Random Forest reached 97.1% accuracy but only about 0.58 macro F1, showing weak performance on the rare classes.

## Limitations

- **Possible label leakage:** features such as `contains_crypto`, `address_found` and the per-chain `*_present` / `*_count` columns closely encode the target, which likely inflates scores.
- **Weighted metrics** are dominated by the large classes; macro F1 or per-class reports better reflect rare-class performance.
- The cuckoo hash table is simplified (no relocation/rehashing) and does not affect model accuracy.
- Data is synthetic, so results may not transfer to real-world text.

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook crypto_classification_analysis.ipynb
```

Place `crypto_detection_research_dataset.csv` in the same folder as the notebook.
