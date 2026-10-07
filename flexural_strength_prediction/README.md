# Flexural Strength Prediction with Regression Models

A scikit-learn notebook (`flexural_strength_prediction.ipynb`) that predicts the flexural strength of fiber-reinforced concrete from its mix composition. It compares up to 19 regression models, with grid-search hyperparameter tuning for most of them.

## Dataset

`flexural_strength.xlsx` (sheet `Sheet1`): 278 samples, 12 input features and one target.

| Type | Columns |
|---|---|
| Fiber properties | `fibercontent`, `Aspect ratio`, `fiberLength` |
| Aggregates and water | `fine aggregate`, `coarse aggregate`, `water` |
| Binders and admixtures | `cementecontent`, `additivecontent`, `silicafumecontent`, `slagcontent`, `flyashcontent`, `metakaolincontent` |
| Target | `flexuralstrength` |

There are no missing values (3 duplicate rows). `Sheet2` and `Sheet3` are empty.

## What It Does

- Loads the data, strips stray spaces from column names, and makes an 80/20 train/test split (`random_state=42`)
- Trains these regressors: Linear, Ridge, Lasso, ElasticNet, Huber, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, SVR, K-NN, MLP, Gaussian Process, and degree 2 and 3 polynomial regression. XGBoost, CatBoost and LightGBM are added automatically if installed.
- Standardizes features (`StandardScaler`) for scale-sensitive models: regularized and robust linear models, SVR, K-NN, MLP, Gaussian Process and polynomial regression
- Tunes most models with `GridSearchCV` (5-fold, scored on negative MSE)
- Reports test-set MSE and R² in a table sorted by R², with bar-chart comparisons and a grid of true-vs-predicted plots

## Results

Tree-based and boosting models perform best. In a test run of the notebook's models (excluding XGBoost, CatBoost and LightGBM, which weren't installed in my environment), Extra Trees and Gradient Boosting both reached R² of about 0.92 (MSE about 0.58), Gaussian Process about 0.89, and Random Forest about 0.88. Linear models reached only about 0.34 to 0.40, which suggests the relationship between mix design and strength is strongly non-linear. Scaling the features improved SVR, K-NN and MLP considerably.

Exact numbers may vary slightly with your library versions, so rerun the notebook for your final results.

## Notes and Limitations

- **Degree 3 polynomial regression overfits badly.** It fits 455 features on about 220 training rows and produces a hugely negative R². The charts clip these values, and you can drop that model or add regularization.
- **Small dataset and a single split.** With 278 rows, one 80/20 split gives noisy estimates. Cross-validation on the full data would be more reliable.
- **Slow models.** Gaussian Process, MLP and the grid searches can take a few minutes.

## Setup

```bash
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn jupyter
pip install xgboost catboost lightgbm   # optional: adds three more models
```

If `openpyxl` is missing you'll get `ModuleNotFoundError: No module named 'openpyxl'`. If the optional boosting packages fail to install, the notebook still runs without them.

## Run

1. Put `flexural_strength.xlsx` in the same folder as `flexural_strength_prediction.ipynb`.
2. Open the notebook in VS Code or Jupyter and run all cells.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, CatBoost, LightGBM
