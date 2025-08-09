# Project title
- Regression — End-to-End: House Price Prediction (XGB, RandomForest, Ridge) with K-Means clustering
![regression image](https://github.com/user-attachments/assets/ef96d7ca-f438-4d45-b83b-47cff5fa5a89)
# Objective
- Segment and model house prices to produce accurate SalePrice predictions. Build an end-to-end pipeline that: prepares data, engineers informative features (including cluster features), tunes tree-based and -  - linear regressors, evaluates on a hold-out set, and delivers interpretable model outputs for downstream use.
# Why we do this project
### Practical goal: predict SalePrice for houses (supervised regression).
### Business value: better pricing decisions, market analysis, and prioritization for renovation/marketing.
### Technical learning: full ML lifecycle — EDA, feature engineering, clustering for segmentation, hyperparameter tuning, model evaluation, and comparison of linear vs tree methods.
### Insight: understand which house attributes drive price and detect outliers/segments.
# Step-by-step approach (high level)
- Data ingestion & overview — understand shape, types, and missingness.
- EDA — distributions, relationships with target, missing-value patterns.
- Feature selection — use mutual information and correlation to shortlist features.
- Feature engineering — create derived features and cluster features; encode categoricals.
- Scaling / normalization where required (for some algorithms/metrics).
- Train / validation split — hold out a validation set for honest evaluation.
- Modeling & tuning — try baseline linear models, tree models, and tune hyperparameters (Grid/Randomized search).
- Evaluation — compare models on MAE (and optionally RMSE/R²), inspect residuals.
- Interpretation & deliverables — cluster profiles, feature importance, saved models, final predictions, and recommended next steps.
- Exploratory Data Analysis (what to inspect & why)
- Missing values: count per column and decide drop vs impute; some categorical columns had large missingness and were dropped or filled.
- Univariate analysis: histograms/boxplots for numerical features (SalePrice skewness is common).
- Bivariate analysis: scatter/stripplots of continuous and categorical features vs SalePrice to visually detect strong predictors (GrLivArea, GarageArea, YearBuilt, etc.).
- Correlation matrix: find strongly correlated numeric predictors (avoid multicollinearity for linear models).
- Mutual information: rank numeric and categorical features by MI with SalePrice to guide selection.
# Outliers: detect extreme values (e.g., very large GrLivArea, GarageArea) that affect model error—note tree models are more robust than linear.
- Feature selection (what you chose and why)
- Numerical features selected (example): OverallQual, GrLivArea, GarageCars, YearBuilt, TotalBsmtSF, GarageArea, 1stFlrSF, FullBath, TotRmsAbvGrd, 2ndFlrSF, Fireplaces, LotArea, OpenPorchSF (these had MI > 0.15).
- Categorical features selected (example): ExterQual, BsmtQual, KitchenQual, GarageFinish, GarageType, Foundation, HeatingQC, BsmtFinType1 (MI > 0.15).
- Selection principle: keep features with demonstrable information (MI or strong visual relationship) and drop or postpone features with very low information or extreme missingness.
- Feature engineering (what you created and why)
# Derived numeric features:
- grlivperfirstflr = GrLivArea / 1stFlrSF (density/second-floor effect).
- Porchtypes = count of porch-related areas > 0 (amenity count).
- 2nd_flr flag for presence of second floor (binary).
- Totalbath = sum of full/half/basement baths (total bathrooms).
- MeanpriceNBH = neighborhood mean SalePrice (location signal).
- Clustering feature: K-Means clusters (e.g., clusters of floor/area features) appended as a categorical/ordinal cluster id to capture latent neighborhood/house-type segmentation.
- Categorical encoding: factorization/ordinal mapping for high-MI categories (so tree models can use them directly).
- Scaling/normalization: careful normalization when combining features or feeding distance-based models (and to standardize numeric ranges where needed).
- Notes: consider log transforming the target (SalePrice) if residuals are skewed; for tree models this is optional but often helpful.
# Model training (models tried, tuning approach)
### Baselines: Linear Regression (baseline MAE reported), Ridge regression (GridSearchCV over alpha).
### Tree models: RandomForestRegressor (RandomizedSearchCV to limit expensive grid search) and XGBRegressor (GridSearchCV over n_estimators, max_depth, learning_rate).
### Other experiments: SVR / MLP were considered but tree models performed best on this dataset.
### Hyperparameter tuning: cross-validation (CV=3 or 5) with neg_mean_absolute_error as scoring; for expensive grids prefer RandomizedSearchCV and parallel n_jobs=-1.
### Important practical fixes: reduce parameter grid size for RandomForest to avoid extremely long runs; use early termination or lower CV folds while iterating.
# Model testing (how models were evaluated & key results)
- Metric used: Mean Absolute Error (MAE) on held-out validation set (clear, interpretable error measure in same units as target). You can also report RMSE and R².
### Reported validation MAE (from your run):
- Linear Regression MAE: ~23,377
- RandomForest MAE: ~18,322
- XGBoost MAE: ~16,798 (best among the tried models on validation)
- Cross-validation diagnostics: CV mean and std (watch out for NaNs in CV std — handle before printing).
- Residual analysis: check residual distributions and residual vs predicted plots to detect heteroscedasticity or systematic errors.
# Model selection: choose the model balancing validation MAE, stability (CV std), inference speed, and interpretability. XGBoost gave the best MAE in this workflow.
# Output (deliverables & recommended artifacts)
<img width="454" height="48" alt="Screenshot 2025-08-09 145602" src="https://github.com/user-attachments/assets/3dae6ba9-1777-45b6-af23-1d53d39395e5" />
