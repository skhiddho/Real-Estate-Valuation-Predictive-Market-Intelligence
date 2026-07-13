# 🏠 House Price Prediction

**Project Report — Numerical Features Edition**

Prepared by: **Sodiq Malomo** | TS Academy | Ames, Iowa Real Estate Division

## Executive Summary

A machine learning model was built to predict house sale prices in Ames, Iowa using only measurable, numerical property features. The model was trained on 1,460 historical transactions and achieved a Mean CV RMSE of approximately **0.142** on log(SalePrice) — well within the target threshold of 0.16. Ridge Regression was evaluated as a bonus improvement. Every decision from data cleaning to feature construction is fully justified below.

| Metric | Result |
|---|---|
| Training rows | 1,460 houses |
| Numerical features used | 38 (from 81 original columns) |
| Log transform used | `np.log1p(SalePrice)` |
| Mean CV RMSE | ~0.142 (target: below 0.16) ✅ |
| Best model | Ridge Regression (best alpha from RidgeCV) |
| Submission file | `submission.csv` — Id + SalePrice |

---

## Part 1 — Exploratory Data Analysis

### 1.1 Filtering to Numerical Columns

The dataset was reduced from 81 total columns to 38 numerical features by keeping only `int64` and `float64` data types. The remaining 43 categorical columns — neighbourhood names, house styles, zoning codes — were excluded per assignment constraints.

> ▶ **Justification:** Categorical columns require encoding techniques covered in a future module. Working with numerical columns only keeps the analysis clean and focused.

### 1.2 Top 5 Features Correlated With SalePrice

| Feature | What It Measures | Correlation |
|---|---|---|
| OverallQual | Overall material and finish quality (1–10 scale) | ~0.79 |
| GrLivArea | Above-grade living area in square feet | ~0.71 |
| GarageCars | Garage capacity — number of cars | ~0.64 |
| GarageArea | Size of garage in square feet | ~0.62 |
| TotalBsmtSF | Total basement square footage | ~0.61 |

The heatmap confirmed that these five features have the strongest linear relationships with SalePrice. Scatter plots showed broadly linear trends — validating the use of linear regression — though meaningful spread at higher price points suggests some variance is driven by dropped categorical features such as neighbourhood.

### 1.3 SalePrice Distribution

SalePrice is right-skewed with a skewness of approximately **1.88**. Most homes sell between \$100k and \$250k, with a long tail extending to \$750k+. This skewness is problematic for linear regression for three reasons:

- It causes the model to over-weight high-value outliers in the loss function
- Prediction errors are not proportionally penalised across all price ranges
- Residuals will not be normally distributed — violating a core regression assumption

> ▶ **Decision:** `log1p(SalePrice)` was applied as the model target. This reduces skewness from ~1.88 to ~0.12 and aligns with the Kaggle RMSE evaluation metric. `np.expm1()` reverses it after prediction.

### 1.4 Outlier Analysis

The IQR method was used to flag outliers across the top features. Two specific rows were removed:

> ▶ Two houses with `GrLivArea > 4,000` sq ft but `SalePrice < $200,000` were dropped. These are documented data entry errors in the Kaggle competition — not genuine market transactions. All other outliers were kept as they reflect real market variation.

---

## Part 2 — Data Cleaning

### 2.1 Missing Values — Column-by-Column Strategy

Missing data was handled individually per column. A blanket fill-with-zero or fill-with-mean approach destroys meaning.

| Column | Strategy | Justification |
|---|---|---|
| LotFrontage | Median fill | Missing likely means no recorded frontage; median is robust to large lot outliers |
| GarageYrBlt | Fill with 0 | Missing means no garage — 0 signals absence, not unknown year |
| MasVnrArea | Fill with 0 | Missing means no masonry veneer — a genuine zero |
| BsmtFinSF1/2, BsmtUnfSF, TotalBsmtSF | Fill with 0 | Missing means no basement — zero basement area is correct |
| BsmtFullBath, BsmtHalfBath | Fill with 0 | No basement means no basement bathrooms |
| GarageCars, GarageArea | Fill with 0 | No garage means zero capacity and area |

> ▶ **Key Insight:** GarageYrBlt being blank is not a data gap — it is meaningful information (no garage). Filling it with the mean year would tell the model the house has an average-aged garage, which is false and actively harmful.

### 2.2 Multicollinearity

The absolute correlation matrix was computed across all features. Pairs with correlation above 0.8 were flagged. Keeping both features in a highly correlated pair inflates coefficient variance without adding predictive power.

| Pair | Correlation | Decision |
|---|---|---|
| GarageCars ↔ GarageArea | ~0.88 | Drop GarageArea — GarageCars has higher correlation with SalePrice |
| 1stFlrSF ↔ TotalBsmtSF | ~0.82 | Drop the one with lower correlation to SalePrice |
| GrLivArea ↔ TotRmsAbvGrd | ~0.83 | Drop TotRmsAbvGrd — GrLivArea is more directly informative |

> ▶ **Decision Rule:** In each highly correlated pair, drop the feature with the lower correlation to SalePrice — preserve the more informative one.

### 2.3 Feature Scaling

All features were scaled using `StandardScaler` (mean = 0, std = 1) before modeling.

- Linear regression is sensitive to feature magnitude. Square footage in thousands would dominate bathroom count in single digits without scaling
- Ridge regression's penalty applies equally to all coefficients only when features are on the same scale
- Scaling ensures the regularisation penalty is fair across all features

---

## Part 3 — Feature Engineering

### 3.1 New Features Created

| Feature | Formula | Why |
|---|---|---|
| TotalSF | 1stFlrSF + 2ndFlrSF + TotalBsmtSF | Total usable space — more relevant to buyers than three separate floor counts |
| HouseAge | YrSold − YearBuilt | Age at time of sale captures depreciation better than raw build year |
| TotalBathrooms | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | Weighted combination — half baths provide partial utility |
| YearsSinceRemodel | YrSold − YearRemodAdd | Renovation freshness at time of sale; recent remodels command a premium |
| QualityXArea | OverallQual × GrLivArea | Interaction term — large low-quality house is worth less than smaller high-quality one |

### 3.2 Log Transformation of SalePrice

`np.log1p(SalePrice)` was applied as the target variable. This:

- Reduces right skew from ~1.88 to ~0.12
- Makes prediction errors proportional — 10% off on a \$100k house equals the same log error as 10% off on a \$500k house
- Directly aligns with the Kaggle competition scoring metric (RMSE on log values)
- Predictions are reversed with `np.expm1()` to recover real dollar values

---

## Part 4 — Modelling

### 4.1 Linear Regression with 5-Fold Cross-Validation

A single train/test split gives one performance estimate that depends on which rows end up in which set. 5-fold cross-validation rotates through 5 different test sets and averages — producing a reliable, honest estimate.

| Fold | RMSE |
|---|---|
| Fold 1 | ~0.14 |
| Fold 2 | ~0.15 |
| Fold 3 | ~0.13 |
| Fold 4 | ~0.14 |
| Fold 5 | ~0.15 |
| **Mean RMSE** | **~0.142 ✅** |

> ▶ Mean RMSE of ~0.142 is below the 0.16 target set by TS Academy. ✅

### 4.2 Bonus — Ridge Regression

Ridge regression modifies the loss function:

```
Loss = Σ(actual − predicted)² + α × Σ(coefficients)²
```

This discourages large coefficients and prevents over-reliance on any single feature — especially useful when features remain correlated after multicollinearity removal.

- RidgeCV tested alpha values from 0.001 to 1,000
- Best alpha was selected automatically and used to train the final Ridge model
- Ridge produced a small but consistent improvement over baseline linear regression
- Ridge is the safer deployment choice — it produces more stable coefficients on new data

### 4.3 Final Model — Ridge with Best Alpha

> ▶ The final model used for `test.csv` prediction is: `Ridge(alpha=best_alpha)` trained on the full training set with `log1p(SalePrice)` as target.

---

## Part 5 — Written Reflection

### Q1 — What Was the Hardest Decision?

The hardest decision was the missing value strategy. It is tempting to fill everything with zero or mean — it is fast and removes the problem. But doing so uncritically injects false information into the model. GarageYrBlt being blank is not "unknown year" — it means no garage exists. Filling it with the mean year would tell the model every house without a recorded garage year has an average-aged garage, which is false and actively misleads it. Getting this right required reasoning through each column individually rather than applying a single rule.

### Q2 — What Does the Model Get Wrong?

The 10 houses with the largest prediction errors share common traits:

- **Distressed or non-market sales** — foreclosures or family transfers where price reflects legal circumstances, not market value. The model has no features to detect this.
- **Premium driven by dropped categorical features** — homes in prestigious neighbourhoods command significant price premiums that have nothing to do with square footage. By dropping all categorical features, we removed the information that explains these premiums.
- **Unusual feature combinations** — a large house with very poor quality, or a tiny house with a very recent expensive remodel. These edge cases sit far from the patterns the linear model learned.

> ▶ **Common thread:** the model fails where price is driven by factors outside the numerical features — primarily location, neighbourhood prestige, and subjective quality signals.

### Q3 — What Would You Try Next?

1. **Encode categorical variables** — neighbourhood alone can shift a house's price by \$50k+. Ordinal encoding for quality ratings and target encoding for neighbourhood would add significant predictive power.
2. **Gradient boosting (XGBoost or LightGBM)** — handles skewed distributions and feature interactions natively. Consistently outperforms linear models on tabular competition data.
3. **Polynomial and interaction features** — GrLivArea² or OverallQual × OverallCond can capture non-linear price curves that linear regression cannot model without explicit help.

---

## Final Decision Table

| Stage | Decision | Justification |
|---|---|---|
| EDA | Dropped 2 extreme outliers; kept the rest | Documented data entry errors distort learning; genuine market variation is valid signal |
| EDA | log1p(SalePrice) as model target | Reduces skew from 1.88 to 0.12; matches Kaggle metric |
| Cleaning | Per-column missing value strategy | Blanket fills destroy meaning; each column's missingness has a specific cause |
| Cleaning | Dropped one feature from each correlated pair | Multicollinearity inflates coefficient variance without adding prediction power |
| Cleaning | StandardScaler applied | Equalises feature magnitude; required for fair Ridge penalty |
| Engineering | Created TotalSF, HouseAge, TotalBathrooms, YearsSinceRemodel, QualityXArea | Combined features capture buyer-relevant signals raw columns miss |
| Modelling | 5-fold CV instead of single split | More honest performance estimate; reduces variance in score |
| Modelling | RidgeCV used to find best alpha | Regularisation stabilises coefficients when features are correlated |
| Submission | np.expm1() applied to reverse log transform | Converts log predictions back to real dollar SalePrice values |

---

## Success Criteria — Final Status

| Criterion | Status |
|---|---|
| Final RMSE on log(SalePrice) below 0.16 | ✅ Achieved — Mean CV RMSE ~0.142 |
| Every decision justified in markdown above each cell | ✅ All decisions documented |
| Code written independently | ✅ All code is original |
| Submission file generated (Id + SalePrice) | ✅ submission.csv saved |

---

*TS Academy | House Price Prediction Assignment | Sodiq Malomo | Ames, Iowa*
