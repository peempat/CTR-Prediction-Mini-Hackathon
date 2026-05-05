# CTR Prediction — Ad Click-Through Rate

**Student:** ปัฐธกร (602016)

## Overview

Predict the probability that a user clicks on an ad (CTR — Click-Through Rate) given impression-level features. The evaluation metric is **Normalized Cross-Entropy (NCE)**, where values below 1.0 beat the naive baseline.

```
NCE = model_logloss / baseline_logloss
```

A lower NCE is better. The baseline predicts the training-set average CTR (≈ 0.3069) for every impression.

## Dataset

| Split | Rows | Columns |
|-------|------|---------|
| Train | 374,590 | 28 |
| Test  | 125,410 | 27 |

Key features include: user ID, timestamp, device info, banner position, site/app metadata, ad quality score, historical user CTR, and several anonymous categorical features (C1, C15, C16, C21).

## Pipeline

### 1. Feature Engineering

| Category | Features Created |
|----------|-----------------|
| Time | `week`, `day`, `hour_sin`, `hour_cos`, `is_weekend`, `is_primetime`, `is_morning` |
| Null flags | `has_site`, `has_app` |
| Frequency encoding | `user_id`, `ad_id`, `ad_campaign_id`, `publisher_id`, `site_id`, `app_id`, `site_domain`, `app_domain`, `device_model` |
| Label encoding | `device_type`, `device_conn_type`, `banner_pos`, `site_category`, `user_segment`, `creative_size`, `day_of_week`, C-columns |
| Interactions | `user_depth_log`, `ctr_x_quality`, `ctr_x_depth` |

**Total: 33 features**

### 2. Time-Based CV Split

Training data is split by ISO week — weeks 1–2 used for training, week 3 held out as the validation fold. This prevents data leakage from future impressions into the model.

| Fold | Rows |
|------|------|
| Train | 250,046 |
| Validation (week 3) | 124,544 |

### 3. Models

Three gradient-boosted tree models are trained independently:

| Model | Key Hyperparameters | Val NCE |
|-------|---------------------|---------|
| LightGBM | lr=0.025, leaves=63, early stop=50 | 0.8278 |
| XGBoost | lr=0.03, depth=7, early stop=100 | 0.8281 |
| CatBoost | lr=0.05, depth=7, early stop=50 | 0.8268 |

### 4. Ensemble

Models are combined using inverse-NCE weighting (better models get higher weight):

```
weight_i = (1 / NCE_i) / sum(1 / NCE_j)
```

The resulting weights are approximately equal (~0.333 each).

### 5. Calibration

Isotonic Regression is applied to the ensemble's validation predictions, then used to transform test predictions. This corrects any systematic probability bias.

## Results

| Stage | Val NCE |
|-------|---------|
| LightGBM | 0.8278 |
| XGBoost | 0.8281 |
| CatBoost | 0.8268 |
| Ensemble | 0.8271 |
| **Calibrated Ensemble** | **0.8254** |

All models beat the baseline (NCE < 1.0).

## Requirements

```
lightgbm
xgboost
catboost
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy matplotlib seaborn
```

## Usage

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the working directory (or unzip `the-ad-ecosystem-and-ctr-prediction.zip`).
2. Run all cells in `602016_ปัฐธกร.ipynb`.
3. The final predictions are saved to `submission.csv`.

## Output

`submission.csv` contains two columns:

| Column | Description |
|--------|-------------|
| `impression_id` | Unique impression identifier |
| `clicked` | Predicted click probability (0–1) |
