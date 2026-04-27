# Random Forest with GridSearchCV for Configuration Performance Learning

This repository contains the code, data, and results for evaluating Random Forest regression with GridSearchCV hyperparameter tuning against a Linear Regression baseline for configuration performance prediction.

## Problem

Modern configurable software systems have vast configuration spaces, making it infeasible to measure every configuration's performance. Configuration performance learning trains machine learning models on a small sample of measured configurations to predict the performance of unmeasured ones. This project investigates whether Random Forest with systematic hyperparameter tuning can outperform a Linear Regression baseline on this task.

## Approach

- **Baseline:** Linear Regression
- **Proposed solution:** Random Forest with GridSearchCV (48 hyperparameter combinations, 3-fold cross-validation)
- **Evaluation:** 93 datasets from 9 software systems (batlik, dconvert, h2, jump3r, kanzi, lrzip, x264, xz, z3), 70/30 train/test split, 30 repeats, evaluated on MAPE, MAE, and RMSE
- **Statistical testing:** Wilcoxon signed-rank test with Cliff's delta effect size

## Results

Random Forest significantly outperforms the baseline across all three metrics (p < 0.001), winning on 75–82 out of 93 datasets with average error reductions of 56–66%.

## Repository Structure

```
├── requirements.pdf          # Dependencies
├── manual.pdf                # Usage instructions
├── replication.pdf           # Reproduction guide
├── requirements.txt          # Python dependencies
├── src/
│   ├── baseline.py           # Linear Regression baseline
│   ├── random_forest.py      # Random Forest + GridSearchCV
│   ├── evaluate.py           # Comparison and statistical tests
│   └── visualize.py          # Figure generation
├── data/
│   ├── datasets/             # 93 CSV datasets (9 systems)
│   └── results/              # Experimental output
└── figures/                  # Generated plots
```

## Quick Start

```bash
git clone https://github.com/ogkomori/ise-coursework.git
cd ise-coursework
pip install -r requirements.txt

python src/baseline.py
python src/random_forest.py
python src/evaluate.py
python src/visualize.py
```

See `manual.pdf` for full usage details and `replication.pdf` for step-by-step reproduction instructions.