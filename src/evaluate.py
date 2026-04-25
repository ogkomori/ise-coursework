"""
Evaluation and statistical comparison between baseline and random forest
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_DIRECTORY = '../data/results/evaluation/'
BASELINE_PATH = '../data/results/baseline/baseline_summary.csv'
RF_PATH = '../data/results/random_forest/rf_summary.csv'

def load_results(
        baseline_path: str,
        rf_path: str
) -> pd.DataFrame:
    """
    Loads results from both approaches
    """

    baseline_df = pd.read_csv(baseline_path)
    rf_df = pd.read_csv(rf_path)

    baseline_df = baseline_df.rename(columns={'MAPE_avg': 'MAPE', 'MAE_avg': 'MAE', 'RMSE_avg': 'RMSE'})
    rf_df = rf_df.rename(columns={'MAPE_avg': 'MAPE', 'MAE_avg': 'MAE', 'RMSE_avg': 'RMSE'})

    merged = pd.merge(
        baseline_df, rf_df,
        on=['system','dataset'],
        suffixes=('_baseline','_rf')
    )

    print(f'This merged DataFrame has {len(merged)} rows.')
    return merged

def wilcoxon_test(
        baseline_values: np.ndarray,
        rf_values: np.ndarray
) -> tuple[float,float]:
    """
    Performs Wilcoxon signed-rank test
    """

    statistic, p_value = stats.wilcoxon(baseline_values, rf_values, alternative='greater')
    return statistic, p_value

def cliff_delta(
        baseline_values: np.ndarray,
        rf_values: np.ndarray
) -> float:
    """
    Calculates Cliff's Delta effect size
    """

    n1, n2 = len(baseline_values), len(rf_values)
    dominance = 0
    for b in baseline_values:
        for r in rf_values:
            if b > r:
                dominance += 1
            elif b < r:
                dominance -= 1

    cd = dominance / (n1 * n2)
    return cd

def run_statistical_comparison(
        merged_df: pd.DataFrame,
        alpha: float = 0.05,
        output_dir: str = OUTPUT_DIRECTORY
) -> pd.DataFrame:
    """
    Performs statistical tests comparing baseline vs RF
    """

    print("Starting statistical comparison")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for metric in ['MAPE','MAE','RMSE']:
        baseline_col = f'{metric}_baseline'
        rf_col = f'{metric}_rf'

        baseline_values = merged_df[baseline_col].values
        rf_values = merged_df[rf_col].values

        statistic, p_value = wilcoxon_test(baseline_values, rf_values)
        print(f"Wilcoxon test done for {metric}")

        cd = cliff_delta(baseline_values, rf_values)
        print(f"Cliff's Delta done for {metric}")

        rf_wins = (rf_values < baseline_values).sum()
        baseline_wins = (rf_values > baseline_values).sum()
        ties = (rf_values == baseline_values).sum()

        results[metric] = {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cliff_delta': cd,
            'rf_wins': rf_wins,
            'baseline_wins': baseline_wins,
            'ties': ties,
            'total': len(baseline_values)
        }

    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Metric'
    results_df.to_csv(os.path.join(output_dir, 'stats_results.csv'))
    print("Statistical comparison saved.")

    return results_df


def calculate_improvements(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage improvements
    """

    for metric in ['MAPE','MAE','RMSE']:
        baseline_col = f'{metric}_baseline'
        rf_col = f'{metric}_rf'

        merged_df[f'{metric}_improvement'] = (
            (merged_df[rf_col] - merged_df[baseline_col]) / merged_df[baseline_col] * 100
        )

    return merged_df

def create_comparison_table(
        merged_df: pd.DataFrame,
        output_dir: str = OUTPUT_DIRECTORY
) -> pd.DataFrame:
    """
    Creates summary comparison table
    """

    print("Starting comparison table")
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for _, row in merged_df.iterrows():
        summary.append({
            'System': row['system'],
            'Dataset': row['dataset'],
            'MAPE_Baseline': f"{row['MAPE_baseline']:.4f}",
            'MAPE_RF': f"{row['MAPE_rf']:.4f}",
            'MAPE_Δ%': f"{row['MAPE_improvement']:.2f}%",
            'MAE_Baseline': f"{row['MAE_baseline']:.4f}",
            'MAE_RF': f"{row['MAE_rf']:.4f}",
            'MAE_Δ%': f"{row['MAE_improvement']:.2f}%",
            'RMSE_Baseline': f"{row['RMSE_baseline']:.4f}",
            'RMSE_RF': f"{row['RMSE_rf']:.4f}",
            'RMSE_Δ%': f"{row['RMSE_improvement']:.2f}%",
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'))
    print("Comparison table saved.")

    return summary_df


if __name__ == '__main__':
    df = load_results(BASELINE_PATH, RF_PATH)
    df = calculate_improvements(df)

    sr = run_statistical_comparison(df)
    ct = create_comparison_table(df)

    print("\n" + "="*60)
    print("Evaluation complete.")
    print("="*60)
