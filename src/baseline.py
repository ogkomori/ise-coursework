"""
Linear Regression Baseline
"""

import glob
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def run_baseline_experiment(
        dataset_path: str,
        num_repeats: int = 30,
        train_frac: float = 0.7,
        random_seed: int = 1
) -> tuple[dict,dict]:
    """
    Runs baseline experiments on a single dataset
    """

    # Load data
    data = pd.read_csv(dataset_path)

    # Store raw metrics for repeats
    raw_metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

    # Repeats
    for rep in range(num_repeats):
        # Split into testing and training sets
        train_data = data.sample(frac=train_frac, random_state=random_seed * rep)
        test_data = data.drop(train_data.index)

        # Split features and target
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        # Initialise model, train, and predict
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation raw metrics
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store raw metrics
        raw_metrics['MAPE'].append(mape)
        raw_metrics['MAE'].append(mae)
        raw_metrics['RMSE'].append(rmse)

    # Calculate averages
    avg_metrics = {
        'MAPE_avg': np.mean(raw_metrics['MAPE']),
        'MAE_avg': np.mean(raw_metrics['MAE']),
        'RMSE_avg': np.mean(raw_metrics['RMSE']),
        'MAPE_std': np.std(raw_metrics['MAPE']),
        'MAE_std': np.std(raw_metrics['MAE']),
        'RMSE_std': np.std(raw_metrics['RMSE']),
        'num_repeats': num_repeats
    }

    return avg_metrics, raw_metrics


def run_all_baseline_experiments(
        system_list: list,
        datasets_dir: str = '../data/datasets/',
        output_dir: str = '../data/results/baseline/'
) -> pd.DataFrame:
    """
    Runs baseline experiments on all datasets
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    raw_results = []

    for system in system_list:
        system_dir = os.path.join(datasets_dir, system)
        csv_files = glob.glob(os.path.join(system_dir, '*.csv'))

        for csv_file in csv_files:
            print(f"\nProcessing: {system} - {os.path.basename(csv_file)}")

            avg_metrics, raw_metrics = run_baseline_experiment(csv_file)

            # Store summary results
            result = {
                'system': system,
                'dataset': os.path.basename(csv_file),
                **avg_metrics
            }
            summary_results.append(result)

            # Store raw per-repeat results
            for rep_idx in range(len(raw_metrics['MAPE'])):
                raw_results.append({
                    'system': system,
                    'dataset': os.path.basename(csv_file),
                    'repeat': rep_idx,
                    'MAPE': raw_metrics['MAPE'][rep_idx],
                    'MAE': raw_metrics['MAE'][rep_idx],
                    'RMSE': raw_metrics['RMSE'][rep_idx]
                })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(output_dir, 'baseline_raw.csv'), index=False)

    return summary_df


if __name__ == '__main__':
    systems = ['batlik','dconvert','h2','jump3r','kanzi','lrzip','x264','xz','z3']

    experiment_results = run_all_baseline_experiments(systems)
    print("\n=== Baseline Experiments Complete ===")
    print("Results saved to: data/results/baseline/")