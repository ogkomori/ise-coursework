"""
Linear Regression Baseline
"""

import glob
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def run_system_experiment(
        dataset_path: str,
        num_repeats: int = 3,
        train_frac: float = 0.7,
        random_seed: int = 1
) -> dict:
    """
    Runs baseline experiments on a single dataset
    """

    # Load data
    data = pd.read_csv(dataset_path)

    # Store metrics for repeats
    metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

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

        # Calculate evaluation metrics
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store metrics
        metrics['MAPE'].append(mape)
        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)

    # Calculate averages
    avg_metrics = {
        'MAPE_avg': np.mean(metrics['MAPE']),
        'MAE_avg': np.mean(metrics['MAE']),
        'RMSE_avg': np.mean(metrics['RMSE']),
        'MAPE_std': np.std(metrics['MAPE']),
        'MAE_std': np.std(metrics['MAE']),
        'RMSE_std': np.std(metrics['RMSE']),
        'num_repeats': num_repeats
    }

    return avg_metrics


def run_all_system_experiments(
        system_list: list,
        datasets_dir: str = '../data/datasets/',
        output_dir: str = '../data/results/baseline/'
) -> pd.DataFrame:
    """
    Runs baseline experiments on all datasets
    """

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for system in system_list:
        system_dir = os.path.join(datasets_dir, system)
        csv_files = glob.glob(os.path.join(system_dir, '*.csv'))

        for csv_file in csv_files:
            print(f"\nProcessing: {system} - {os.path.basename(csv_file)}")

            avg_metrics = run_system_experiment(csv_file)

            # Store results
            result = {
                'system': system,
                'dataset': os.path.basename(csv_file),
                **avg_metrics
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)

    return results_df

if __name__ == '__main__':
    systems = ['batlik','dconvert','h2','jump3r','kanzi','lrzip','x264','xz','z3']

    experiment_results = run_all_system_experiments(systems)
    print("\n=== Baseline Experiments Complete ===")
    print("Results saved to: data/results/baseline/baseline_results.csv")