"""
Random Forest with GridSearchCV
"""

import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

OUTPUT_DIRECTORY = '../data/results/random_forest/'

def run_random_forest_experiment(
        dataset_path: str,
        num_repeats: int = 30,
        train_frac: float = 0.7,
        random_seed: int = 1,
        param_grid: dict = None
) -> tuple[dict,dict,list]:
    """
    Runs random forest experiments on a single dataset
    """

    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [10,20,None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2],
            'max_features': ['sqrt','log2']
        }

    # Load data
    data = pd.read_csv(dataset_path)

    # Store metrics for repeats
    raw_metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}
    best_params_list = []

    # Repeats
    for rep in tqdm(range(num_repeats), desc=f"Processing {os.path.basename(dataset_path)}"):
        # Split into testing and training sets
        train_data = data.sample(frac=train_frac, random_state=random_seed * rep)
        test_data = data.drop(train_data.index)

        # Split features and target
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        # Train RandomForest model with GridSearchCV and predict
        model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train,y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate evaluation metrics and store
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        raw_metrics['MAPE'].append(mape)
        raw_metrics['MAE'].append(mae)
        raw_metrics['RMSE'].append(rmse)

        # Store best parameters
        best_params_list.append(grid_search.best_params_)

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

    return avg_metrics, raw_metrics, best_params_list

def run_all_random_forests(
        system_list: list,
        datasets_dir: str = '../data/datasets/',
        output_dir: str = OUTPUT_DIRECTORY
):
    """
    Run random forest experiments on all datasets
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    raw_results = []

    for system in system_list:
        system_dir = os.path.join(datasets_dir, system)
        csv_files = glob.glob(os.path.join(system_dir, '*.csv'))

        for csv_file in csv_files:
            print(f"\nProcessing: {system} - {os.path.basename(csv_file)}")

            avg_metrics, raw_metrics, best_params = run_random_forest_experiment(csv_file)

            # Store summary
            summary_results.append({
                'system': system,
                'dataset': os.path.basename(csv_file),
                **avg_metrics
            })

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

    # Save both
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(output_dir, 'rf_summary.csv'), index=False)

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(os.path.join(output_dir, 'rf_raw.csv'), index=False)

    return summary_df


if __name__ == '__main__':
    systems = ['batlik','dconvert','h2','jump3r','kanzi','lrzip','x264','xz','z3']

    experiment_results = run_all_random_forests(systems)
    print("\n=== Random Forest Experiments Complete ===")
    print(f"Results saved to: {OUTPUT_DIRECTORY}")
