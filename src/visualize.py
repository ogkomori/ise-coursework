"""
Create publication-quality figures for the report
NO SCREENSHOTS - only proper matplotlib/seaborn plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIRECTORY = '../figures'

def plot_metric_comparison_boxplot(merged_df, metric='MAPE', output_dir=OUTPUT_DIRECTORY):
  """
  Create box plot comparing baseline vs RF for a specific metric
  """
  os.makedirs(output_dir, exist_ok=True)

  baseline_col = f'{metric}_Baseline'
  rf_col = f'{metric}_RF'

  # Prepare data for plotting
  data_for_plot = pd.DataFrame({
    'Baseline': merged_df[baseline_col],
    'Random Forest': merged_df[rf_col]
  })

  fig, ax = plt.subplots(figsize=(6, 4))

  bp = ax.boxplot(
    [data_for_plot['Baseline'], data_for_plot['Random Forest']],
    tick_labels=['Baseline\n(Linear Regression)', 'Random Forest\n(GridSearchCV)'],
    patch_artist=True,
    widths=0.6
  )

  # Customize colors
  colors = ['#FF9999', '#66B2FF']
  for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

  ax.set_ylabel(f'{metric}', fontsize=11)
  ax.set_title(f'{metric} Comparison: Baseline vs. Random Forest', fontsize=12, fontweight='bold')
  ax.grid(axis='y', alpha=0.3)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f'{metric}_boxplot.pdf'), dpi=300, bbox_inches='tight')
  plt.savefig(os.path.join(output_dir, f'{metric}_boxplot.png'), dpi=300, bbox_inches='tight')
  plt.close()


def plot_improvement_bar_chart(merged_df, output_dir=OUTPUT_DIRECTORY):
  """
  Create bar chart showing % improvement by system
  """
  os.makedirs(output_dir, exist_ok=True)

  # Calculate average improvement per system
  system_improvements = merged_df.groupby('System').agg({
    'MAPE_Δ%': 'mean',
    'MAE_Δ%': 'mean',
    'RMSE_Δ%': 'mean'
  }).reset_index()

  # Sort by MAPE improvement
  system_improvements = system_improvements.sort_values('MAPE_Δ%')

  fig, ax = plt.subplots(figsize=(10, 5))

  x = np.arange(len(system_improvements))
  width = 0.25

  ax.bar(x - width, system_improvements['MAPE_Δ%'], width, label='MAPE', alpha=0.8)
  ax.bar(x, system_improvements['MAE_Δ%'], width, label='MAE', alpha=0.8)
  ax.bar(x + width, system_improvements['RMSE_Δ%'], width, label='RMSE', alpha=0.8)

  ax.set_xlabel('Software System', fontsize=11)
  ax.set_ylabel('Improvement (%)', fontsize=11)
  ax.set_title('Performance Improvement by System\n(Negative = RF is Better)', fontsize=12, fontweight='bold')
  ax.set_xticks(x)
  ax.set_xticklabels(system_improvements['System'], rotation=45, ha='right')
  ax.legend(fontsize=10)
  ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
  ax.grid(axis='y', alpha=0.3)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, 'improvement_by_system.pdf'), dpi=300, bbox_inches='tight')
  plt.savefig(os.path.join(output_dir, 'improvement_by_system.png'), dpi=300, bbox_inches='tight')
  plt.close()


def plot_scatter_comparison(merged_df, metric='MAPE', output_dir=OUTPUT_DIRECTORY):
  """
  Create scatter plot showing baseline vs RF performance
  Points below diagonal = RF is better
  """
  os.makedirs(output_dir, exist_ok=True)

  baseline_col = f'{metric}_Baseline'
  rf_col = f'{metric}_RF'

  fig, ax = plt.subplots(figsize=(6, 6))

  # Plot points
  ax.scatter(merged_df[baseline_col], merged_df[rf_col], alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

  # Diagonal line (y=x)
  min_val = min(merged_df[baseline_col].min(), merged_df[rf_col].min())
  max_val = max(merged_df[baseline_col].max(), merged_df[rf_col].max())
  ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')

  ax.set_xlabel(f'Baseline {metric}', fontsize=11)
  ax.set_ylabel(f'Random Forest {metric}', fontsize=11)
  ax.set_title(f'{metric}: Baseline vs. Random Forest\n(Below diagonal = RF is better)', fontsize=12, fontweight='bold')
  ax.legend(fontsize=10)
  ax.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f'{metric}_scatter.pdf'), dpi=300, bbox_inches='tight')
  plt.savefig(os.path.join(output_dir, f'{metric}_scatter.png'), dpi=300, bbox_inches='tight')
  plt.close()


def create_all_figures(merged_df):
  """Generate all figures for the report"""

  print("\nGenerating figures...")

  # Box plots for each metric
  for metric in ['MAPE', 'MAE', 'RMSE']:
    plot_metric_comparison_boxplot(merged_df, metric)
    print(f"  ✓ {metric} box plot created")

  # Improvement bar chart
  plot_improvement_bar_chart(merged_df)
  print("  ✓ Improvement bar chart created")

  # Scatter plots
  for metric in ['MAPE', 'MAE', 'RMSE']:
    plot_scatter_comparison(merged_df, metric)
    print(f"  ✓ {metric} scatter plot created")

  print("\nAll figures saved to: figures/")
  print("Format: Both PDF (for LaTeX) and PNG (for general use)")


if __name__ == "__main__":
  # Load merged results
  df = pd.read_csv('../data/results/evaluation/comparison_table.csv')
  for m in ['MAPE', 'MAE', 'RMSE']:
    df[f'{m}_Baseline'] = df[f'{m}_Baseline'].astype(float)
    df[f'{m}_RF'] = df[f'{m}_RF'].astype(float)
    df[f'{m}_Δ%'] = df[f'{m}_Δ%'].str.replace('%', '').astype(float)

  create_all_figures(df)