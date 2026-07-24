import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind
import numpy as np
import os

# Set style
sns.set_style("whitegrid")

# Read the CSV file
csv_path = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/top_100_extrapolation_analysis2.csv"
df = pd.read_csv(csv_path)

# Calculate relative score difference (normalized by full score)
df['relative_score_diff'] = df['score_diff'] / (df['full_score'].abs() + 1e-8)

# Create error bins
df['error_bin'] = df['relative_score_diff'].apply(lambda x: 'High Error (>1%)' if x > 1 else 'Low Error (≤1%)')

# Create score magnitude bins (quartiles)
df['score_magnitude_bin'] = pd.qcut(df['full_score'], q=4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'], duplicates='drop')

print(f"Loaded {len(df)} samples from {csv_path}")
print(f"\nDataframe columns: {df.columns.tolist()}")
print(f"\nScore diff stats:\n{df['score_diff'].describe()}")
print(f"\nRelative score diff stats:\n{df['relative_score_diff'].describe()}")
print(f"\nError bins: {df['error_bin'].value_counts()}")
print(f"\nScore magnitude bins: {df['score_magnitude_bin'].value_counts().sort_index()}")

# Distance columns to plot
distance_cols = [
    "min_distance",
    "max_distance",
    "mean_distance",
    "min_distance_same_label",
    "max_distance_same_label",
    "mean_distance_same_label",
    "rank_difference"
]

colors = {True: '#2ecc71', False: '#e74c3c'}  # green for correct, red for incorrect

# ============================================================================
# PLOT 1: RELATIVE SCORE DIFFERENCE
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axes = axes.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df[mask][dist_col], 
            df[mask]['relative_score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df[dist_col].notna() & df['relative_score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df[valid_mask][dist_col], df[valid_mask]['relative_score_diff'])
        spearman_corr, spearman_pval = spearmanr(df[valid_mask][dist_col], df[valid_mask]['relative_score_diff'])
        
        # Add trend line
        z = np.polyfit(df[valid_mask][dist_col], df[valid_mask]['relative_score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[dist_col].min(), df[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col}\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/relative_score_diff_vs_distance.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nRelative score diff plot saved to {output_path}")

# ============================================================================
# PLOT 2: ABSOLUTE SCORE DIFFERENCE
# ============================================================================
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 13))
axes2 = axes2.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes2[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df[mask][dist_col], 
            df[mask]['score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df[dist_col].notna() & df['score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df[valid_mask][dist_col], df[valid_mask]['score_diff'])
        spearman_corr, spearman_pval = spearmanr(df[valid_mask][dist_col], df[valid_mask]['score_diff'])
        
        # Add trend line
        z = np.polyfit(df[valid_mask][dist_col], df[valid_mask]['score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[dist_col].min(), df[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col}\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path2 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/absolute_score_diff_vs_distance.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Absolute score diff plot saved to {output_path2}")

# ============================================================================
# REGRESSION PLOTS WITH CONFIDENCE INTERVALS - RELATIVE
# ============================================================================
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 13))
axes3 = axes3.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes3[idx]
    
    # Use seaborn regplot for better visualization with confidence intervals
    sns.regplot(
        data=df,
        x=dist_col,
        y='relative_score_diff',
        ax=ax,
        scatter_kws={'alpha': 0.5, 's': 60},
        line_kws={'color': 'r', 'linewidth': 2}
    )
    
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(f'{dist_col} vs Relative Score Difference', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path3 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/relative_score_diff_vs_distance_regplot.png"
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"Relative regression plot saved to {output_path3}")

# ============================================================================
# REGRESSION PLOTS WITH CONFIDENCE INTERVALS - ABSOLUTE
# ============================================================================
fig4, axes4 = plt.subplots(3, 3, figsize=(18, 13))
axes4 = axes4.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes4[idx]
    
    # Use seaborn regplot for better visualization with confidence intervals
    sns.regplot(
        data=df,
        x=dist_col,
        y='score_diff',
        ax=ax,
        scatter_kws={'alpha': 0.5, 's': 60},
        line_kws={'color': 'r', 'linewidth': 2}
    )
    
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(f'{dist_col} vs Absolute Score Difference', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path4 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/absolute_score_diff_vs_distance_regplot.png"
plt.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"Absolute regression plot saved to {output_path4}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("CORRELATION SUMMARY - RELATIVE SCORE DIFFERENCE")
print("="*60)
for dist_col in distance_cols:
    valid_mask = df[dist_col].notna() & df['relative_score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df[valid_mask][dist_col], df[valid_mask]['relative_score_diff'])
        spearman_corr, spearman_pval = spearmanr(df[valid_mask][dist_col], df[valid_mask]['relative_score_diff'])
        print(f"\n{dist_col}:")
        print(f"  Pearson:  r={pearson_corr:>7.4f}, p-value={pearson_pval:.3e}")
        print(f"  Spearman: ρ={spearman_corr:>7.4f}, p-value={spearman_pval:.3e}")

print("\n" + "="*60)
print("CORRELATION SUMMARY - ABSOLUTE SCORE DIFFERENCE")
print("="*60)
for dist_col in distance_cols:
    valid_mask = df[dist_col].notna() & df['score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df[valid_mask][dist_col], df[valid_mask]['score_diff'])
        spearman_corr, spearman_pval = spearmanr(df[valid_mask][dist_col], df[valid_mask]['score_diff'])
        print(f"\n{dist_col}:")
        print(f"  Pearson:  r={pearson_corr:>7.4f}, p-value={pearson_pval:.3e}")
        print(f"  Spearman: ρ={spearman_corr:>7.4f}, p-value={spearman_pval:.3e}")

print("\n" + "="*60)
print("SCORE DIFFERENCE STATISTICS")
print("="*60)
print(f"Max absolute score difference: {df['score_diff'].max():.6f}")
print(f"Mean absolute score difference: {df['score_diff'].mean():.6f}")
print(f"Max relative score difference: {df['relative_score_diff'].max():.6f}")
print(f"Mean relative score difference: {df['relative_score_diff'].mean():.6f}")

# ============================================================================
# BINNED ANALYSIS - BOX PLOTS BY ERROR THRESHOLD
# ============================================================================
fig5, axes5 = plt.subplots(3, 3, figsize=(18, 13))
axes5 = axes5.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes5[idx]
    
    # Create boxplot
    sns.boxplot(
        data=df,
        x='error_bin',
        y=dist_col,
        ax=ax,
        palette=['#2ecc71', '#e74c3c'],
        width=0.6,
        hue='error_bin',
        legend=False
    )
    
    # Add individual points
    sns.stripplot(
        data=df,
        x='error_bin',
        y=dist_col,
        ax=ax,
        color='black',
        alpha=0.3,
        size=4,
        jitter=True
    )
    
    # Compute t-test
    low_error = df[df['error_bin'] == 'Low Error (≤1%)'][dist_col]
    high_error = df[df['error_bin'] == 'High Error (>1%)'][dist_col]
    
    p_val = np.nan
    if len(low_error) > 0 and len(high_error) > 0:
        t_stat, p_val = ttest_ind(low_error, high_error)
        if idx == 0:
            print(f"\n{dist_col} - T-test:")
            print(f"  Low Error (≤1%): mean={low_error.mean():.4f}, std={low_error.std():.4f}")
            print(f"  High Error (>1%): mean={high_error.mean():.4f}, std={high_error.std():.4f}")
            print(f"  t-statistic={t_stat:.4f}, p-value={p_val:.3e}")
    
    ax.set_xlabel('Error Bin', fontsize=11, fontweight='bold')
    ax.set_ylabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_title(f'{dist_col} by Error Threshold\n(p={p_val:.3e})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_path5 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/binned_boxplot_by_error.png"
plt.savefig(output_path5, dpi=150, bbox_inches='tight')
print(f"\nBoxplot by error threshold saved to {output_path5}")

# ============================================================================
# BINNED ANALYSIS - VIOLIN PLOTS BY ERROR THRESHOLD
# ============================================================================
fig6, axes6 = plt.subplots(3, 3, figsize=(18, 13))
axes6 = axes6.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes6[idx]
    
    # Create violin plot
    sns.violinplot(
        data=df,
        x='error_bin',
        y=dist_col,
        ax=ax,
        palette=['#2ecc71', '#e74c3c']
    )
    
    # Add swarmplot for individual points
    sns.swarmplot(
        data=df,
        x='error_bin',
        y=dist_col,
        ax=ax,
        color='black',
        alpha=0.4,
        size=4
    )
    
    ax.set_xlabel('Error Bin', fontsize=11, fontweight='bold')
    ax.set_ylabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_title(f'{dist_col} Distribution by Error Threshold', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_path6 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/binned_violin_by_error.png"
plt.savefig(output_path6, dpi=150, bbox_inches='tight')
print(f"Violin plot by error threshold saved to {output_path6}")

# ============================================================================
# BINNED ANALYSIS - BAR PLOTS WITH MEAN DISTANCES
# ============================================================================
fig7, axes7 = plt.subplots(3, 3, figsize=(18, 13))
axes7 = axes7.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes7[idx]
    
    # Compute mean and std for each error bin
    stats_by_bin = df.groupby('error_bin')[dist_col].agg(['mean', 'std', 'count'])
    
    colors_list = ['#2ecc71', '#e74c3c']
    bins_list = stats_by_bin.index.tolist()
    
    # Create bar plot
    x_pos = np.arange(len(bins_list))
    means = stats_by_bin['mean'].values
    stds = stats_by_bin['std'].values
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Error Bin', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Mean {dist_col}', fontsize=11, fontweight='bold')
    ax.set_title(f'Mean {dist_col} by Error Threshold', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bins_list, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_path7 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/binned_mean_distances_by_error.png"
plt.savefig(output_path7, dpi=150, bbox_inches='tight')
print(f"Mean distances bar plot by error threshold saved to {output_path7}")

print("\n" + "="*60)
print("BINNED ANALYSIS SUMMARY (Error > 1% vs ≤ 1%)")
print("="*60)
for dist_col in distance_cols:
    low_error = df[df['error_bin'] == 'Low Error (≤1%)'][dist_col]
    high_error = df[df['error_bin'] == 'High Error (>1%)'][dist_col]
    
    if len(low_error) > 0 and len(high_error) > 0:
        t_stat, p_val = ttest_ind(low_error, high_error)
        print(f"\n{dist_col}:")
        print(f"  Low Error (≤1%):  n={len(low_error):3d}, mean={low_error.mean():.6f}, std={low_error.std():.6f}")
        print(f"  High Error (>1%): n={len(high_error):3d}, mean={high_error.mean():.6f}, std={high_error.std():.6f}")
        print(f"  t-test: t={t_stat:>7.4f}, p-value={p_val:.3e}")

# ============================================================================
# SCATTER PLOTS - RELATIVE SCORE DIFF - LOW ERROR (≤1%)
# ============================================================================
fig8, axes8 = plt.subplots(3, 3, figsize=(18, 13))
axes8 = axes8.flatten()

df_low_error = df[df['error_bin'] == 'Low Error (≤1%)']

for idx, dist_col in enumerate(distance_cols):
    ax = axes8[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df_low_error['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df_low_error[mask][dist_col], 
            df_low_error[mask]['relative_score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df_low_error[dist_col].notna() & df_low_error['relative_score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['relative_score_diff'])
        spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['relative_score_diff'])
        
        # Add trend line
        z = np.polyfit(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['relative_score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_low_error[dist_col].min(), df_low_error[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col} (Error ≤ 1%)\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path8 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/scatter_relative_low_error.png"
plt.savefig(output_path8, dpi=150, bbox_inches='tight')
print(f"\nRelative score diff scatter (Error ≤ 1%) saved to {output_path8}")

# ============================================================================
# SCATTER PLOTS - RELATIVE SCORE DIFF - HIGH ERROR (>1%)
# ============================================================================
fig9, axes9 = plt.subplots(3, 3, figsize=(18, 13))
axes9 = axes9.flatten()

df_high_error = df[df['error_bin'] == 'High Error (>1%)']

for idx, dist_col in enumerate(distance_cols):
    ax = axes9[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df_high_error['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df_high_error[mask][dist_col], 
            df_high_error[mask]['relative_score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df_high_error[dist_col].notna() & df_high_error['relative_score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['relative_score_diff'])
        spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['relative_score_diff'])
        
        # Add trend line
        z = np.polyfit(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['relative_score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_high_error[dist_col].min(), df_high_error[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col} (Error > 1%)\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path9 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/scatter_relative_high_error.png"
plt.savefig(output_path9, dpi=150, bbox_inches='tight')
print(f"Relative score diff scatter (Error > 1%) saved to {output_path9}")

# ============================================================================
# SCATTER PLOTS - ABSOLUTE SCORE DIFF - LOW ERROR (≤1%)
# ============================================================================
fig10, axes10 = plt.subplots(3, 3, figsize=(18, 13))
axes10 = axes10.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes10[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df_low_error['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df_low_error[mask][dist_col], 
            df_low_error[mask]['score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df_low_error[dist_col].notna() & df_low_error['score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['score_diff'])
        spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['score_diff'])
        
        # Add trend line
        z = np.polyfit(df_low_error[valid_mask][dist_col], df_low_error[valid_mask]['score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_low_error[dist_col].min(), df_low_error[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col} (Error ≤ 1%)\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path10 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/scatter_absolute_low_error.png"
plt.savefig(output_path10, dpi=150, bbox_inches='tight')
print(f"Absolute score diff scatter (Error ≤ 1%) saved to {output_path10}")

# ============================================================================
# SCATTER PLOTS - ABSOLUTE SCORE DIFF - HIGH ERROR (>1%)
# ============================================================================
fig11, axes11 = plt.subplots(3, 3, figsize=(18, 13))
axes11 = axes11.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes11[idx]
    
    # Plot scatter for each correctness status
    for is_correct, color in colors.items():
        mask = df_high_error['is_correct'] == is_correct
        label = "Correct" if is_correct else "Incorrect"
        ax.scatter(
            df_high_error[mask][dist_col], 
            df_high_error[mask]['score_diff'],
            alpha=0.6, 
            s=80,
            color=color,
            label=label,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Compute correlation
    valid_mask = df_high_error[dist_col].notna() & df_high_error['score_diff'].notna()
    if valid_mask.sum() > 0:
        pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['score_diff'])
        spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['score_diff'])
        
        # Add trend line
        z = np.polyfit(df_high_error[valid_mask][dist_col], df_high_error[valid_mask]['score_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_high_error[dist_col].min(), df_high_error[dist_col].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
    else:
        pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
    
    # Labels and formatting
    ax.set_xlabel(dist_col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
    ax.set_title(
        f'{dist_col} (Error > 1%)\n'
        f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
        fontsize=10,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path11 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/scatter_absolute_high_error.png"
plt.savefig(output_path11, dpi=150, bbox_inches='tight')
print(f"Absolute score diff scatter (Error > 1%) saved to {output_path11}")

# ============================================================================
# RANK DIFFERENCE vs SCORE ERROR - ALL DATA
# ============================================================================
fig12, axes12 = plt.subplots(1, 2, figsize=(14, 5))

# Relative score difference
ax = axes12[0]
for is_correct, color in colors.items():
    mask = df['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df[mask]['rank_difference'], 
        df[mask]['relative_score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df['rank_difference'].notna() & df['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df[valid_mask]['rank_difference'], df[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df[valid_mask]['rank_difference'], df[valid_mask]['relative_score_diff'])
    
    z = np.polyfit(df[valid_mask]['rank_difference'], df[valid_mask]['relative_score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['rank_difference'].min(), df['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Relative Score Error\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute score difference
ax = axes12[1]
for is_correct, color in colors.items():
    mask = df['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df[mask]['rank_difference'], 
        df[mask]['score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df['rank_difference'].notna() & df['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df[valid_mask]['rank_difference'], df[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df[valid_mask]['rank_difference'], df[valid_mask]['score_diff'])
    
    z = np.polyfit(df[valid_mask]['rank_difference'], df[valid_mask]['score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['rank_difference'].min(), df['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Absolute Score Error\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path12 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/rank_difference_vs_score_error.png"
plt.savefig(output_path12, dpi=150, bbox_inches='tight')
print(f"Rank difference vs score error saved to {output_path12}")

# ============================================================================
# RANK DIFFERENCE vs SCORE ERROR - LOW ERROR (≤1%)
# ============================================================================
fig13, axes13 = plt.subplots(1, 2, figsize=(14, 5))

# Relative score difference
ax = axes13[0]
for is_correct, color in colors.items():
    mask = df_low_error['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df_low_error[mask]['rank_difference'], 
        df_low_error[mask]['relative_score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df_low_error['rank_difference'].notna() & df_low_error['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['relative_score_diff'])
    
    z = np.polyfit(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['relative_score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_low_error['rank_difference'].min(), df_low_error['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Relative Score Error (Error ≤ 1%)\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute score difference
ax = axes13[1]
for is_correct, color in colors.items():
    mask = df_low_error['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df_low_error[mask]['rank_difference'], 
        df_low_error[mask]['score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df_low_error['rank_difference'].notna() & df_low_error['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['score_diff'])
    
    z = np.polyfit(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_low_error['rank_difference'].min(), df_low_error['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Absolute Score Error (Error ≤ 1%)\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path13 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/rank_difference_vs_score_error_low.png"
plt.savefig(output_path13, dpi=150, bbox_inches='tight')
print(f"Rank difference vs score error (Low error) saved to {output_path13}")

# ============================================================================
# RANK DIFFERENCE vs SCORE ERROR - HIGH ERROR (>1%)
# ============================================================================
fig14, axes14 = plt.subplots(1, 2, figsize=(14, 5))

# Relative score difference
ax = axes14[0]
for is_correct, color in colors.items():
    mask = df_high_error['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df_high_error[mask]['rank_difference'], 
        df_high_error[mask]['relative_score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df_high_error['rank_difference'].notna() & df_high_error['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['relative_score_diff'])
    
    z = np.polyfit(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['relative_score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_high_error['rank_difference'].min(), df_high_error['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Relative Score Error (Error > 1%)\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute score difference
ax = axes14[1]
for is_correct, color in colors.items():
    mask = df_high_error['is_correct'] == is_correct
    label = "Correct" if is_correct else "Incorrect"
    ax.scatter(
        df_high_error[mask]['rank_difference'], 
        df_high_error[mask]['score_diff'],
        alpha=0.6, 
        s=80,
        color=color,
        label=label,
        edgecolors='black',
        linewidth=0.5
    )

valid_mask = df_high_error['rank_difference'].notna() & df_high_error['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['score_diff'])
    
    z = np.polyfit(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['score_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_high_error['rank_difference'].min(), df_high_error['rank_difference'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Rank Difference', fontsize=11, fontweight='bold')
ax.set_ylabel('Absolute Score Difference', fontsize=11, fontweight='bold')
ax.set_title(
    f'Rank Difference vs Absolute Score Error (Error > 1%)\n'
    f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
    f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
    fontsize=10,
    fontweight='bold'
)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path14 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/rank_difference_vs_score_error_high.png"
plt.savefig(output_path14, dpi=150, bbox_inches='tight')
print(f"Rank difference vs score error (High error) saved to {output_path14}")

print("\n" + "="*60)
print("RANK DIFFERENCE vs SCORE ERROR CORRELATION")
print("="*60)

# All data
print("\nAll Data:")
valid_mask = df['rank_difference'].notna() & df['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df[valid_mask]['rank_difference'], df[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df[valid_mask]['rank_difference'], df[valid_mask]['relative_score_diff'])
    print(f"  Relative Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

valid_mask = df['rank_difference'].notna() & df['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df[valid_mask]['rank_difference'], df[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df[valid_mask]['rank_difference'], df[valid_mask]['score_diff'])
    print(f"  Absolute Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

# Low error
print("\nLow Error (≤1%):")
valid_mask = df_low_error['rank_difference'].notna() & df_low_error['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['relative_score_diff'])
    print(f"  Relative Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

valid_mask = df_low_error['rank_difference'].notna() & df_low_error['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_low_error[valid_mask]['rank_difference'], df_low_error[valid_mask]['score_diff'])
    print(f"  Absolute Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

# High error
print("\nHigh Error (>1%):")
valid_mask = df_high_error['rank_difference'].notna() & df_high_error['relative_score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['relative_score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['relative_score_diff'])
    print(f"  Relative Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

valid_mask = df_high_error['rank_difference'].notna() & df_high_error['score_diff'].notna()
if valid_mask.sum() > 0:
    pearson_corr, pearson_pval = pearsonr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['score_diff'])
    spearman_corr, spearman_pval = spearmanr(df_high_error[valid_mask]['rank_difference'], df_high_error[valid_mask]['score_diff'])
    print(f"  Absolute Score Diff: Pearson r={pearson_corr:.4f} (p={pearson_pval:.3e}), Spearman ρ={spearman_corr:.4f} (p={spearman_pval:.3e})")

# ============================================================================
# ANALYSIS BY SCORE MAGNITUDE BINS (QUARTILES)
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS BY SCORE MAGNITUDE (QUARTILES)")
print("="*60)

# Boxplots by score magnitude
fig15, axes15 = plt.subplots(3, 3, figsize=(18, 13))
axes15 = axes15.flatten()

for idx, dist_col in enumerate(distance_cols):
    ax = axes15[idx]
    
    sns.boxplot(
        data=df,
        x='score_magnitude_bin',
        y=dist_col,
        ax=ax,
        palette='Set2',
        hue='score_magnitude_bin',
        legend=False
    )
    
    sns.stripplot(
        data=df,
        x='score_magnitude_bin',
        y=dist_col,
        ax=ax,
        color='black',
        alpha=0.3,
        size=3,
        jitter=True
    )
    
    ax.set_xlabel('Score Magnitude Bin', fontsize=10, fontweight='bold')
    ax.set_ylabel(dist_col, fontsize=10, fontweight='bold')
    ax.set_title(f'{dist_col} by Score Magnitude', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

output_path15 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/boxplot_by_score_magnitude.png"
plt.savefig(output_path15, dpi=150, bbox_inches='tight')
print(f"\nBoxplot by score magnitude saved to {output_path15}")

# Bar plots showing mean relative error by score magnitude
fig16, ax = plt.subplots(figsize=(10, 6))

error_by_magnitude = df.groupby('score_magnitude_bin').agg({
    'relative_score_diff': ['mean', 'std', 'count'],
    'score_diff': ['mean', 'std']
}).round(6)

bins_order = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
means = df.groupby('score_magnitude_bin')['relative_score_diff'].mean().reindex(bins_order)
stds = df.groupby('score_magnitude_bin')['relative_score_diff'].std().reindex(bins_order)

x_pos = np.arange(len(bins_order))
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Score Magnitude Bin', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Relative Score Error', fontsize=12, fontweight='bold')
ax.set_title('Relative Score Error by Score Magnitude', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(bins_order)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_path16 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/mean_error_by_score_magnitude.png"
plt.savefig(output_path16, dpi=150, bbox_inches='tight')
print(f"Mean error by score magnitude saved to {output_path16}")

# Print summary statistics
print("\nScore Magnitude Binning Summary:")
for bin_name in bins_order:
    bin_data = df[df['score_magnitude_bin'] == bin_name]
    if len(bin_data) > 0:
        print(f"\n{bin_name}:")
        print(f"  Count: {len(bin_data)}")
        print(f"  Score range: [{bin_data['full_score'].min():.4f}, {bin_data['full_score'].max():.4f}]")
        print(f"  Relative Error: mean={bin_data['relative_score_diff'].mean():.6f}, std={bin_data['relative_score_diff'].std():.6f}")
        print(f"  Absolute Error: mean={bin_data['score_diff'].mean():.6f}, std={bin_data['score_diff'].std():.6f}")

# Scatter plots: rank_difference vs relative_score_diff colored by score magnitude
fig17, ax = plt.subplots(figsize=(12, 8))

for bin_name, color in zip(bins_order, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
    bin_data = df[df['score_magnitude_bin'] == bin_name]
    ax.scatter(
        bin_data['rank_difference'],
        bin_data['relative_score_diff'],
        alpha=0.6,
        s=80,
        label=bin_name,
        color=color,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel('Rank Difference', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Score Difference', fontsize=12, fontweight='bold')
ax.set_title('Rank Difference vs Relative Score Error\nColored by Score Magnitude', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path17 = "/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/rank_vs_error_by_score_magnitude.png"
plt.savefig(output_path17, dpi=150, bbox_inches='tight')
print(f"\nRank vs error by score magnitude saved to {output_path17}")

# ============================================================================
# SCATTER PLOTS: DISTANCE vs RANK DIFFERENCE BY SCORE MAGNITUDE
# ============================================================================
bins_order = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']

for bin_idx, bin_name in enumerate(bins_order):
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()
    
    df_bin = df[df['score_magnitude_bin'] == bin_name]
    
    for idx, dist_col in enumerate(distance_cols):
        ax = axes[idx]
        
        # Plot scatter for each correctness status
        for is_correct, color in colors.items():
            mask = df_bin['is_correct'] == is_correct
            label = "Correct" if is_correct else "Incorrect"
            ax.scatter(
                df_bin[mask][dist_col], 
                df_bin[mask]['rank_difference'],
                alpha=0.6, 
                s=80,
                color=color,
                label=label,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Compute correlation
        valid_mask = df_bin[dist_col].notna() & df_bin['rank_difference'].notna()
        if valid_mask.sum() > 1:
            pearson_corr, pearson_pval = pearsonr(df_bin[valid_mask][dist_col], df_bin[valid_mask]['rank_difference'])
            spearman_corr, spearman_pval = spearmanr(df_bin[valid_mask][dist_col], df_bin[valid_mask]['rank_difference'])
            
            # Add trend line
            z = np.polyfit(df_bin[valid_mask][dist_col], df_bin[valid_mask]['rank_difference'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_bin[dist_col].min(), df_bin[dist_col].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2, label='Trend')
        else:
            pearson_corr = pearson_pval = spearman_corr = spearman_pval = np.nan
        
        # Labels and formatting
        ax.set_xlabel(dist_col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Rank Difference', fontsize=10, fontweight='bold')
        ax.set_title(
            f'{dist_col}\n'
            f'Pearson r={pearson_corr:.3f} (p={pearson_pval:.3e})\n'
            f'Spearman ρ={spearman_corr:.3f} (p={spearman_pval:.3e})',
            fontsize=9,
            fontweight='bold'
        )
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = f"/ceph/hdd/shared/schmidt_schwinn_data_pruning/unsupervised-data-pruning/analysis/distance_vs_rank_by_magnitude_{bin_idx+1}_{bin_name.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distance vs rank difference for {bin_name} saved to {output_path}")
    plt.close()

# Print correlations summary by score magnitude
print("\n" + "="*60)
print("DISTANCE vs RANK DIFFERENCE BY SCORE MAGNITUDE")
print("="*60)

for bin_name in bins_order:
    df_bin = df[df['score_magnitude_bin'] == bin_name]
    print(f"\n{bin_name} (n={len(df_bin)}):")
    
    for dist_col in distance_cols:
        valid_mask = df_bin[dist_col].notna() & df_bin['rank_difference'].notna()
        if valid_mask.sum() > 1:
            pearson_corr, pearson_pval = pearsonr(df_bin[valid_mask][dist_col], df_bin[valid_mask]['rank_difference'])
            spearman_corr, spearman_pval = spearmanr(df_bin[valid_mask][dist_col], df_bin[valid_mask]['rank_difference'])
            print(f"  {dist_col}:")
            print(f"    Pearson:  r={pearson_corr:>7.4f}, p-value={pearson_pval:.3e}")
            print(f"    Spearman: ρ={spearman_corr:>7.4f}, p-value={spearman_pval:.3e}")

plt.show()
