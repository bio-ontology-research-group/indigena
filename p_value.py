from scipy import stats

# Metric M scores for each fold
method_a_folds = [0.95, 0.86, 0.89, 0.95, 0.94, 0.92, 0.95, 0.94, 0.93, 0.93] # convkb-d
method_b_folds = [0.90, 0.88, 0.89, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.91] # lin-bma
   

# Paired t-test (one-tailed since you claim A > B)
t_stat, p_value = stats.ttest_rel(method_a_folds, method_b_folds, alternative='greater')

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Non-parametric alternative (more robust with small sample size)
stat, p_value = stats.wilcoxon(method_a_folds, method_b_folds, alternative='greater')
print(f"Wilcoxon p-value: {p_value:.4f}")
