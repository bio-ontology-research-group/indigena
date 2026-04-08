from scipy import stats

# Metric M scores for each fold
method_a_folds = [0.95, 0.86, 0.89, 0.95, 0.94, 0.92, 0.95, 0.94, 0.93, 0.93] # convkb-d
method_b_folds = [0.90, 0.88, 0.89, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.91] # lin-bma

all_lin_bma_ranks = []
all_convkb_d_ranks = []

for fold in range(10):
    lin_file = f"data/baseline_results/resnik_lin_bma_fold{fold}_results_ranks.txt"
    convkb_d_file = f"data/results/kge_results_convkb_transd_transductive_fold_{fold}_seed_0_dim_100_bs_8192_lr_1e-05_hdr_0_ranks.txt"

    with open(lin_file, 'r') as f:
        lines = f.readlines()
        lin_pairs = [l.strip().split('\t') for l in lines]
        lin_diseases = [pair[0] for pair in lin_pairs]
        lin_ranks = [int(pair[1]) for pair in lin_pairs]

    with open(convkb_d_file, 'r') as f:
        lines = f.readlines()
        convkb_d_pairs = [l.strip().split('\t') for l in lines]
        convkb_d_diseases = [pair[0] for pair in convkb_d_pairs]
        convkb_d_ranks = [int(pair[1]) for pair in convkb_d_pairs]

    assert lin_diseases == convkb_d_diseases, "Diseases do not match between methods"

    all_lin_bma_ranks.extend(lin_ranks)
    all_convkb_d_ranks.extend(convkb_d_ranks)



# Paired t-test (one-tailed since you claim A > B)
t_stat, p_value = stats.ttest_rel(all_convkb_d_ranks, all_lin_bma_ranks, alternative='less')

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Non-parametric alternative (more robust with small sample size)
stat, p_value = stats.wilcoxon(all_convkb_d_ranks, all_lin_bma_ranks, alternative='less')
print(f"Wilcoxon p-value: {p_value:.4f}")
