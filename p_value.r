# Load required libraries
library(stats)

# Read the data from all folds
all_lin_bma_ranks <- c()
all_convkb_d_ranks <- c()

for (fold in 0:9) {
  lin_file <- sprintf("data/baseline_results/resnik_lin_bma_fold%d_results_ranks.txt", fold)
  convkb_d_file <- sprintf("data/results/kge_results_convkb_transd_transductive_fold_%d_seed_0_dim_100_bs_8192_lr_1e-05_hdr_0_ranks.txt", fold)
  
  # Read lin_bma ranks
  lin_data <- read.table(lin_file, sep="\t", stringsAsFactors=FALSE)
  lin_diseases <- lin_data[,1]
  lin_ranks <- as.integer(lin_data[,2])
  
  # Read convkb_d ranks
  convkb_d_data <- read.table(convkb_d_file, sep="\t", stringsAsFactors=FALSE)
  convkb_d_diseases <- convkb_d_data[,1]
  convkb_d_ranks <- as.integer(convkb_d_data[,2])
  
  # Verify diseases match
  if (!all(lin_diseases == convkb_d_diseases)) {
    stop(sprintf("Diseases do not match in fold %d", fold))
  }
  
  # Append to overall vectors
  all_lin_bma_ranks <- c(all_lin_bma_ranks, lin_ranks)
  all_convkb_d_ranks <- c(all_convkb_d_ranks, convkb_d_ranks)
}

cat("Total samples:", length(all_convkb_d_ranks), "\n\n")

# ============================================================
# 1. WILCOXON SIGNED-RANK TEST (for paired data)
# ============================================================
cat(strrep("=", 60), "\n")
cat("WILCOXON SIGNED-RANK TEST (Paired Data)\n")
cat(strrep("=", 60), "\n")
cat("Null hypothesis: The median difference between paired samples is 0\n")
cat("Alternative: ConvKB-D ranks are less than Lin-BMA ranks (one-tailed)\n\n")

# One-tailed test: testing if convkb_d ranks < lin_bma ranks (lower is better)
signed_rank_result <- wilcox.test(all_convkb_d_ranks, all_lin_bma_ranks, 
                                   paired = TRUE, 
                                   alternative = "less",
                                   exact = FALSE)  # Use normal approximation for large samples

cat("Wilcoxon Signed-Rank Test Results:\n")
cat("  V-statistic:", signed_rank_result$statistic, "\n")
cat("  p-value:", format(signed_rank_result$p.value, scientific=TRUE), "\n")
cat("  Method:", signed_rank_result$method, "\n\n")

# Two-tailed for completeness
signed_rank_result_two <- wilcox.test(all_convkb_d_ranks, all_lin_bma_ranks, 
                                       paired = TRUE, 
                                       alternative = "two.sided",
                                       exact = FALSE)
cat("Two-tailed p-value:", format(signed_rank_result_two$p.value, scientific=TRUE), "\n\n")


# ============================================================
# 2. WILCOXON RANK-SUM TEST / Mann-Whitney U (for unpaired data)
# ============================================================
cat(strrep("=", 60), "\n")
cat("WILCOXON RANK-SUM TEST / MANN-WHITNEY U (Unpaired Data)\n")
cat(strrep("=", 60), "\n")
cat("Null hypothesis: The two samples come from the same distribution\n")
cat("Alternative: ConvKB-D ranks are less than Lin-BMA ranks (one-tailed)\n")
cat("NOTE: This assumes data is UNPAIRED (not appropriate if same diseases)\n\n")

# One-tailed test
rank_sum_result <- wilcox.test(all_convkb_d_ranks, all_lin_bma_ranks, 
                                paired = FALSE, 
                                alternative = "less",
                                exact = FALSE)

cat("Wilcoxon Rank-Sum Test Results:\n")
cat("  W-statistic:", rank_sum_result$statistic, "\n")
cat("  p-value:", format(rank_sum_result$p.value, scientific=TRUE), "\n")
cat("  Method:", rank_sum_result$method, "\n\n")

# Two-tailed for completeness
rank_sum_result_two <- wilcox.test(all_convkb_d_ranks, all_lin_bma_ranks, 
                                    paired = FALSE, 
                                    alternative = "two.sided",
                                    exact = FALSE)
cat("Two-tailed p-value:", format(rank_sum_result_two$p.value, scientific=TRUE), "\n\n")


# ============================================================
# 3. PAIRED T-TEST (parametric alternative)
# ============================================================
cat(strrep("=", 60), "\n")
cat("PAIRED T-TEST (Parametric Alternative)\n")
cat(strrep("=", 60), "\n")

t_result <- t.test(all_convkb_d_ranks, all_lin_bma_ranks, 
                   paired = TRUE, 
                   alternative = "less")

cat("Paired t-test Results:\n")
cat("  t-statistic:", t_result$statistic, "\n")
cat("  p-value:", format(t_result$p.value, scientific=TRUE), "\n")
cat("  Degrees of freedom:", t_result$parameter, "\n")
cat("  95% CI upper bound:", t_result$conf.int[2], "\n\n")


# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================
cat(strrep("=", 60), "\n")
cat("SUMMARY STATISTICS\n")
cat(strrep("=", 60), "\n")

cat("\nConvKB-D Ranks:\n")
cat("  Mean:", mean(all_convkb_d_ranks), "\n")
cat("  Median:", median(all_convkb_d_ranks), "\n")
cat("  SD:", sd(all_convkb_d_ranks), "\n")
cat("  Min:", min(all_convkb_d_ranks), "\n")
cat("  Max:", max(all_convkb_d_ranks), "\n")

cat("\nLin-BMA Ranks:\n")
cat("  Mean:", mean(all_lin_bma_ranks), "\n")
cat("  Median:", median(all_lin_bma_ranks), "\n")
cat("  SD:", sd(all_lin_bma_ranks), "\n")
cat("  Min:", min(all_lin_bma_ranks), "\n")
cat("  Max:", max(all_lin_bma_ranks), "\n")

cat("\nDifference (ConvKB-D - Lin-BMA):\n")
rank_diff <- all_convkb_d_ranks - all_lin_bma_ranks
cat("  Mean difference:", mean(rank_diff), "\n")
cat("  Median difference:", median(rank_diff), "\n")
cat("  SD of difference:", sd(rank_diff), "\n")


# ============================================================
# 5. INTERPRETATION GUIDE
# ============================================================
cat("\n")
cat(strrep("=", 60), "\n")
cat("INTERPRETATION GUIDE\n")
cat(strrep("=", 60), "\n")
cat("\nWhich test should you use?\n")
cat("- SIGNED-RANK TEST: Use this if the same diseases appear in both methods\n")
cat("  (i.e., paired/matched data). This is most likely your case.\n")
cat("\n- RANK-SUM TEST: Use this only if the diseases are completely different\n")
cat("  between methods (i.e., unpaired/independent samples).\n")
cat("\nLower ranks are better, so we test if ConvKB-D < Lin-BMA.\n")
cat("If p-value < 0.05, we reject the null hypothesis.\n")
