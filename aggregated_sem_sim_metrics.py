import sys
import click as ck
from evaluate_sem_sim import compute_metrics

def print_as_tex(stats):
    string = ""
    for metric, stat in stats.items():
        string += f"{stat['mean']:.2f} {{ \scriptsize $\\pm$ {stat['std']:.2f} }} & "
    string = string[:-2] + " \\\\"
    print(string)

@ck.command()
@ck.option('--root_dir', default='data/baseline_results', help='Root directory containing result files.')
@ck.option('--pairwise_measure', '-pw', type=ck.Choice(['resnik', 'lin']), default='resnik', help='Pairwise semantic similarity measure.')
@ck.option('--groupwise_measure', '-gw', type=ck.Choice(['bma', 'bmm', 'simgic']), default='bma', help='Groupwise semantic similarity measure.')
def main(root_dir, pairwise_measure, groupwise_measure):
    
    metrics_to_extract = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auc"]

    if groupwise_measure == "simgic":
        pairwise_measure = ""
    

    metrics_summary = dict()
    for fold in range(10):
        input_file = f"{root_dir}/resnik_{pairwise_measure}_{groupwise_measure}_fold{fold}_results.txt"
        input_file = input_file.replace("__", "_")
        _, macro_metrics = compute_metrics(input_file)
        for metric in metrics_to_extract:
            if metric not in metrics_summary:
                metrics_summary[metric] = []
            metrics_summary[metric].append(macro_metrics[metric])

    metrics_stats = {}
    for metric, values in metrics_summary.items():
        mean_value = sum(values) / len(values)
        std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
        metrics_stats[metric] = {'mean': mean_value, 'std': std_value}



    print_as_tex(metrics_stats)
    

    
if __name__ == "__main__":
    main()

    
