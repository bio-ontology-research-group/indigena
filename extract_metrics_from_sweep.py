import wandb

# Initialize the W&B API
api = wandb.Api()

# Specify your entity, project, and sweep ID
entity = "ferzcam"  # e.g., your username or team name
project = "indiga"
 


def get_mean_and_std(sweep_id, metrics):
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    metrics_summary = {metric: [] for metric in metrics}
    for run in sweep.runs:
        run_data = run.summary
        for metric in metrics:
            metrics_summary[metric].append(run_data[metric])

    metrics_stats = {}
    for metric, values in metrics_summary.items():
        mean_value = sum(values) / len(values)
        std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
        metrics_stats[metric] = {'mean': mean_value, 'std': std_value}

    return metrics_stats


def print_as_tex(stats):
    string = ""
    for metric, stat in stats.items():
        string += f"{stat['mean']:.2f} {{ \scriptsize $\\pm$ {stat['std']:.2f} }} & "
    string = string[:-2] + " \\\\"
    print(string)
        
if __name__ == "__main__":

    sweeps = {"graph1": "qc7xg7te","graph2": "0klscpge", "graph3": "a3ucqric", "graph4": "bavsfeki"}

    for name, sweep_id in sweeps.items():
        print(f"Metrics for {name} (Sweep ID: {sweep_id}):")
        metrics_to_extract = ["mac_mr", "mac_mrr", "mac_hits@1", "mac_hits@3", "mac_hits@10", "mac_hits@100", "mac_auc"]
        stats = get_mean_and_std(sweep_id, metrics_to_extract)
        print_as_tex(stats)
        for metric, stat in stats.items():
            print(f"{metric}: Mean = {stat['mean']}, Std = {stat['std']}")


            
