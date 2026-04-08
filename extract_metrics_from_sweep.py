import json
import traceback
import wandb

# Initialize the W&B API
api = wandb.Api()

# Specify your entity, project, and sweep ID
entity = "ferzcam"  # e.g., your username or team name
project = "indigena"
 


def get_mean_and_std(sweep_id, metrics):
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    metrics_summary = {metric: [] for metric in metrics}

    try:
        for run in sweep.runs:
            raw = run.summary._json_dict
            run_data = json.loads(raw) if isinstance(raw, str) else raw
            for metric in metrics:
                if metric not in run_data:
                    # print(f"  Run {run.id} ({run.name}): missing metric '{metric}'. Available keys: {list(run_data.keys())}")
                    continue
                value = run_data[metric]
                # if not isinstance(value, (int, float)):
                    # print(f"  Run {run.id} ({run.name}): metric '{metric}' has unexpected type {type(value).__name__!r}, value={value!r}")
                metrics_summary[metric].append(value)

        
        metrics_stats = {}
        for metric, values in metrics_summary.items():
            assert len(values) == 10, f"Expected 10 values for metric '{metric}', but got {len(values)}"
            mean_value = sum(values) / len(values)
            std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
            metrics_stats[metric] = {'mean': mean_value, 'std': std_value}

        return metrics_stats
    except Exception as e:
        print(f"An error occurred while processing sweep {sweep_id}: {e}")
        traceback.print_exc()
        return None

def print_as_tex(stats):
    string = ""
    for metric, stat in stats.items():
        string += f"{stat['mean']:.2f}\std{{{stat['std']:.2f}}} & "
    string = string[:-2] + " \\\\"
    print(string)
        
if __name__ == "__main__":

                                                                                                                  # }
    sweeps_folds_graph3_transductive_new = {"transe": "thsz3hh6", #"7t3rhx5e", #"blekpapj",
                                            "transh": "zz4w6yoz", #"d25ey4cd", #"yr6bv4lr",
                                            "transd": "7aczvp0w", #"mdorzx19", #"p050njbp",
                                            "convkb_transe": "6ypptcgh", #"qz8sxfx4", #"retnc0g5",
                                            "convkb_transd": "stid95xy", #"wvb1tmar", #"dh5i40rz"
                                            }
    
    sweeps_folds_graph4_transductive_new = {"transe": "6woovu3s", #"xlv3qc4e", #"8pyausld",
                                            "transh": "ispr5gjj", #"hgsf82b0", #"yunxk3ke",
                                            "transd": "xqmz88pd", #"4loalkm5", #"tx54b9mk",
                                            "convkb_transe": "x9q660ux", #"cl10tt3v", #"iin960jx",
                                            "convkb_transd": "od5fgq02", #"2zy4qaf3", #"j7cngwn8"
                                            }

    sweeps_folds_transd_inductive = {"graph1" : "q6fvsf9u", #"44v5ip5a",
                                     "graph2" : "bqqjdqlw", #"68eq1lh0",
                                     "graph3" : "ydvx17zz", #"zvse80iw",
                                     "graph4" : "pesnxxig", #"m1plxm3j",
                                     }

    sweeps_folds_convkb_transd_inductive = {"graph1" : "hc686p6h", #"ejf7kh70",
                                            "graph2" : "ob9xvasl", #"hpvvywyk",
                                            "graph3" : "i3eexoda", #"0lzb9egu",
                                            "graph4" : "sung7r49", #"j02r3b2q"
                                     }

    # sweeps = sweeps_folds_graph3_transductive_new
    # sweeps = sweeps_folds_graph4_transductive_new
    # sweeps = sweeps_folds_transd_inductive
    sweeps = sweeps_folds_convkb_transd_inductive
        
    
    for name, sweep_id in sweeps.items():
        # print(f"Metrics for {name} (Sweep ID: {sweep_id}):")
        # metrics_to_extract = ["mac_mr", "mac_mrr", "mac_hits@1", "mac_hits@3", "mac_hits@10", "mac_hits@100", "mac_auc"]
        
        # stats = get_mean_and_std(sweep_id, metrics_to_extract)
        # print_as_tex(stats)
        # for metric, stat in stats.items():
            # print(f"{metric}: Mean = {stat['mean']}, Std = {stat['std']}")

        # continue

        print(f"Inductive BMA metrics for {name} (Sweep ID: {sweep_id}):")
        # inductive_bma_metrics_to_extract = ["test_imac_bma_mr", "test_imac_bma_mrr", "test_imac_bma_hits@100", "test_imac_bma_auc"]
        inductive_bma_metrics_to_extract = ["test_imac_bma_mr", "test_imac_bma_mrr", "test_imac_bma_hits@1", "test_imac_bma_hits@3", "test_imac_bma_hits@10", "test_imac_bma_hits@100", "test_imac_bma_auc"]
        inductive_bma_stats = get_mean_and_std(sweep_id, inductive_bma_metrics_to_extract)
        print_as_tex(inductive_bma_stats)

        inductive_bmm_metrics_to_extract = ["test_imac_bmm_mr", "test_imac_bmm_mrr", "test_imac_bmm_hits@100", "test_imac_bmm_auc"]
        print(f"Inductive BMM metrics for {name} (Sweep ID: {sweep_id}):")
        inductive_bmm_stats = get_mean_and_std(sweep_id, inductive_bmm_metrics_to_extract)
        print_as_tex(inductive_bmm_stats)
        
        print(f"Transductive similarity metrics for {name} (Sweep ID: {sweep_id}):")
        transductive_sim_metrics_to_extract = ["test_sim_tmac_mr", "test_sim_tmac_mrr", "test_sim_tmac_hits@100", "test_sim_tmac_auc"]
        transductive_sim_stats = get_mean_and_std(sweep_id, transductive_sim_metrics_to_extract)
        if transductive_sim_stats:
            print_as_tex(transductive_sim_stats)
        else:
            print("No transductive similarity metrics found.")
            
        print(f"Transductive function metrics for {name} (Sweep ID: {sweep_id}):")
        transductive_func_metrics_to_extract = ["test_func_tmac_mr", "test_func_tmac_mrr", "test_func_tmac_hits@100", "test_func_tmac_auc"]
        transductive_func_stats = get_mean_and_std(sweep_id, transductive_func_metrics_to_extract)
        if transductive_func_stats:
            print_as_tex(transductive_func_stats)
        else:
            print("No transductive function metrics found.")

        print("----------------------------------------")
        if transductive_func_stats:
            print_as_tex(transductive_func_stats)
        if transductive_sim_stats:
            print_as_tex(transductive_sim_stats)
        print_as_tex(inductive_bmm_stats)
        print_as_tex(inductive_bma_stats)
        print("----------------------------------------")
