from pykeen.stoppers import Stopper
import torch as th
from evaluation import evaluate_model

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ValidationStopper(Stopper):
    def __init__(self,
                 model,
                 triples_factory,
                 file_identifier,
                 val_disease_genes,
                 gene2pheno,
                 disease2pheno,
                 eval_genes,
                 mode,
                 graph3,
                 graph4,
                 tolerance,
                 model_out_filename,
                 criterion,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.triples_factory = triples_factory
        self.file_identifier = file_identifier
        self.val_disease_genes = val_disease_genes
        self.gene2pheno = gene2pheno
        self.disease2pheno = disease2pheno
        self.eval_genes = eval_genes
        self.mode = mode
        self.graph3 = graph3
        self.graph4 = graph4
        self.tolerance = tolerance
        self.curr_tolerance = tolerance
        self.model_out_filename = model_out_filename
        self.best_val_mr = float('inf')
        self.criterion = criterion
        
    def get_summary_dict(self, *args, **kwargs):
        return dict()

    def should_stop(self, epoch):
        if self.curr_tolerance <= 0:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation MR for {self.tolerance} evaluations.")
            return True
        else:
            return False

    def should_evaluate(self, epoch):
        if epoch % 10 == 0:
            self.model.eval()
            val_output_prefix = f"data/results/validation_{self.file_identifier}"

            (val_inductive_macro_metrics,
             val_transductive_sim_macro_metrics,
             val_transductive_function_macro_metrics) = evaluate_model(
                model=self.model,
                test_disease_genes=self.val_disease_genes,
                gene2pheno=self.gene2pheno,
                disease2pheno=self.disease2pheno,
                eval_genes=self.eval_genes,
                triples_factory=self.triples_factory,
                mode=self.mode,
                graph3=self.graph3,
                graph4=self.graph4,
                 output_file_prefix=val_output_prefix,
                 criterion=self.criterion
            )

            # Choose validation metric based on graph mode
            if self.graph4 and self.mode == "transductive":
                val_mr = val_transductive_function_macro_metrics['mr']
                metric_type = "transductive_function"
                
            elif self.graph3 and self.mode == "transductive":
                val_mr = val_transductive_sim_macro_metrics['mr']
                metric_type = "transductive_similarity"
            else:
                val_mr = val_inductive_macro_metrics['mr']
                metric_type = "inductive"

            if val_mr < self.best_val_mr:
                self.best_val_mr = val_mr
                self.curr_tolerance = self.tolerance
                th.save(self.model.state_dict(), self.model_out_filename)
                logger.info(f"\nEpoch {epoch} - New best validation {metric_type} MR: {val_mr:.4f}. Model saved.")
            else:
                self.curr_tolerance -= 1

            logger.info(f"Epoch {epoch}, Val {metric_type} MR: {val_mr:.4f}, Best Val MR: {self.best_val_mr:.4f}, Tolerance left: {self.curr_tolerance}")

            return True
        else:
            return False
