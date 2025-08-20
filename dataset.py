from mowl.datasets import PathDataset, OWLClasses
from mowl.owlapi import OWLAPIAdapter
import os
from util import load_gene_phenotypes, load_disease_phenotypes

class GDADataset(PathDataset):

    def __init__(self, fold=0):
        self.root = os.path.abspath("data")
        train_path = os.path.join(self.root, "upheno.owl")
        super(GDADataset, self).__init__(train_path)
        
        self.fold_path = os.path.join(self.root, f"gene_disease_folds/fold_{fold}")
        self._gene_phenotypes = None
        self._disease_phenotypes = None

        
        

    def _load_data(self):
        classes = set(self.classes.as_str)
        existing_mp_phenotypes = set()
        existing_hp_phenotypes = set()
        for cls in classes:
            if "MP_" in cls:
                existing_mp_phenotypes.add(cls)
            elif "HP_" in cls:
                existing_hp_phenotypes.add(cls)

        
        self._gene_phenotypes = load_gene_phenotypes(os.path.join(self.root, "MGI_GenePheno.rpt"), existing_mp_phenotypes)
        self._disease_phenotypes = load_disease_phenotypes(os.path.join(self.root, "phenotype.hpoa"), existing_hp_phenotypes)


    @property
    def gene_phenotypes(self):
        if self._gene_phenotypes is None:
            self._load_data()
        return self._gene_phenotypes

    @property
    def disease_phenotypes(self):
        if self._disease_phenotypes is None:
            self._load_data()
        return self._disease_phenotypes
        
    @property
    def evaluation_classes(self):
        adapter = OWLAPIAdapter()
        if self._evaluation_classes is None:

            genes = set([gene for gene, _ in self.gene_phenotypes])
            diseases = set([disease for _, disease in self.disease_phenotypes])

            genes = set([adapter.create_class(gene) for gene in genes])
            diseases = set([adapter.create_class(disease) for disease in diseases])
                        
            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"


    def get_fold(self, fold):
        train_path = os.path.join(self.fold_path, f"train.csv")
        test_path = os.path.join(self.fold_path, f"test.csv")

        train_df = pd.read_csv(train_path,sep=',')
        test_df = pd.read_csv(test_path,sep=',')

        train_pairs = [(row['Gene'], row['Disease']) for _, row in train_df.iterrows()]
        test_pairs = [(row['Gene'], row['Disease']) for _, row in test_df.iterrows()]

        return train_pairs, test_pairs
        
        
