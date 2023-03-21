from BaseDataset import BaseDataset
import pandas as pd
import numpy as np


class HeartDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset if dataset is not None else pd.read_csv(
            "datasets/heart.csv").drop_duplicates()
        self.predicted_attr = "target"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.n_neighbors = 42
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.protected_attr = ['sex']
        self.num_repetitions = 10

    def get_metrics(self, df_train):
        d = self.evaluate_metrics('sex', 1, 'cp', df_train)
        d.update(self.evaluate_metrics('sex', 1, 'thal', df_train, True))
        return d
