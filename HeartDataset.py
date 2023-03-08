from BaseDataset import BaseDataset
import pandas as pd
import numpy as np


class HeartDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset if dataset is not None else pd.read_csv(
            "datasets/heart.csv").drop_duplicates()
        self.predicted_attr = "target"
        self.max_iter = 1000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.n_neighbors = 42
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.protected_attr = ['sex']
        self.num_repetitions = 3

    def run(self):
        return super()._run()

    def get_metrics(self):
        df_train = self.X_train.reset_index()
        df_train[self.predicted_attr] = self.y_train.reset_index()[
            self.predicted_attr]
        h.evaluate_metrics('sex', 1, 'cp', df_train)
        h.evaluate_metrics('sex', 1, 'thal', df_train, True)


h = HeartDataset()
print('==========Before===========')
h.run()
h.gen_graph('sex', df_type='fullDataset')
h.evaluate_metrics('sex', 1, 'cp', h.dataset)
h.evaluate_metrics('sex', 1, 'thal', h.dataset, True)

print('==========After===========')
df = h.dataset.loc[h.dataset['sex'] == 0]
remove_n = 40
drop_indices = np.random.choice(df.index, remove_n, replace=False)
df_subset = h.dataset.drop(drop_indices)
h2 = HeartDataset(df_subset)
h2.run()

df_out = h2.x_train_list[0].reset_index()
y_hats = pd.DataFrame(h2.predicted_list[0])
df_out["Actual"] = h2.y_train_list[0].reset_index()[
    h2.predicted_attr]
df_out["Prediction"] = y_hats.reset_index()[0]

df_out2 = h2.x_test_list[0].reset_index()
y_hats = pd.DataFrame(h2.predicted_list[0])
df_out2["Actual"] = h2.y_test_list[0].reset_index()[
    h2.predicted_attr]
df_out2["Prediction"] = y_hats.reset_index()[0]


h2.gen_graph('sex', df_type='modifiedDatasetTRAIN',
             dataset=df_out, predicted_attr='Actual')
h2.gen_graph('sex', df_type='modifiedDatasetTEST',
             dataset=df_out2, predicted_attr='Actual')
h2.gen_graph('sex', df_type='modifiedDatasetFull', dataset=h2.dataset)
h2.evaluate_metrics('sex', 1, 'cp')
h2.evaluate_metrics('sex', 1, 'thal', cddl_only=True)
