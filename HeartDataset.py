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

    def perturbe(self, X_train, y_train):
        complete_x_train = X_train.reset_index()
        complete_x_train['target'] = y_train.reset_index()['target']
        
        new_x_train = self.remove_instances(complete_x_train, 0.35, 0)
        new_x_train = self.remove_instances(new_x_train, 0.7, 1)

        new_y = new_x_train['target']
        new_x_train =  new_x_train.drop(self.predicted_attr, axis=1)
        new_x_train =  new_x_train.drop('index', axis=1)

        return new_x_train, new_y

    def remove_instances(self, dataset, percentage, target):
        new_x = dataset.loc[ (dataset['sex'] == 0) & (dataset['target'] == target) ]
        drop_indices = np.random.choice(new_x.index, round(len(new_x)*percentage), replace=False)
        new_x = dataset.drop(drop_indices)
        return new_x

    def run(self):
        return super().run()

    def get_metrics(self):
        df_train = self.X_train.reset_index()
        df_train[self.predicted_attr] = self.y_train.reset_index()[
            self.predicted_attr]
        h.evaluate_metrics('sex', 1, 'cp', df_train)
        h.evaluate_metrics('sex', 1, 'thal', df_train, True)


h = HeartDataset()
print('==========Before===========')

h.execute_models()
h.gen_graph('sex', df_type='fullDataset')

df_train = h.x_train_list[0].reset_index()
df_train[h.predicted_attr] = h.y_train_list[0].reset_index()[h.predicted_attr]

df_test = h.x_test_list[0].reset_index()
df_test[h.predicted_attr] = h.y_test_list[0].reset_index()[h.predicted_attr]

h.gen_graph('sex', df_type='traindataset', dataset=df_train)
h.gen_graph('sex', df_type='testdataset', dataset=df_test)
h.evaluate_metrics('sex', 1, 'cp', h.dataset)
h.evaluate_metrics('sex', 1, 'thal', h.dataset, True)

print('==========After===========')

h2 = HeartDataset()
h2.dropper = True
h2.execute_models()

df_out = h2.x_train_list[0].reset_index()
df_out["Actual"] = h2.y_train_list[0].reset_index()[h2.predicted_attr]


df_out2 = h2.x_test_list[0].reset_index()
y_hats = pd.DataFrame(h2.predicted_list['DecisionTreeClassifier'][0])
df_out2["Actual"] = h2.y_test_list[0].reset_index()[h2.predicted_attr]
df_out2["Prediction"] = y_hats.reset_index()[0]


h2.gen_graph('sex', df_type='modifiedDatasetTRAIN',
             dataset=df_out, predicted_attr='Actual')

h2.gen_graph('sex', df_type='modifiedDatasetTEST',
             dataset=df_out2, predicted_attr='Actual')

h2.gen_graph('sex', df_type='modifiedDatasetTEST',
             dataset=df_out2, predicted_attr='Prediction')

full_dataset_train = h2.x_train_list[0]
full_dataset_train['target'] = h2.y_train_list[0]
full_dataset_test = h2.x_test_list[0]
full_dataset_test['target'] = h2.y_test_list[0]
full_dataset = pd.concat([full_dataset_train, full_dataset_test], ignore_index=True)
h2.evaluate_metrics('sex', 1, 'cp', dataset=full_dataset)
h2.evaluate_metrics('sex', 1, 'thal', cddl_only=True, dataset=full_dataset)
h2.gen_graph('sex', df_type='modifiedDatasetFull', dataset=full_dataset)
print(h2.model_conf_matrix['DecisionTreeClassifier'][0])
# mas qual é a proporção de acertos no valor protegido?
