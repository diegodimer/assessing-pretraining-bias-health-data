from BaseDataset import BaseDataset
import pandas as pd


class HeartDataset(BaseDataset):
    def __init__(self):
        self.dataset = pd.read_csv("datasets/heart.csv").drop_duplicates()
        self.predicted_attr = "target"
        self.max_iter = 1000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.n_neighbors = 42
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.protected_attr = ['sex']
        super().__init__()

    def run(self):
        return super()._run()

    # def result_checker(self):
    #     df_out = self.X_test.reset_index()
    #     y_hats  = pd.DataFrame(self.model_predicted)
    #     df_out["Actual"] = self.y_test.reset_index()['target']
    #     df_out["Prediction"] = y_hats.reset_index()[0]

    #     self.gen_graph('sex', dataset = df_out, predicted_attr = 'Actual', labels_labels=['Female', 'Male'], file_name="heart-analysis/testSet_gender")

    #     df_err = df_out.loc[ df_out['Actual'] != df_out['Prediction']]
    #     self.gen_graph('sex', dataset = df_err, predicted_attr = 'Prediction', labels_labels=['Female', 'Male'], file_name="heart-analysis/errors_gender")
    #     df_corr = df_out.loc[ df_out['Actual'] == df_out['Prediction']]
    #     self.gen_graph('sex', dataset = df_corr, predicted_attr = 'Prediction', labels_labels=['Female', 'Male'], file_name="heart-analysis/acerts_gender")

    def get_metrics(self):
        df_train = self.X_train.reset_index()
        df_train[self.predicted_attr] = self.y_train.reset_index()[
            self.predicted_attr]
        h.evaluate_metrics('sex', 1, 'cp', df_train)
        h.evaluate_metrics('sex', 1, 'thal', df_train, True)


h = HeartDataset()
h.run()
# h.result_checker(labels_labels = ['Female', 'Male'])
h.get_metrics()
