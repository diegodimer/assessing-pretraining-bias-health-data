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
        super().__init__()

    def run(self):
        return super()._run()

    def result_checker(self):
        df_out = self.X_test.reset_index()
        y_hats  = pd.DataFrame(self.model_predicted)
        df_out["Actual"] = self.y_test.reset_index()['target']
        df_out["Prediction"] = y_hats.reset_index()[0]

        df_err = df_out.loc[ df_out['Actual'] != df_out['Prediction']]
        self.gen_graph('sex', dataset = df_err, predicted_attr = 'Prediction', labels_labels=['Female', 'Male'])

    


h = HeartDataset()
h.run()