from BaseDataset import BaseDataset
import pandas as pd


class AlcoholDataset(BaseDataset):
    def __init__(self):
        self.dataset = self.custom_preprocessing(
            pd.read_csv("datasets/banco_flavia.csv"))
        self.predicted_attr = "phq_diagnosis"
        self.max_iter = 1000
        self.n_estimators = 40
        self.random_state = 12
        self.max_depth = 20
        self.n_neighbors = 296
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.negative_outcome = 1
        self.protected_attr = ['gender', 'cotas']
        super().__init__()

    def run(self):
        return super()._run()

    def custom_preprocessing(self, df):

        # maps single, single in a relationship, widow, divorced -> single
        def marital_status(status):
            if status == 3:
                return 1
            else:  # single
                return 0

        def children(num):
            if num == 1:
                return 0
            elif num >= 2:
                return 1
            else:
                raise

        df = df.drop('year', axis=1)
        df = df.drop('gender_id', axis=1)
        df = df.drop('alcohol_dose', axis=1)
        df = df.drop('alcohol_binge', axis=1)
        df = df.drop('Audit_Total', axis=1)
        df = df.drop('Risk_Stratification', axis=1)
        df = df.drop('suicidal_attempt', axis=1)
        df = df.drop('suicide_attempt_dic', axis=1)
        df = df.drop('phq_1', axis=1)
        df = df.drop('phq_2', axis=1)
        df = df.drop('phq_total', axis=1)
        df = df.drop('suicide_ideation_month', axis=1)
        df = df.drop('bullying_yes', axis=1)
        df = df.drop('family_income', axis=1)
        df = df.drop_duplicates()
        df['marital_status'] = df['marital_status'].apply(
            lambda x: marital_status(x))
        df['children'] = df['children'].apply(lambda x: children(x))

        return df

    def get_metrics(self):
        df_train = self.X_train.reset_index()
        df_train[self.predicted_attr] = self.y_train.reset_index()[
            self.predicted_attr]
        h.evaluate_metrics('gender', 2, 'Sleep', df_train)
        h.evaluate_metrics('gender', 2, 'change_giveup', df_train, True)
        h.evaluate_metrics('cotas', 1, 'change_giveup', df_train, True)
        h.evaluate_metrics('cotas', 1, 'Sleep', df_train, True)


h = AlcoholDataset()
h.run()
