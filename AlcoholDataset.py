from BaseDataset import BaseDataset
import pandas as pd


class AlcoholDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset if dataset is not None else self.custom_preprocessing(
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
        self.protected_attr = ['gender', 'age>18']
        self.num_repetitions = 6
        self.protected_attr_mappings = {'gender': {"Women": 0, "Men": 1}, 'age>18': {
            "Students under 18 yo": 0, "Students over 18 yo": 1}}

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

        def gender(gender):
            return gender-1

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
        df['gender'] = df['gender'].apply(lambda x: gender(x))

        return df

    def get_metrics(self, df_train, print_metrics=True):
        self.stratify_age(df_train)
        d = self.evaluate_metrics(
            'gender', 1, 'change_giveup', df_train, print_metrics=print_metrics)
        # d.update(self.evaluate_metrics(
        #     'cotas', 0, 'change_giveup', df_train, print_metrics=print_metrics))
        d.update(self.evaluate_metrics(
            'age>18', 1, 'change_giveup', df_train, print_metrics=print_metrics))
        return d

    def stratify_age(self, df):
        df['age>18'] = (df['age'] > 18).astype(int)
