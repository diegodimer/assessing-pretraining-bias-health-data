from BaseDataset import BaseDataset
import pandas as pd


class IntersectionalBiasDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset if dataset is not None else self.custom_preprocessing(
            pd.read_csv("datasets/intersectional-bias.csv"))
        self.predicted_attr = "Diagnosis"
        self.max_iter = 1000
        self.n_estimators = 20
        self.random_state = 12
        self.max_depth = 10
        self.n_neighbors = 318
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.protected_attr = ['Sex', 'Race']
        self.num_repetitions = 5

    def custom_preprocessing(self, df):
        def discretize_sex(x):
            if x == 'Female':
                return 0
            elif x == 'Male':
                return 1
            else:
                raise

        def discretize_race(x):
            if x == 'Black':
                return 0
            elif x == 'White':
                return 1
            elif x == 'Hispanic':
                return 2
            elif x == 'Asian':
                return 3
            else:
                raise

        def discretize_housing(x):
            if x == 'Stable':
                return 0
            elif x == 'Unstable':
                return 1
            else:
                raise

        def discretize_delay(x):
            if x == 'No':
                return 0
            elif x == 'Yes':
                return 1
            else:
                raise

        df['Sex'] = df['Sex'].apply(lambda x: discretize_sex(x))
        df['Race'] = df['Race'].apply(lambda x: discretize_race(x))
        df['Housing'] = df['Housing'].apply(lambda x: discretize_housing(x))
        df['Delay'] = df['Delay'].apply(lambda x: discretize_delay(x))

        return df

    def get_metrics(self, df_train):
        d = self.evaluate_metrics('Sex', 1, 'Rumination', df_train)
        d.update(self.evaluate_metrics('Sex', 1, 'Tension', df_train, True))
        d.update(self.evaluate_metrics('Race', 1, 'Rumination', df_train, True))
        d.update(self.evaluate_metrics('Race', 1, 'Tension', df_train, True))
        return d