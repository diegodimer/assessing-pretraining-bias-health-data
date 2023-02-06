from BaseDataset import BaseDataset
import pandas as pd
# import matplotlib.pyplot as plt

class IntersectionalBiasDataset(BaseDataset):
    def __init__(self):
        self.dataset = self.custom_preprocessing(pd.read_csv("datasets/intersectional-bias.csv"))
        self.predicted_attr = "Diagnosis"
        self.max_iter = 1000
        self.n_estimators = 20
        self.random_state = 12
        self.max_depth = 10
        self.n_neighbors = 39
        self.criterion = 'entropy'
        self.positive_outcome = 0
        super().__init__()

    def run(self):
        return super()._run()
    
    def custom_preprocessing(self, df):
        def discretize_sex(x):
            if x == 'Female':
                return 0
            else:
                return 1

        def discretize_race(x):
            if x == 'Black':
                return 0
            elif x == 'White':
                return 1
            elif x == 'Hispanic':
                return 2
            elif x == 'Asian':
                return 3
            else: raise
        
        def discretize_housing(x):
            if x == 'Stable':
                return 0
            elif x == 'Unstable':
                return 1
            else: raise
        
        def discretize_delay(x):
            if x == 'No':
                return 0
            elif x == 'Yes':
                return 1
            else: raise

        df['Sex'] = df['Sex'].apply(lambda x: discretize_sex(x))
        df['Race'] = df['Race'].apply(lambda x: discretize_race(x))
        df['Housing'] = df['Housing'].apply(lambda x: discretize_housing(x))
        df['Delay'] = df['Delay'].apply(lambda x: discretize_delay(x))

        return df

h = IntersectionalBiasDataset()
h.run()
print(h.evaluate_metrics('Sex', 1, 0, 'Race'))