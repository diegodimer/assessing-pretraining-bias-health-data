from BaseDataset import BaseDataset
import pandas as pd

class AlcoholDataset(BaseDataset):
    def __init__(self):
        self.dataset = self.custom_preprocessing(pd.read_csv("datasets/banco_flavia.csv"))
        self.predicted_attr = "phq_diagnosis"
        self.max_iter = 1000
        self.n_estimators = 40
        self.random_state = 12
        self.max_depth = 20
        self.n_neighbors = 296
        self.criterion = 'entropy'
        super().__init__()
        
    def run(self):
        return super()._run()

    def custom_preprocessing(self, df):

        # 1 -> Feminino
        # 2 -> Masculino
        def risk_labels(risk, gender):
            if risk <= 2 and gender == 1 or risk <= 3 and gender == 2:
                return 0
            else:
                return 1

        df['Risk_Stratification'] = df.apply(lambda x: risk_labels(
            x.Risk_Stratification, x.gender), axis=1)
    
        df = df.drop('year', axis=1) 
        df = df.drop('gender_id', axis=1) 
        df = df.drop('alcohol_dose', axis=1)
        df = df.drop('alcohol_binge', axis=1)
        df = df.drop('Audit_Total', axis=1)
        # df = df.drop('Risk_Stratification', axis=1)
        df = df.drop('suicidal_attempt', axis=1)
        df = df.drop('phq_1', axis=1)
        df = df.drop('phq_2', axis=1)
        df = df.drop('phq_total', axis=1)
        df = df.drop('suicide_ideation_month', axis=1)
        df = df.drop('bullying_yes', axis=1)
        df = df.drop('family_income', axis=1)
        df = df.drop_duplicates()
    
        return df

h = AlcoholDataset()
h.run()