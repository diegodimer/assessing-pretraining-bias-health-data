from BaseDataset import BaseDataset
import pandas as pd

class AlcoholDataset(BaseDataset):
    def run(self):
        dataset = self.custom_preprocessing(pd.read_csv("datasets/banco_flavia.csv"))
        predicted_attr = "Risk_Stratification"
        return super()._run(dataset, predicted_attr)

    def custom_preprocessing(self, df):
        def age_group(x):
            if x <= 20:
                return 0 #'<=20'
            elif x > 20 and x <= 25:
                return 1 #'>20, <=25'
            elif x > 25 and x <= 30:
                return 2 # '>25, <=30'
            else:
                return 3 #'>30, <=35'

        # 1 -> Feminino
        # 2 -> Masculino
        def risk_labels(risk, gender):
            if risk <= 2 and gender == 1 or risk <= 3 and gender == 2:
                return 0
            else:
                return 1

        # df['age'] = df['age'].apply(lambda x: age_group(x))

        df['Risk_Stratification'] = df.apply(lambda x: risk_labels(
            x.Risk_Stratification, x.gender), axis=1)

        return df

h = AlcoholDataset()
h.run()