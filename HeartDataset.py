from BaseDataset import BaseDataset
import pandas as pd

class HeartDataset(BaseDataset):
    def run(self):
        dataset = pd.read_csv("datasets/heart.csv")
        predicted_attr = "target"
        return super()._run(dataset, predicted_attr)

h = HeartDataset()
h.run()