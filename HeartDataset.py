from BaseDataset import BaseDataset
import pandas as pd
import matplotlib.pyplot as plt

class HeartDataset(BaseDataset):
    def __init__(self):
        self.dataset = pd.read_csv("datasets/heart.csv")
        self.predicted_attr = "target"
        self.max_iter = 1000
        self.n_estimators = 20
        self.random_state = 12
        self.max_depth = 10
        self.n_neighbors = 39
        self.criterion = 'entropy'
        super().__init__()

    def run(self):
        return super()._run()

    def categorize(self,protected_attr):
        labels = self.dataset[protected_attr].unique().tolist()
        outcomes = self.dataset[self.predicted_attr].unique().tolist()
        bar_ind = []
        bar_list = []
        for i in labels:
            for j in outcomes:
                bar_ind.append(len(self.dataset[ (self.dataset[self.predicted_attr] == j) & (self.dataset[protected_attr] == i) ]))
            bar_list.append(bar_ind)
            bar_ind = []

        width = 0.35       # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        for i,j in enumerate(bar_list):
            ax.bar(labels, j, width, label=f'{i}')

        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.legend()
        fig.savefig('caterogize.png')

h = HeartDataset()
h.categorize('sex')