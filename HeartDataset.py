from BaseDataset import BaseDataset
import pandas as pd
# import matplotlib.pyplot as plt

class HeartDataset(BaseDataset):
    def __init__(self):
        self.dataset = pd.read_csv("datasets/heart.csv").drop_duplicates()
        self.predicted_attr = "target"
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

    # def categorize(self,protected_attr):
    #     labels = ["female", "male"]
    #     outcomes = self.dataset[self.predicted_attr].unique().tolist()
    #     negative = []
    #     positive = []
   
    #     negative.append( len(self.dataset[ (self.dataset[self.predicted_attr] == 0) & (self.dataset[protected_attr] == 0) ]))
    #     negative.append( len(self.dataset[ (self.dataset[self.predicted_attr] == 0) & (self.dataset[protected_attr] == 1) ]))
    #     positive.append( len(self.dataset[ (self.dataset[self.predicted_attr] == 1) & (self.dataset[protected_attr] == 0) ]))
    #     positive.append( len(self.dataset[ (self.dataset[self.predicted_attr] == 1) & (self.dataset[protected_attr] == 1) ]))


    #     width = 0.35       # the width of the bars: can also be len(x) sequence
    #     fig, ax = plt.subplots()

    #     ax.bar(labels, negative, width, label=f'0')
    #     ax.bar(labels, positive, width, label=f'1', bottom=negative)
    #     for bars in ax.containers:
    #         ax.bar_label(bars)
    #     ax.set_ylabel('Count')
    #     # ax.set_title('Gender x Target')
    #     ax.legend()
    #     fig.savefig('gender_target_heart.png')

    # def categorize2(self,protected_attr):
    #     labels = self.dataset[protected_attr].unique().tolist()
    #     outcomes = self.dataset[self.predicted_attr].unique().tolist()
    #     outcomes.sort()
    #     bar_ind = []
    #     bar_list = []
    #     for i in outcomes:
    #         for j in labels:
    #             bar_ind.append(len(self.dataset[ (self.dataset[self.predicted_attr] == i) & (self.dataset[protected_attr] == j) ]))
    #         bar_list.append(bar_ind)
    #         bar_ind = []

    #     width = 0.35       # the width of the bars: can also be len(x) sequence
    #     fig, ax = plt.subplots()
    #     previous = []
    #     for i,j in enumerate(bar_list):
    #         if i == 0:
    #             ax.bar(labels, j, width, label=f'{i}')
    #         if i!=0:
    #             ax.bar(labels, j, width, label=f'{i}', bottom=previous)
    #         previous = j
    #     plt.figure(figsize=(20,12))

    #     ax.set_ylabel('count')
    #     ax.legend()
    #     # for bars in ax.containers: ## if the bars should have the values
    #     #     ax.bar_label(bars)
    #     fig.savefig('caterogizeage.png')
    


h = HeartDataset()
h.run()
print(h.evaluate_metrics('sex', 1, 0, 'age'))
h.gen_graph('sex', labels_labels = ["Female", "Male"] )