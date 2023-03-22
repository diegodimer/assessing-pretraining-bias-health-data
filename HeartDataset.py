from BaseDataset import BaseDataset
import pandas as pd
import numpy as np


class HeartDataset(BaseDataset):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset if dataset is not None else pd.read_csv(
            "datasets/heart.csv").drop_duplicates()
        self.predicted_attr = "target"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.n_neighbors = 42
        self.criterion = 'entropy'
        self.positive_outcome = 0
        self.protected_attr = ['sex']
        self.num_repetitions = 10

    def get_metrics(self, df_train):
        d = self.evaluate_metrics('sex', 1, 'cp', df_train)
        d.update(self.evaluate_metrics('sex', 1, 'thal', df_train, True))
        return d

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
# x_min, x_max = self.X_test[['age','cp']].head(25).values[:, 0].min() , self.X_test[['age','cp']].head(25).values[:, 0].max()
# y_min, y_max = self.X_test[['age','cp']].head(25).values[:, 1].min() , self.X_test[['age','cp']].head(25).values[:, 1].max()
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                     np.arange(y_min, y_max, 0.02))
# model2 = KNeighborsClassifier(n_neighbors=1)
# model2.fit((self.X_train[['age','cp']]).head(30), self.y_train.head(30))
# Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # Plot also the training points
# # plt.scatter(self.X_train.values[:, 0], self.X_train.values[:, 1], c=self.y_train, cmap=cmap_bold,
# #             edgecolor='k', s=20)
# # plt.xlim(xx.min(), xx.max())
# # plt.ylim(yy.min(), yy.max())
# plt.title("KNN classification (k = %i, weights = '%s')" % (model2.n_neighbors, model2.weights))
# plt.xlabel("age")
# plt.ylabel("cp")
# plt.savefig("KNN.png", bbox_inches='tight')
