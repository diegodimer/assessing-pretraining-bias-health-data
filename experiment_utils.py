

import math
import pandas as pd
from matplotlib import gridspec, pyplot as plt
import numpy as np


def feature_importante(name, h):
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)

    feature_names = list(h.X_train.columns)
    # forest.feature_importances_
    importances = np.mean(
        [tree.feature_importances_ for tree in h.estimators['RandomForestClassifier']], axis=0)
    std = np.std(
        [tree.feature_importances_ for tree in h.estimators['RandomForestClassifier']], axis=0)
    forest_importances = pd.DataFrame(importances, index=feature_names)
    forest_importances['std'] = std
    forest_importances = forest_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[0])
    forest_importances[0].plot.bar(ax=ax, yerr=forest_importances['std'])
    ax.set_title("Feature Importance RandomForestClassifier")

    feature_names = list(h.X_train.columns)
    importances = np.mean(
        [tree.feature_importances_ for tree in h.estimators['DecisionTreeClassifier']], axis=0)
    std = np.std(
        [tree.feature_importances_ for tree in h.estimators['DecisionTreeClassifier']], axis=0)
    forest_importances = pd.DataFrame(importances, index=feature_names)
    forest_importances['std'] = std
    forest_importances = forest_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[1])
    forest_importances[0].plot.bar(ax=ax, yerr=forest_importances['std'])
    ax.set_title("Feature Importance DecisionTreeClassifier")

    importances = np.mean([pow(math.e, w.coef_[0])
                          for w in h.estimators['LogisticRegression']], axis=0)  # pow(math.e, w)
    std = np.std([pow(math.e, w.coef_[0])
                 for w in h.estimators['LogisticRegression']], axis=0)
    logreg_importances = pd.DataFrame(importances, index=feature_names)
    logreg_importances['std'] = std
    logreg_importances = logreg_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[2])
    logreg_importances[0].plot.bar(ax=ax, yerr=std)
    ax.set_title("Feature Importance LogisticRegression")

    fig.savefig(f"{type(h).__name__}/{name}/importances.png",
                bbox_inches='tight')
    plt.close(fig)
