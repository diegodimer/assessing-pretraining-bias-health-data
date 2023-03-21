

from collections import defaultdict
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
    plt.tight_layout()
    fig.savefig(f"{type(h).__name__}/{name}/importances.png".replace(">", ""),
                )
    plt.close(fig)


def generate_model_pies(h, name, model_dic, pie_name):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(18, 10))
    for idx, model in enumerate(h.models):
        # h.gen_graph(column, df_type=f'{name}/{model}', dataset=full_dataset_test, labels_labels=[
        #             "Female", "Male"], graph_title=f"{model}", predicted_attr=model)

        ax = fig.add_subplot(gs[idx])
        wedges, _ = ax.pie(list(model_dic[model].values()), autopct=None)
        ax.set_title(f"{model}")
        percents = []
        for i in list(model_dic[model].values()):
            percents.append(100.*i/sum(list(model_dic[model].values())))

        labels = ['{0} - {1:1.2f}% ({2})'.format(i, j, k) for i, j, k in zip(
            list(model_dic[model].keys()), percents, list(model_dic[model].values()))]
        ax.legend(wedges, labels,
                  title=f"{model}",
                  loc="center left",
                  bbox_to_anchor=(-0.7, 0, 0, 1))
    fig.savefig(f"{type(h).__name__}/{name}/{pie_name}.png".replace(">", ""))
    plt.close('all')


def remove_instances(x, conditions, value):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(value, new_x_size) if value >= 1 else min(round(new_x_size*value), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain

def generate_pies(h, name, full_dataset_test):
    d = defaultdict(lambda: defaultdict(dict))
    d_wrongs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    d_correct = defaultdict(lambda: defaultdict(dict))
    for attr in h.protected_attr_mappings.keys():
        for val in h.protected_attr_mappings[attr].keys():
            for model in h.models:
                d[attr][model][f"{val} predicted correctly"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
                d[attr][model][f"{val} false positive"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
                d[attr][model][f"{val} false negative"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
                d_wrongs[attr][val][model][f"{val} predicted correctly"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[h.predicted_attr] == full_dataset_test[model])])
                d_wrongs[attr][val][model][f"{val} false negative"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
                d_wrongs[attr][val][model][f"{val} false positive"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
                d_correct[attr][model][f"{val} predicted correctly"] = len(full_dataset_test.loc[(full_dataset_test[attr] == h.protected_attr_mappings[attr][val]) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
                
    for attr in d.keys():
            generate_model_pies(h, name, d[attr], f"piechart-complete-{attr}")
    
    for attr in d_wrongs.keys():
        for val in d_wrongs[attr].keys():
            generate_model_pies(h, name, d_wrongs[attr][val], f"piechart-wrong-{val}")
    
    for attr in d_correct.keys():
        generate_model_pies(h, name, d_correct[attr], f"piechart-correct-{attr}")


def evaluate_train_and_test_sets(h, name, stratify_age=False):
    metrics_all = defaultdict(list)
    gs_test = {attr: gridspec.GridSpec(round(h.num_repetitions/2), 2) for attr in h.protected_attr}
    gs_train = {attr: gridspec.GridSpec(round(h.num_repetitions/2), 2) for attr in h.protected_attr}
    fig_testSets = {attr: (plt.figure(figsize=(10, 20))) for attr in h.protected_attr}
    fig_trainSets = {attr: (plt.figure(figsize=(10, 20))) for attr in h.protected_attr}
    
    for attr in h.protected_attr:
        for i in range(h.num_repetitions):
            train_set = h.x_train_list[i].reset_index()
            train_set[h.predicted_attr] = h.y_train_list[i].reset_index()[
                h.predicted_attr]
            test_set = h.x_test_list[i].reset_index()
            test_set[h.predicted_attr] = h.y_test_list[i].reset_index()[
                h.predicted_attr]
            for model in h.models:
                y_hats = pd.DataFrame(h.predicted_list[model][i])
                test_set[model] = y_hats.reset_index()[0]
            # gs = gridspec.GridSpec(1, 2)
            # fig = plt.figure(figsize=(12,5))
            # ax = fig.add_subplot(gs[0])
            # h.gen_graph(dataset=train_set, file_name=f"{name}/{i}-train", graph_title=f"Train set #{i}", labels_labels=["Female", "Male"], ax=ax)
            # ax = fig.add_subplot(gs[1])
            # h.gen_graph(dataset=test_set, file_name=f"{name}/{i}-test", graph_title=f"Test set #{i}", labels_labels=["Female", "Male"], ax=ax)
            f = h.get_metrics(train_set, print_metrics=False)
            for t in f.keys():
                metrics_all[t].append(f[t])
            if stratify_age:
                h.stratify_age(test_set)
            
            ax = fig_testSets[attr].add_subplot(gs_test[attr][i])
            h.gen_graph(dataset=test_set, file_name=f"{name}/{i}-test",
                        graph_title=f"Test set #{i}", labels_labels=h.protected_attr_mappings[attr].keys(), ax=ax, protected_attr=attr)

            ax2 = fig_trainSets[attr].add_subplot(gs_train[attr][i])
            h.gen_graph(dataset=train_set, file_name=f"{name}/{i}-test",
                        graph_title=f"Train set #{i}", labels_labels=h.protected_attr_mappings[attr].keys(), ax=ax2, protected_attr=attr)

        fig_testSets[attr].savefig(f"{type(h).__name__}/{name}/testSetsGrouped-{attr}.png".replace(">", ""))
        fig_trainSets[attr].savefig(f"{type(h).__name__}/{name}/trainSetsGrouped-{attr}.png".replace(">", ""))
    return metrics_all


def get_full_sets_graphs(h, name, stratify_age=False):
    full_dataset_train = pd.concat(
        [h.x_train_list[i].reset_index() for i in range(h.num_repetitions)])
    full_dataset_train[h.predicted_attr] = pd.concat(
        [h.y_train_list[i].reset_index()[h.predicted_attr] for i in range(h.num_repetitions)])
    full_dataset_train.drop("index", axis=1)
    plt.rcParams["figure.autolayout"] = True
    full_dataset_test = pd.concat(
        [h.x_test_list[i].reset_index() for i in range(h.num_repetitions)])
    full_dataset_test[h.predicted_attr] = pd.concat(
        [h.y_test_list[i].reset_index()[h.predicted_attr] for i in range(h.num_repetitions)])
    full_dataset_test.drop("index", axis=1)

    # ADD MODELS PREDICTIONS TO FULL DATASET TRAIN
    for model in h.models:
        df_hats = pd.concat([pd.DataFrame(h.predicted_list[model][i]).reset_index()[
                            0] for i in range(h.num_repetitions)])
        full_dataset_test[model] = df_hats

    # PRINT METRICS (IT IS OVER ALL TRAINING DATA)
    print(
        f"\nMetrics calculated over the train datasets (concatenanted from all {h.num_repetitions} repetitions)")
    h.get_metrics(full_dataset_train)
    if(stratify_age):
        h.stratify_age(full_dataset_test)

    
    fig, ax = plt.subplots()
    total_size = len(full_dataset_test)
    right = []
    wrong = []
    labels = list(h.models)
    for p in labels:
        right.append((len(full_dataset_test.loc[(full_dataset_test[p] == full_dataset_test[h.predicted_attr])])/total_size) * 100.0)
        wrong.append((len(full_dataset_test.loc[(full_dataset_test[p] != full_dataset_test[h.predicted_attr])])/total_size) * 100.0)
    
    df = pd.DataFrame({'correct': right, 'wrong': wrong}, index=labels)
    ax = df.plot.barh(ax=ax, color=['green', 'red'])
    ax.legend()
    for bars in ax.containers:  # if the bars should have the values
        plt.bar_label(bars, labels=[f'{x:,.2f}%' for x in bars.datavalues], label_type='center')
    fig.savefig(f"{type(h).__name__}/{name}/FullTest-Predictions.png".replace(">", ""))
    plt.close(fig)

    # GENERATE GRAPHS FOR FULL TRAINING AND TEST DATA
    for attr in h.protected_attr:
        h.gen_graph(attr, df_type=f'{name}/FulltrainDataset-{attr}',
                    dataset=full_dataset_train, labels_labels=list(h.protected_attr_mappings[attr].keys()), graph_title="Complete train set")
        h.gen_graph(attr, df_type=f'{name}/FulltestDataset-{attr}',
                    dataset=full_dataset_test, labels_labels=list(h.protected_attr_mappings[attr].keys()), graph_title="Complete test set")
                    
    return full_dataset_test
