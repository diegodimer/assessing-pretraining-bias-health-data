from collections import defaultdict
from HeartDataset import HeartDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from experiment_utils import evaluate_train_and_test_sets, feature_importante, generate_pies, get_full_sets_graphs


def gen_graph_for_sets(h: HeartDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_pies(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Box Plot for Metric Values')
    # pd.DataFrame(metrics_all).boxplot(ax=ax1, rot=45)
    # fig1.savefig("boxplot-metrics.png")
    plt.close('all')

    feature_importante(name, h)

def generate_pie(h, name, full_dataset_test, model_dic, pie_name):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(18, 10))
    for idx, model in enumerate(h.models):
        h.gen_graph('sex', df_type=f'{name}/{model}', dataset=full_dataset_test, labels_labels=[
                    "Female", "Male"], graph_title=f"{model}", predicted_attr=model)

        ax = fig.add_subplot(gs[idx])
        wedges, _ = ax.pie(list(model_dic[idx].values()), autopct=None)
        ax.set_title(f"{model}")
        percents = []
        for i in list(model_dic[idx].values()):
            percents.append(100.*i/sum(list(model_dic[idx].values())))

        labels = ['{0} - {1:1.2f}% ({2})'.format(i, j, k) for i, j, k in zip(
            list(model_dic[idx].keys()), percents, list(model_dic[idx].values()))]
        ax.legend(wedges, labels,
                  title=f"{model}",
                  loc="center left",
                  bbox_to_anchor=(-0.7, 0, 0, 1))
    fig.savefig(f"{type(h).__name__}/{name}/{pie_name}.png")
    plt.close('all')


def remove_instances(x, target, value, sex=0):
    new_x = x.loc[(x['sex'] == sex) & (x['target'] == target)]
    drop_indices = np.random.choice(
        new_x.index, value if value >= 1 else round(len(new_x)*value), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def remove_instances_2(x, conditions, value):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(value, new_x_size) if value >= 1 else min(round(new_x_size*value), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def original_dataset():
    h = HeartDataset()
    print('==========Original Dataset===========')

    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "original-dataset")


def high_imbalance():
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]
        # tirar também: mulheres com target 1 e thal 0, mulheres com 0 e thal 3, mulheres com target 1 e cp 2, mulheres com target 1 e cp 1, mulheres com target 0 e cp 0
        # new_x_train = remove_instances_2(new_x_train, [ new_x_train['thal'] == 2, new_x_train['target'] == 1, new_x_train['sex'] == 0], 0.1)
        new_x_train = remove_instances_2(new_x_train, [
                                         new_x_train['thal'] == 2, new_x_train['target'] == 0, new_x_train['sex'] == 0], 0.85)
        new_x_train = remove_instances_2(new_x_train, [
                                         new_x_train['thal'] == 3, new_x_train['target'] == 0, new_x_train['sex'] == 0], 0.80)
        # new_x_train = remove_instances_2(new_x_train, [ new_x_train['thal'] == 3, new_x_train['target'] == 1, new_x_train['sex'] == 0], 0.2)
        # new_x_train = remove_instances_2(new_x_train, [ new_x_train['cp'] == 2, new_x_train['target'] == 1, new_x_train['sex'] == 0], 0.1)
        new_x_train = remove_instances_2(new_x_train, [
                                         new_x_train['cp'] == 2, new_x_train['target'] == 0, new_x_train['sex'] == 0], 0.80)
        # new_x_train = remove_instances_2(new_x_train, [ new_x_train['cp'] == 0, new_x_train['target'] == 1, new_x_train['sex'] == 0], 0.2)
        new_x_train = remove_instances_2(new_x_train, [
                                         new_x_train['cp'] == 0, new_x_train['target'] == 0, new_x_train['sex'] == 0], 0.80)
        # new_x_train = remove_instances(new_x_train, 0, 0.7)
        new_x_train = remove_instances(new_x_train, 1, 0.2)
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print("Remove 90%/ of women with negative output and 30% with positive output, respectively")
    h = HeartDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    def perturbe(X_train, y_train):
        complete_x_train = X_train.reset_index()
        complete_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]

        positive_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train[h.predicted_attr] == 1)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train[h.predicted_attr] == 1)])
        negative_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train[h.predicted_attr] == 0)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train[h.predicted_attr] == 0)])

        complete_x_train = remove_instances(
            complete_x_train, 1, positive_out_in_train, sex=1)
        new_x_train = remove_instances(
            complete_x_train, 0, negative_out_in_train, sex=1)
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = HeartDataset()
    h.dropper = True
    h.gen_graph()
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "equal-balance")


all_acs = []
all_f1s = []
original_dataset()
high_imbalance()
equal_balance()
for i in range(4):
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f} ".format(
        all_acs[i], all_acs[i+4], all_acs[i+8]), end="")
    print("\n")
print("\n====")
for i in range(4):
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f} ".format(
        all_f1s[i], all_f1s[i+4], all_f1s[i+8]), end="")
    print("\n")

print("avg acc (all models)")
for i in range(3):
    idx = i*4
    print(round(sum(all_acs[idx:idx+4])/4, 3))

print("avg f1 (all models)")
for i in range(3):
    idx = i*4
    print(round(sum(all_f1s[idx:idx+4])/4, 3))