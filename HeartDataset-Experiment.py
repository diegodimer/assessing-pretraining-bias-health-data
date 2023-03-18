from HeartDataset import HeartDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from experiment_utils import feature_importante


def gen_graph_for_sets(h: HeartDataset, name: str):
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
    h.evaluate_metrics('sex', 1, 'cp', dataset=full_dataset_train)
    h.evaluate_metrics('sex', 1, 'thal', cddl_only=True,
                       dataset=full_dataset_train)

    # GENERATE GRAPHS FOR FULL TRAINING AND TEST DATA
    h.gen_graph('sex', df_type=f'{name}/FulltrainDataset',
                dataset=full_dataset_train, labels_labels=["Female", "Male"], graph_title="Complete train set")
    h.gen_graph('sex', df_type=f'{name}/FulltestDataset',
                dataset=full_dataset_test, labels_labels=["Female", "Male"], graph_title="Complete test set")

    # GRAPH FOR EACH OF THE MODELS TRAINED (THE MODEL'S PREDICTIONS)
    d = [dict() for x in range(4)]
    d_wrongs = [dict() for x in range(4)]
    d_male = [dict() for x in range(4)]
    d_female = [dict() for x in range(4)]
    d_correct = [dict() for x in range(4)]
    for idx, model in enumerate(h.models):
        d[idx]["Women predicted correctly"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
        d[idx]["Women false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d[idx]["Women false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d[idx]["Men predicted correctly"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
        d[idx]["Men false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d[idx]["Men false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d_wrongs[idx]["Women false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d_wrongs[idx]["Women false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d_wrongs[idx]["Men false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d_wrongs[idx]["Men false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d_female[idx]["Women predicted correctly"] = len(full_dataset_test.loc[(
            full_dataset_test['sex'] == 0) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
        d_female[idx]["Women false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d_female[idx]["Women false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 0) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d_male[idx]["Men predicted correctly"] = len(full_dataset_test.loc[(
            full_dataset_test['sex'] == 1) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
        d_male[idx]["Men false positive"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 0) & (full_dataset_test[model] == 1)])
        d_male[idx]["Men false negative"] = len(full_dataset_test.loc[(full_dataset_test['sex'] == 1) & (
            full_dataset_test[h.predicted_attr] == 1) & (full_dataset_test[model] == 0)])
        d_correct[idx]["Women predicted correctly"] = len(full_dataset_test.loc[(
            full_dataset_test['sex'] == 0) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])
        d_correct[idx]["Men predicted correctly"] = len(full_dataset_test.loc[(
            full_dataset_test['sex'] == 1) & (full_dataset_test[model] == full_dataset_test[h.predicted_attr])])

    generate_pie(h, name, full_dataset_test, d, "piechart")
    generate_pie(h, name, full_dataset_test, d_correct, "correct-pie")
    generate_pie(h, name, full_dataset_test, d_wrongs, "wrongs-pie")
    generate_pie(h, name, full_dataset_test, d_female, "female-pie")
    generate_pie(h, name, full_dataset_test, d_male, "male-pie")

    gs_test = gridspec.GridSpec(5, 2)
    gs_train = gridspec.GridSpec(5, 2)
    fig_testSets = plt.figure(figsize=(10, 20))
    fig_trainSets = plt.figure(figsize=(10, 20))
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

        ax = fig_testSets.add_subplot(gs_test[i])
        h.gen_graph(dataset=test_set, file_name=f"{name}/{i}-test",
                    graph_title=f"Test set #{i}", labels_labels=["Female", "Male"], ax=ax)

        ax2 = fig_trainSets.add_subplot(gs_train[i])
        h.gen_graph(dataset=train_set, file_name=f"{name}/{i}-test",
                    graph_title=f"Train set #{i}", labels_labels=["Female", "Male"], ax=ax2)

        # fig.savefig(f"{type(h).__name__}/{name}/{i}-train&test.png")

        # gs = gridspec.GridSpec(2, 2)
        # fig = plt.figure(figsize=(20,10))
        # ax = fig.add_subplot(gs[0])
        # h.gen_graph(dataset=test_set, graph_title=f"KNN #{i}", labels_labels=["Female", "Male"], ax=ax, predicted_attr='KNeighborsClassifier')
        # ax = fig.add_subplot(gs[1])
        # h.gen_graph(dataset=test_set, graph_title=f"Logistic Regression #{i}", labels_labels=["Female", "Male"], ax=ax, predicted_attr='LogisticRegression')
        # ax = fig.add_subplot(gs[2])
        # h.gen_graph(dataset=test_set, graph_title=f"Decision Tree #{i}", labels_labels=["Female", "Male"], ax=ax, predicted_attr='DecisionTreeClassifier')
        # ax = fig.add_subplot(gs[3])
        # h.gen_graph(dataset=test_set, graph_title=f"Random Forest#{i}", labels_labels=["Female", "Male"], ax=ax, predicted_attr='RandomForestClassifier')
        # fig.savefig(f"{type(h).__name__}/{name}/{i}-models.png")
        # plt.close(fig)

    fig_testSets.savefig(f"{type(h).__name__}/{name}/testSetsGrouped.png")
    fig_trainSets.savefig(f"{type(h).__name__}/{name}/trainSetsGrouped.png")
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
                                         new_x_train['cp'] == 0, new_x_train['target'] == 0, new_x_train['sex'] == 0], 0.8)
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
