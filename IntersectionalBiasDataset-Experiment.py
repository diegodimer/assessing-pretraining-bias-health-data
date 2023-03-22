from collections import defaultdict
from IntersectionalBiasDataset import IntersectionalBiasDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from experiment_utils import feature_importante, generate_pies, remove_instances, evaluate_train_and_test_sets, get_full_sets_graphs


def gen_graph_for_sets(h: IntersectionalBiasDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_pies(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Box Plot for Metric Values')
    # pd.DataFrame(metrics_all).boxplot(ax=ax1, rot=45)
    # fig1.savefig("boxplot-metrics.png")
    plt.close('all')

    feature_importante(name, h)


def original_dataset():
    h = IntersectionalBiasDataset()
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
        # best_CI = -1
        # best_CI2 = -1
        # best_KL = -1
        # best_KL2 = -1
        # best_KS = -1
        # best_KS2 = -1
        # best_CDDL = -1
        # best_CDDL2 = -1
        # best_ci_val = []
        # best_ci2_val = []
        # best_kl_val = []
        # best_kl2_val = []
        # best_ks_val = []
        # best_ks2_val = []
        # best_cddl_val = []
        # best_cddl2_val = []
        # attribute_combinations = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
        # for attr in [0,1],[1,0]:
        #     for i in np.arange(0.0, 1.0, 0.1):
        #         for j in np.arange(0.0, 1.0, 0.1):
        #             for k in np.arange(0.0, 1.0, 0.1):
        #                 for l in np.arange(0.0, 1.0, 0.1):
        #                     # test_x_train = remove_instances(new_x_train, [new_x_train['age>18'] == 0], i)
        #                     # test_x_train = remove_instances(new_x_train, [new_x_train['change_giveup'] == 1], i)
        #                     test_x_train = remove_instances(new_x_train, [new_x_train['Race'] == 0, new_x_train['Diagnosis'] == attr[0]], i)
        #                     test_x_train = remove_instances(test_x_train,[test_x_train['Race'] == 0, test_x_train['Diagnosis'] == attr[1]], j)
        #                     test_x_train = remove_instances(test_x_train,[test_x_train['Diagnosis'] == attr[1], test_x_train['Sex'] == 0], k)
        #                     test_x_train = remove_instances(test_x_train,[test_x_train['Diagnosis'] == attr[1], test_x_train['Sex'] == 0], l)
        #                     d = h.get_metrics(test_x_train, False)
        #                     if d['Class Imbalance (Race)'] > best_CI:
        #                         best_CI = d['Class Imbalance (Race)']
        #                         best_ci_val = [i,j,k,l, attr]
        #                     if d['Class Imbalance (Sex)'] > best_CI2:
        #                         best_CI2 = d['Class Imbalance (Sex)']
        #                         best_ci2_val = [i,j,k,l, attr]
        #                     if d['KL Divergence (Race)'] > best_KL:
        #                         best_KL = d['KL Divergence (Race)']
        #                         best_kl_val = [i,j,k,l, attr]
        #                     if d['KL Divergence (Sex)'] > best_KL2:
        #                         best_KL2 = d['KL Divergence (Sex)']
        #                         best_kl2_val = [i,j,k,l, attr]
        #                     if d['KS (Race)'] > best_KS:
        #                         best_KS = d['KS (Race)']
        #                         best_ks_val = [i,j,k,l, attr]
        #                     if d['KS (Sex)'] > best_KS2:
        #                         best_KS2 = d['KS (Sex)']
        #                         best_ks2_val = [i,j,k,l, attr]
        #                     if d['CDDL (Race, Rumination)'] > best_CDDL:
        #                         best_CDDL = d['CDDL (Race, Rumination)']
        #                         best_cddl_val = [i,j,k,l, attr]
        #                     if d['CDDL (Sex, Rumination)'] > best_CDDL2:
        #                         best_CDDL2 = d['CDDL (Sex, Rumination)']
        #                         best_cddl2_val = [i,j,k,l, attr]
        # new_x_train = remove_instances(new_x_train, [new_x_train['Race'] == 1, new_x_train['Diagnosis'] == 1], 0.95)
        new_x_train = remove_instances(
            new_x_train, [new_x_train['Race'] == 0, new_x_train['Diagnosis'] == 0], 0.95)
        new_x_train = remove_instances(
            new_x_train, [new_x_train['Race'] == 0, new_x_train['Diagnosis'] == 1], 0.5)
        new_x_train = remove_instances(
            new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diagnosis'] == 0], 0.8)
        new_x_train = remove_instances(
            new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diagnosis'] == 1], 0.95)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print("Remove 95% of instances of non-white with negative output, 50% of non-white with positive output,  80% of women with negative output, 95% of women with positive output")
    h = IntersectionalBiasDataset()
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
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]

        attribute_combinations = np.array(np.meshgrid(
            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 6)

        for attr_comb in attribute_combinations:
            var11 = len(new_x_train.loc[(new_x_train['Sex'] == attr_comb[0]) & (
                new_x_train['Diagnosis'] == attr_comb[1]) & (new_x_train['Race'] == attr_comb[2])])
            var12 = len(new_x_train.loc[(new_x_train['Sex'] == attr_comb[3]) & (
                new_x_train['Diagnosis'] == attr_comb[4]) & (new_x_train['Race'] == attr_comb[5])])
            if var11 > var12:
                new_x_train = remove_instances(new_x_train, [
                                               new_x_train['Sex'] == attr_comb[0], new_x_train['Diagnosis'] == attr_comb[1], new_x_train['Race'] == attr_comb[2]], var11 - var12)
            elif var12 > var11:
                new_x_train = remove_instances(new_x_train, [
                                               new_x_train['Sex'] == attr_comb[3], new_x_train['Diagnosis'] == attr_comb[4], new_x_train['Race'] == attr_comb[5]], var12 - var11)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = IntersectionalBiasDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.gen_graph()
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
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f}  ".format(
        all_acs[i], all_acs[i+4], all_acs[i+8]), end="")
    print("\n")
print("\n====")
for i in range(4):
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f}  ".format(
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