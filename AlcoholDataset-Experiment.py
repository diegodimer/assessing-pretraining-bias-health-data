from collections import defaultdict
from AlcoholDataset import AlcoholDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from experiment_utils import feature_importante, generate_pies, remove_instances, evaluate_train_and_test_sets, get_full_sets_graphs


def gen_graph_for_sets(h: AlcoholDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name, stratify_age=True)
    generate_pies(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name, stratify_age=True)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Box Plot for Metric Values')
    # pd.DataFrame(metrics_all).boxplot(ax=ax1, rot=45)
    # fig1.savefig("boxplot-metrics.png")
    plt.close('all')

    feature_importante(name, h)


def original_dataset():
    h = AlcoholDataset()
    print('==========Original Dataset===========')

    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "original-dataset")


def high_imbalance_gender():
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]

        new_x_train = remove_instances(
            new_x_train, [new_x_train['gender'] == 0, new_x_train['phq_diagnosis'] == 0], 0.99)
        new_x_train = remove_instances(
            new_x_train, [new_x_train['gender'] == 0, new_x_train['phq_diagnosis'] == 1], 0.91)
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance (Gender)==========")
    print("Remove 99% of women with negative output and 91% with positive output, respectively")
    h = AlcoholDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance-gender")


def high_imbalance_age():
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]
        h.stratify_age(new_x_train)
        # best_CI = -1
        # best_KL = -1
        # best_KS = -1
        # best_CDDL = -1
        # best_ci_val = []
        # best_kl_val = []
        # best_ks_val = []
        # best_cddl_val = []
        # for k in [1,1]:
        #     for l in [1]:
        #         for i in np.arange(0.0, 1.0, 0.1):
        #             for j in np.arange(0.0, 1.0, 0.1):
        #             # test_x_train = remove_instances(new_x_train, [new_x_train['age>18'] == 0], i)
        #                 # test_x_train = remove_instances(new_x_train, [new_x_train['change_giveup'] == 1], i)
        #                 test_x_train = remove_instances(new_x_train, [new_x_train['age>18'] == k, new_x_train['phq_diagnosis'] == l], i)
        #                 test_x_train = remove_instances(test_x_train, [test_x_train['age>18'] == l, test_x_train['phq_diagnosis'] == k], j)
        #                 d = h.get_metrics(test_x_train, False)
        #                 if d['Class Imbalance (age>18)'] > best_CI:
        #                     best_CI = d['Class Imbalance (age>18)']
        #                     best_ci_val = [i,j, k, l]
        #                 if d['KL Divergence (age>18)'] > best_KL:
        #                     best_KL = d['KL Divergence (age>18)']
        #                     best_kl_val = [i,j, k, l]
        #                 if d['KS (age>18)'] > best_KS:
        #                     best_KS = d['KS (age>18)']
        #                     best_ks_val = [i,j, k, l]
        #                 if d['CDDL (age>18, change_giveup)'] > best_CDDL:
        #                     best_CDDL = d['CDDL (age>18, change_giveup)']
        #                     best_cddl_val = [i,j, k, l]
        new_x_train = remove_instances(
            new_x_train, [new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], 0.95)
        # new_x_train = remove_instances(new_x_train, [new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], 0.9)
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)
        new_x_train = new_x_train.drop('age>18', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance (Age)==========")
    print("Remove 95% of instances with age>18 and positive outcome")
    h = AlcoholDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance-age")


def equal_balance():
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[
            h.predicted_attr]
        h.stratify_age(new_x_train)
        # this is way more optimized in IntersectionalBiasDataset.py
        var11 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 0)])
        var12 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 0)])
        if var11 > var12:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 0], var11 - var12)
        elif var12 > var11:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 0], var12 - var11)

        var21 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 1)])
        var22 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 1)])
        if var21 > var22:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 1], var21 - var22)
        elif var22 > var21:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 1], var22 - var21)

        var31 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 0)])
        var32 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 0)])
        if var31 > var32:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 0], var31 - var32)
        elif var32 > var31:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 0], var32 - var31)

        var41 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 1)])
        var42 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 1)])
        if var41 > var42:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], var41 - var42)
        elif var42 > var41:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], var42 - var41)

        var51 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 0)])
        var12 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 0)])
        if var51 > var12:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 0], var51 - var12)
        elif var12 > var51:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 0], var12 - var51)

        var61 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 1)])
        var62 = len(new_x_train.loc[(new_x_train['gender'] == 1) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 1)])
        if var61 > var62:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], var61 - var62)
        elif var62 > var61:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 1, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 1], var62 - var61)

        var71 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 0)])
        var72 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 0)])
        if var71 > var72:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 0], var71 - var72)
        elif var72 > var71:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 0], var72 - var71)

        var81 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 1) & (new_x_train['phq_diagnosis'] == 1)])
        var82 = len(new_x_train.loc[(new_x_train['gender'] == 0) & (
            new_x_train['age>18'] == 0) & (new_x_train['phq_diagnosis'] == 1)])
        if var81 > var82:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 1, new_x_train['phq_diagnosis'] == 1], var81 - var82)
        elif var82 > var81:
            new_x_train = remove_instances(new_x_train, [
                                           new_x_train['gender'] == 0, new_x_train['age>18'] == 0, new_x_train['phq_diagnosis'] == 1], var82 - var81)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)
        new_x_train = new_x_train.drop('age>18', axis=1)

        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = AlcoholDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.stratify_age(h.dataset)
    h.gen_graph()
    h.dataset = h.dataset.drop('age>18', axis=1)
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "equal-balance")


all_acs = []
all_f1s = []
original_dataset()
high_imbalance_gender()
high_imbalance_age()
equal_balance()
for i in range(4):
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f} & {: >2.3f} ".format(
        all_acs[i], all_acs[i+4], all_acs[i+8], all_acs[i+12]), end="")
    print("\n")
print("\n====")
for i in range(4):
    print(" & {: >2.3f} & {: >2.3f} & {: >2.3f} & {: >2.3f} ".format(
        all_f1s[i], all_f1s[i+4], all_f1s[i+8], all_f1s[i+12]), end="")
    print("\n")
