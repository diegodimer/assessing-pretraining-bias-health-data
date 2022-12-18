from distutils.log import error
from typing import List
import pandas as pd
import numpy as np
from clarify_helper import pdfs_aligned_nonzero
from numpy import Infinity


class PreTrainingBias():

    def _class_imbalance(self, n_a, n_d):
        return (n_a - n_d) / (n_a + n_d)

    def _difference_in_positive_proportions_of_labels(self, q_a, q_d):
        return q_a - q_d

    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))

    def _divide(self, a, b) -> float:
        if b == 0 and a == 0:
            return 0.0
        if b == 0:
            if a < 0:
                return -Infinity
            return Infinity
        return a / b

    # def global_evaluation(self, df, target, correlated_attr_name, label_values_or_threshold):
    #     analysis = {"Class Imbalance": {}, "DPPL": {},
    #                 "KL Divergence": {}, "CDDL": {}}

    #     binary_attrs = []

    #     for i in df.columns:
    #         facet_counts = df[i].value_counts(sort=True)
    #         if (len(facet_counts) == 2) and i != target:
    #             binary_attrs.append(i)
    #             # print(f"{i} is a binary attribute. {facet_counts.index[0]} is the dominating with {facet_counts.values[0]} records over {facet_counts.index[1]} with {facet_counts.values[1]}")

    #             # class imbalance
    #             analysis["Class Imbalance"][i] = self._class_imbalance(
    #                 facet_counts.values[0], facet_counts.values[1])

    #             # Difference in Positive Proportions in Labels (DPPL)
    #             num_facet_and_pos_label = df[i].where(
    #                 df[target] == label_values_or_threshold).value_counts(sort=True)
    #             num_facet_and_pos_label_adv = num_facet_and_pos_label.values[0]
    #             num_facet_and_pos_label_disadv = num_facet_and_pos_label.values[1]
    #             num_facet_adv = facet_counts.values[0]
    #             num_facet_disadv = facet_counts.values[1]
    #             q_a = num_facet_and_pos_label_adv / num_facet_adv
    #             q_d = num_facet_and_pos_label_disadv / num_facet_disadv
    #             analysis["DPPL"][i] = self._difference_in_positive_proportions_of_labels(
    #                 q_a, q_d)

    #             # KL Divergence
    #             label = df[target]
    #             sensitive_facet_index = df[i] == facet_counts.index[1]
    #             (Pa, Pd) = pdfs_aligned_nonzero(
    #                 label[~sensitive_facet_index], label[sensitive_facet_index])
    #             analysis["KL Divergence"][i] = self._kl_divergence(Pa, Pd)

    #             # CDDL - Conditional Demographic Disparity in Labels
    #             feature = df[i]  # a variavel binaria que to analisando
    #             sensitive_facet_index = df[i] == facet_counts.index[1]
    #             positive_label_index = df[target] == label_values_or_threshold
    #             group_variable = df[correlated_attr_name]  # Age pro example
    #             analysis["CDDL"][i] = self._CDDL(
    #                 feature, sensitive_facet_index, positive_label_index, group_variable)

    #     return analysis

    def class_imbalance(self, df, label, threshold=None):
        facet_counts = df[label].value_counts(sort=True)
        if (len(facet_counts) == 2):
            return self._class_imbalance(facet_counts.values[0], facet_counts.values[1])
        else:  # is not a binary attr
            if threshold == None:
                raise Exception("Threshold not defined")
            a = len(df[df[label] > threshold])
            b = len(df[df[label] <= threshold])
            return self._class_imbalance(max(a, b), min(a, b))

    def class_imbalance_per_label(self, df, label, privileged_group, unprivileged_group) -> float:
        return self._class_imbalance((df[label].values == privileged_group).sum(), (df[label].values == unprivileged_group).sum())

    def KL_divergence(self, df, target, protected_attribute: str, privileged_group, unprivileged_group) -> float:
        label = df[target]
        P_list = list()
        sensitive_facet_index = df[protected_attribute] == unprivileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        P_list = pdfs_aligned_nonzero(label[unsensitive_facet_index], label[sensitive_facet_index])
        ks_val = 0
        for i, j in enumerate(P_list[0]): # j = 0,2 , i = 0
            ks_val += self._kl_divergence(j, P_list[1][i])
        return ks_val
    
    
    def KS(self, df, target, protected_attribute: str, privileged_group, unprivileged_group) -> float:
        label = df[target]
        P_list = list()
        sensitive_facet_index = df[protected_attribute] == unprivileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        P_list = pdfs_aligned_nonzero(label[unsensitive_facet_index], label[sensitive_facet_index])
        ks_val = 0
        for i, j in enumerate(P_list[0]): 
            ks_val = max(ks_val, abs(np.subtract(j,P_list[1][i])))
        return ks_val

    def CDDL(self, df: pd.DataFrame, target: str, positive_outcome, protected_attribute, unprivileged_group, group_variable) -> float:
        unique_groups = np.unique(df[group_variable])
        CDD = np.array([])
        counts = np.array([])
        for subgroup_variable in unique_groups:
            counts = np.append(counts, (df[group_variable].values == subgroup_variable).sum())
            numA = len (df[(df[target]==positive_outcome) & (df[protected_attribute]==unprivileged_group) & (df[group_variable] == subgroup_variable)]) 
            denomA = len (df[(df[target]==positive_outcome) & (df[group_variable] == subgroup_variable)])
            A = numA / denomA if denomA != 0 else 0
            numD =  len (df[(df[target]!=positive_outcome) & (df[protected_attribute]==unprivileged_group) & (df[group_variable] == subgroup_variable)])
            denomD =  len (df[(df[target]!=positive_outcome) & (df[group_variable] == subgroup_variable)])
            D = numD / denomD if denomD != 0 else 0
            CDD = np.append(CDD, D - A)
        return self._divide(np.sum(counts * CDD), np.sum(counts))

    
