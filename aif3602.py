import sys
sys.path.append("../")
# Metrics function
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
## sudo apt-get install python3-tk

import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.preprocessing.lfr import LFR
from alcohol_dataset import AlcoholDataset, load_preproc_data_alcohol
from sklearn.linear_model import LogisticRegression

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics


# df.loc[df['Sex'] == 1]
def main():
    dataset_orig = load_preproc_data_alcohol()
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

##
    privileged_groups = [{'cotas': 0}]
    unprivileged_groups = [{'cotas': 1}]  
    # Metric used (should be one of allowed_metrics)
    metric_name = "Statistical parity difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05

    #random seed for calibrated equal odds prediction
    np.random.seed(1)

    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # print out some labels, names, etc.
    print("#### Training Dataset shape")
    print(dataset_orig_train.features.shape)
    print("#### Favorable and unfavorable labels")
    print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
    print("#### Protected attribute names")
    print(dataset_orig_train.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(dataset_orig_train.privileged_protected_attributes, 
        dataset_orig_train.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(dataset_orig_train.feature_names)

    ## Metric for original training data

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("#### Original training dataset")
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

  
    ## Train classifier on original data

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred


    ## Obtain scores for validation and test sets

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)


    # Find the optimal parameters from the validation set
    # Best threshold for classification only (no fairness)
    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                dataset_orig_valid_pred, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)



    # Estimate optimal parameters for the ROC method
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                    privileged_groups=privileged_groups, 
                                    low_class_thresh=0.01, high_class_thresh=0.99,
                                    num_class_thresh=100, num_ROC_margin=50,
                                    metric_name=metric_name,
                                    metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
    print("Optimal ROC margin = %.4f" % ROC.ROC_margin)




    # Predictions from Validation Set

    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    print("#### Validation set")
    print("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

    metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred, 
                    unprivileged_groups, privileged_groups)

    # Transform the validation set
    dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

    print("#### Validation set")
    print("##### Transformed predictions - With fairness constraints")
    metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred, 
                    unprivileged_groups, privileged_groups)



    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    print("#### Test set")
    print("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                    unprivileged_groups, privileged_groups)


    # Metrics for the transformed test set
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

    print("#### Test set")
    print("##### Transformed predictions - With fairness constraints")
    metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                    unprivileged_groups, privileged_groups)

main()
