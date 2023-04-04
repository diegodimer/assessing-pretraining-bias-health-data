# Assessing pre-training bias in Health data and estimating its impact on machine learning algorithms
This code was used in the work presented for the partial fulfillment of the requirements for the degree of Bachelor in Computer Science. The full text should be available on lume (https://www.lume.ufrgs.br/)
## Definition
- BaseDataset implements the basic functions for a new dataset to be evaluated
    - When adding a new dataset, it is needed to create a new file that inherits from this class, and fill all the variables in the init construct of this base class
    - This class has the following methods: 
        - execute_models(): generate pandas profiling report if not existent and run the four machine learning algorithms used in this work over the _num_repetitions_ training sets. Prints in the end the accuracy and f1 score and return the mean value for these metrics
        - evaluate_metrics(protected_attribute, privileged_group, group_variable, dataset=None, cddl_only=False, print_metrics=True): evaluate the pre-training metrics (CI, KL Divergence, KS and CDDL) for the dataset provided on the _dataset_ parameter, or the whole dataset if none is provided, if _cddl_only_ is set to true, return only value for CDDL, print the metrics values if necessary
        - gen_graph(self, protected_attr=None, labels_labels=None, outcomes_labels=None, dataset=None, predicted_attr=None, file_name=None, df_type=None, graph_title=None, ax=None): generate the graph for the distribution on the _protected_attr_, can be passed labels to the protected attr and the outcome values, full dataset will be used unless provided a new one, will write to a file in the dataset directory with the name provided or a default name will be used. Axis can be passed to not save the graph and only add to the plot (usefull for subplots)
        - save_tree(): prints the decision tree used for the training
        - best_neighbors_finder(): find the optimal k-value for the KNN, and generates the graph of the values considering accuracy, f1-score and error rate

- IntersectionalBiasDataset, HeartDataset and AlcoholDataset are the three datasets used in the work, in the original format using the BaseDataset implementation. All experiments and graph generation for the reports on the work used the *-Experiment.py files, which were not planned to be reused in future work, but kept for reproducibility.


```
@monography{tcc_diego,
author={Diego Dimer Rodrigues},
title={Assessing pre-training bias in Health data
and estimating its impact on machine
learning algorithms}, 
school={Universidade Federal do Rio Grande do Sul}, 
year={2023},
type={TCC}
}
```