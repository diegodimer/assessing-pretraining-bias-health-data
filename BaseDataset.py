import pandas as pd
import pandas_profiling as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PreTrainingBias import PreTrainingBias

class BaseDataset():
    dataset         = None
    predicted_attr  = None
    X_train         = None
    X_test          = None
    y_train         = None
    y_test          = None
    max_iter        = None
    n_estimators    = None
    random_state    = None
    max_depth       = None
    n_neighbors     = None
    criterion       = None
    positive_outcome = None
    negative_outcome = None
    model_predicted = None
    
    def __init__(self) -> None:
        self.ptb = PreTrainingBias()
    
    def _run(self): 
        my_file = Path(f"{type(self).__name__}.html")

        if not my_file.exists():
            pp.ProfileReport(self.dataset).to_file(f"{type(self).__name__}.html")

        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
        
        print(self.y_test.value_counts())
        print(self.y_train.value_counts())

        # models
        lr = LogisticRegression(max_iter=self.max_iter)
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state,max_depth=self.max_depth)
        dt = DecisionTreeClassifier(criterion = 'entropy',random_state=self.random_state,max_depth=self.max_depth)
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.run_model(lr)
        self.run_model(rf)
        self.run_model(dt)
        self.run_model(knn)

    def run_model(self, model):
        model_name = type(model).__name__
        model.fit(self.X_train,self.y_train)
        self.model_predicted = model.predict(self.X_test)
        model_conf_matrix = confusion_matrix(self.y_test, self.model_predicted)
        model_acc_score = accuracy_score(self.y_test, self.model_predicted)
        model_f1_score = f1_score(self.y_test,self.model_predicted)
        print(f"confussion matrix for {model_name}: \n{model_conf_matrix}")
        print(f"Accuracy of {model_name}: {model_acc_score*100}")
        print(f"F1 Score of {model_name}: {model_f1_score*100}\n" )
        # print(classification_report(self.y_test,self.model_predicted))

    def evaluate_metrics(self, protected_attribute, privileged_group, unprivileged_group, group_variable):
        return self.ptb.global_evaluation (self.dataset, self.predicted_attr, self.positive_outcome, protected_attribute, privileged_group, unprivileged_group, group_variable)
        
    def best_neighbors_finder(self):
        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
        accuracy = []
        f1 = []
        error_rate = []
        max_n = self.X_train.shape[0] + 1
        for i in range(1,max_n):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(self.X_train,self.y_train)
            self.model_predicted = model.predict(self.X_test)

            model_acc_score = accuracy_score(self.y_test, self.model_predicted) 
            model_f1_score = f1_score(self.y_test,self.model_predicted) 
            error_rate.append(np.mean(self.model_predicted != self.y_test))
            accuracy.append(model_acc_score)
            f1.append(model_f1_score)

        plt.figure(figsize=(20,12))
        plt.plot(range(1,max_n),error_rate,color='blue', markersize=10, label='error rate')
        plt.plot(range(1,max_n),accuracy,color='magenta', markersize=10, label='accuracy')
        plt.plot(range(1,max_n),f1,color='green', markersize=10, label='f1 score')
        plt.title('Performance Metrics vs. K Value')
        plt.xlabel('K')
        plt.legend(title='Metric')
        plt.savefig(f'{type(self).__name__}.png')
        req_k_value = error_rate.index(min(error_rate))+1
        req_acc_value = accuracy.index(max(accuracy))+1
        req_f1_value = f1.index(max(f1))+1
        print("Minimum error:-",min(error_rate),"at K =",req_k_value)
        print("Maximum acc:-",max(accuracy),"at K =",req_acc_value)
        print("maximum f1:-",max(f1),"at K =",req_f1_value)

    def gen_graph(self,protected_attr, labels_labels = None, outcomes_labels = None, dataset = None, predicted_attr = None):
        if dataset is None:
            dataset = self.dataset
        if predicted_attr is None:
            predicted_attr = self.predicted_attr
            
        labels = dataset[protected_attr].unique().tolist()
        labels.sort()
        outcomes = dataset[predicted_attr].unique().tolist()
        outcomes.sort()

        bar_ind = []
        bar_list = []
        for i in outcomes:
            for j in labels:
                bar_ind.append(len(dataset[ (dataset[predicted_attr] == i) & (dataset[protected_attr] == j) ]))
            bar_list.append(bar_ind)
            bar_ind = []

        width = 0.35       # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        previous = None
        for i,j in enumerate(bar_list):
            if i == 0:
                ax.bar(labels, j, width, label=f'{i}')
                previous = np.array(j)
            if i!=0:
                ax.bar(labels, j, width, label=f'{i}', bottom=previous)
                previous += np.array(j)
        plt.figure(figsize=(40,24))
        if labels_labels is not None:
            x_ticks_labels = labels_labels
            ax.set_xticks(labels)
            ax.set_xticklabels(x_ticks_labels)
        if outcomes_labels is not None:
            ax.legend(outcomes_labels)
        else:
            ax.legend(title=predicted_attr)
        ax.set_ylabel('count')
        for bars in ax.containers: ## if the bars should have the values
            ax.bar_label(bars)
        fig.savefig(f"{type(self).__name__}-{protected_attr}.png")
