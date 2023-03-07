import pandas as pd
import pandas_profiling as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PreTrainingBias import PreTrainingBias
import graphviz


class BaseDataset():
    # base variables to be overwritten by child class
    dataset          = None
    predicted_attr   = None
    max_iter         = None
    n_estimators     = None
    random_state     = None
    max_depth        = None
    n_neighbors      = None
    criterion        = None
    positive_outcome = None
    negative_outcome = None
    model_predicted  = None
    num_repetitions  = 2
    # variables to be used by this main class
    accs_lr          = []
    f1s_lr           = []
    accs_dt          = []
    f1s_dt           = []
    accs_rf          = []
    f1s_rf           = []
    accs_knn         = []
    f1s_knn          = []
    x_train_list     = []
    x_test_list      = []
    y_train_list     = []
    y_test_list      = []

    def __init__(self) -> None:
        self.ptb = PreTrainingBias()

    def _run(self):
        # self._gen_pp_report()
        for i in range(self.num_repetitions):
            self._gen_train_test_sets(i)
            self._execute_models()
        self._print_mean(self.accs_lr)
        self._print_mean(self.f1s_lr)
        self._print_mean(self.accs_dt)
        self._print_mean(self.f1s_dt)
        self._print_mean(self.accs_rf)
        self._print_mean(self.f1s_rf)
        self._print_mean(self.accs_knn)
        self._print_mean(self.f1s_knn)
     
    def _print_mean(self, num_list: list):
        mean = (sum(num_list)/len(num_list))
        print("{:.3f}".format(mean))

    def _gen_train_test_sets(self, random_state):
        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state = random_state)
        self.x_train_list.append(self.X_train)
        self.x_test_list.append(self.X_test)
        self.y_train_list.append(self.y_train)
        self.y_test_list.append(self.y_test)

    def _gen_pp_report(self):
        my_file = Path(f"{type(self).__name__}.html")
        if not my_file.exists():
            pp.ProfileReport(self.dataset).to_file(f"{type(self).__name__}.html")

    def _execute_models(self):         
        self.lr = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.dt = DecisionTreeClassifier(criterion = 'entropy',random_state=self.random_state,max_depth=self.max_depth)
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state,max_depth=self.max_depth)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        acc_lr, f1_lr  = self._run_model(self.lr)
        acc_dt, f1_dt  = self._run_model(self.dt)
        acc_rf, f1_rf  = self._run_model(self.rf)
        acc_knn, f1_knn  = self._run_model(self.knn)

        self.accs_lr.append(acc_lr)
        self.f1s_lr.append(f1_lr)
        self.accs_dt.append(acc_dt)
        self.f1s_dt.append(f1_dt)
        self.accs_rf.append(acc_rf)
        self.f1s_rf.append(f1_rf)
        self.accs_knn.append(acc_knn)
        self.f1s_knn.append(f1_knn)

    def _run_model(self, model):
        # model_name = type(model).__name__
        model.fit(self.X_train,self.y_train)
        self.model_predicted = model.predict(self.X_test)
        # model_conf_matrix = confusion_matrix(self.y_test, self.model_predicted)
        model_acc_score = accuracy_score(self.y_test, self.model_predicted) * 100
        model_f1_score = f1_score(self.y_test,self.model_predicted) * 100
        # print(f"confussion matrix for {model_name}: \n{model_conf_matrix}")
        # print(f"Accuracy of {model_name}: {model_acc}")
        # print(f"F1 Score of {model_name}: {model_f1}\n" )
        # print(classification_report(self.y_test,self.model_predicted))
        return model_acc_score, model_f1_score

    def evaluate_metrics(self, protected_attribute, privileged_group, group_variable, dataset = None, cddl_only=False):
        if dataset is None:
            dataset = self.dataset
        dic = self.ptb.global_evaluation (dataset, self.predicted_attr, self.positive_outcome, protected_attribute, privileged_group, group_variable)

        for key in dic:
            if cddl_only:
                if 'CDDL' in key:
                    val = "{:.3f}".format(dic[key])
                    print("{: <50} {: >50}".format(key,val))
                    break
            else:
                val = "{:.3f}".format(dic[key])
                print("{: <50} {: >50}".format(key,val))

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

    def gen_graph(self,protected_attr=None, labels_labels = None, outcomes_labels = None, dataset = None, predicted_attr = None, file_name = None, df_type=None):
        if dataset is None:
            dataset = self.dataset
        if predicted_attr is None:
            predicted_attr = self.predicted_attr
        if protected_attr is None:
            protected_attr = self.protected_attr
        if type(protected_attr) == str:
            protected_attr = [protected_attr]
  
        for attr in protected_attr:
            labels = dataset[attr].unique().tolist()
            labels.sort()
            outcomes = dataset[predicted_attr].unique().tolist()
            outcomes.sort()
            bar_ind = []
            bar_list = []
            for i in outcomes:
                for j in labels:
                    bar_ind.append(len(dataset[ (dataset[predicted_attr] == i) & (dataset[attr] == j)]))
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
            if file_name is not None:
                fig.savefig(f"{file_name}.png")
            else:
                fig.savefig(f"{type(self).__name__}/{df_type}{predicted_attr}-{attr}.png")

    def save_tree(self):
        dot_data = tree.export_graphviz(self.dt, out_file=None, 
                            feature_names=self.X_train.columns,  
                            class_names=[str(x) for x in self.y_test.unique()],  
                            filled=True, rounded=True,  
                            special_characters=True,impurity=False,max_depth=3)
        
        graph = graphviz.Source(dot_data)
        graph.render(f"tree-{type(self).__name__}")

    def result_checker(self, labels_labels = None, protected_attr=None):
        df_out = self.X_test.reset_index()
        y_hats  = pd.DataFrame(self.model_predicted)
        df_out["Actual"] = self.y_test.reset_index()[self.predicted_attr]
        df_out["Prediction"] = y_hats.reset_index()[0]

        if protected_attr == None:
            protected_attr = self.protected_attr

        for i in self.protected_attr:
            self.gen_graph(i, dataset = df_out, predicted_attr = 'Actual', labels_labels=labels_labels, file_name=f"{type(self).__name__}/testSet-{i}")
            df_err = df_out.loc[ df_out['Actual'] != df_out['Prediction']]
            self.gen_graph(i, dataset = df_err, predicted_attr = 'Prediction', labels_labels=labels_labels, file_name=f"{type(self).__name__}/errors-{i}")
            df_corr = df_out.loc[ df_out['Actual'] == df_out['Prediction']]
            self.gen_graph(i, dataset = df_corr, predicted_attr = 'Prediction', labels_labels=labels_labels, file_name=f"{type(self).__name__}/acerts-{i}")
