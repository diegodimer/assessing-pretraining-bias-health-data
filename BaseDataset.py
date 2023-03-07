import pandas as pd
import pandas_profiling as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PreTrainingBias import PreTrainingBias
import graphviz
from collections import defaultdict


class BaseDataset():
    # base variables to be overwritten by child class
    dataset = None
    predicted_attr = None
    max_iter = None
    n_estimators = None
    random_state = None
    max_depth = None
    n_neighbors = None
    criterion = None
    positive_outcome = None
    negative_outcome = None
    num_repetitions = 5
    # variables to be used by this main class
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    predicted_list = []
    accs = defaultdict(list)
    f1s = defaultdict(list)
    models = defaultdict()

    def __init__(self) -> None:
        self.ptb = PreTrainingBias()

        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=self.max_iter, random_state=self.random_state)

        self.models['Decision Tree'] = DecisionTreeClassifier(
            criterion='entropy', random_state=self.random_state, max_depth=self.max_depth)

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, max_depth=self.max_depth)

        self.models['KNN'] = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def _run(self):
        # self._gen_pp_report()
        for i in range(self.num_repetitions):
            self._gen_train_test_sets(i)
            for model_name in self.models:
                acc, f1 = self._run_model(self.models[model_name])
                self.accs[model_name].append(acc)
                self.f1s[model_name].append(f1)

        for model_name in self.accs:
            acc = self.accs[model_name]
            f1 = self.f1s[model_name]
            acc_mean = (sum(acc)/len(acc))
            f1_mean = (sum(f1)/len(f1))
            print("{: <30}{: >30.3f}".format(model_name + ' acc', acc_mean))
            print("{: <30}{: >30.3f}".format(model_name + ' f1', f1_mean))

    def _gen_train_test_sets(self, random_state):
        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state)
        self.x_train_list.append(self.X_train)
        self.x_test_list.append(self.X_test)
        self.y_train_list.append(self.y_train)
        self.y_test_list.append(self.y_test)

    def _gen_pp_report(self):
        my_file = Path(f"{type(self).__name__}/report.html")
        if not my_file.exists():
            pp.ProfileReport(self.dataset).to_file(
                f"{type(self).__name__}/report.html")

    def _run_model(self, model):
        model.fit(self.X_train, self.y_train)
        model_predicted = model.predict(self.X_test)
        self.predicted_list.append(model_predicted)
        # model_conf_matrix = confusion_matrix(self.y_test, self.model_predicted)
        model_acc_score = accuracy_score(
            self.y_test, model_predicted) * 100
        model_f1_score = f1_score(self.y_test, model_predicted) * 100
        return model_acc_score, model_f1_score

    def evaluate_metrics(self, protected_attribute, privileged_group, group_variable, dataset=None, cddl_only=False):
        if dataset is None:
            dataset = self.dataset
        dic = self.ptb.global_evaluation(
            dataset, self.predicted_attr, self.positive_outcome, protected_attribute, privileged_group, group_variable)

        for key in dic:
            if cddl_only:
                if 'CDDL' in key:
                    print("{: <30}{: >30.3f}".format(key, dic[key]))
                    break
            else:
                print("{: <30}{: >30.3f}".format(key, dic[key]))

    def gen_graph(self, protected_attr=None, labels_labels=None, outcomes_labels=None, dataset=None, predicted_attr=None, file_name=None, df_type=None):
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
                    bar_ind.append(
                        len(dataset[(dataset[predicted_attr] == i) & (dataset[attr] == j)]))
                bar_list.append(bar_ind)
                bar_ind = []

            # the width of the bars: can also be len(x) sequence
            width = 0.35
            fig, ax = plt.subplots()
            previous = None
            for i, j in enumerate(bar_list):
                if i == 0:
                    ax.bar(labels, j, width, label=f'{i}')
                    previous = np.array(j)
                if i != 0:
                    ax.bar(labels, j, width, label=f'{i}', bottom=previous)
                    previous += np.array(j)
            plt.figure(figsize=(40, 24))
            if labels_labels is not None:
                x_ticks_labels = labels_labels
                ax.set_xticks(labels)
                ax.set_xticklabels(x_ticks_labels)
            if outcomes_labels is not None:
                ax.legend(outcomes_labels)
            else:
                ax.legend(title=predicted_attr)
            ax.set_ylabel('count')
            for bars in ax.containers:  # if the bars should have the values
                ax.bar_label(bars)
            if file_name is not None:
                fig.savefig(f"{type(self).__name__}/{file_name}.png")
            else:
                fig.savefig(
                    f"{type(self).__name__}/{df_type}-{predicted_attr}-{attr}.png")

    def result_checker(self, repetition_number, labels_labels=None, protected_attr=None):
        df_out = self.x_test_list[repetition_number].reset_index()
        y_hats = pd.DataFrame(self.predicted_list[repetition_number])
        df_out["Actual"] = self.y_test_list[repetition_number].reset_index()[
            self.predicted_attr]
        df_out["Prediction"] = y_hats.reset_index()[0]

        if protected_attr is None:
            protected_attr = self.protected_attr

        for i in self.protected_attr:
            self.gen_graph(i, dataset=df_out, predicted_attr='Actual',
                           labels_labels=labels_labels, df_type=f'testSet-{repetition_number}')

            df_err = df_out.loc[df_out['Actual'] != df_out['Prediction']]
            self.gen_graph(i, dataset=df_err, predicted_attr='Prediction',
                           labels_labels=labels_labels, df_type=f'error-{repetition_number}')

            df_corr = df_out.loc[df_out['Actual'] == df_out['Prediction']]
            self.gen_graph(i, dataset=df_corr, predicted_attr='Prediction',
                           labels_labels=labels_labels, df_type=f'correct-{repetition_number}')

    def save_tree(self):
        dot_data = tree.export_graphviz(self.models['Decision Tree'], out_file=None,
                                        feature_names=self.X_train.columns,
                                        class_names=[
                                            str(x) for x in self.y_test.unique()],
                                        filled=True, rounded=True,
                                        special_characters=True, impurity=False, max_depth=3)

        graph = graphviz.Source(dot_data)
        graph.render(f"tree-{type(self).__name__}")

    def best_neighbors_finder(self):
        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=0)
        accuracy = []
        f1 = []
        error_rate = []
        max_n = self.X_train.shape[0] + 1
        for i in range(1, max_n):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(self.X_train, self.y_train)
            model_predicted = model.predict(self.X_test)

            model_acc_score = accuracy_score(self.y_test, model_predicted)
            model_f1_score = f1_score(self.y_test, model_predicted)
            error_rate.append(np.mean(model_predicted != self.y_test))
            accuracy.append(model_acc_score)
            f1.append(model_f1_score)

        plt.figure(figsize=(20, 12))
        plt.plot(range(1, max_n), error_rate, color='blue',
                 markersize=10, label='error rate')
        plt.plot(range(1, max_n), accuracy, color='magenta',
                 markersize=10, label='accuracy')
        plt.plot(range(1, max_n), f1, color='green',
                 markersize=10, label='f1 score')
        plt.title('Performance Metrics vs. K Value')
        plt.xlabel('K')
        plt.legend(title='Metric')
        plt.savefig(f'{type(self).__name__}.png')
        req_k_value = error_rate.index(min(error_rate))+1
        req_acc_value = accuracy.index(max(accuracy))+1
        req_f1_value = f1.index(max(f1))+1
        print("Minimum error:-", min(error_rate), "at K =", req_k_value)
        print("Maximum acc:-", max(accuracy), "at K =", req_acc_value)
        print("maximum f1:-", max(f1), "at K =", req_f1_value)
