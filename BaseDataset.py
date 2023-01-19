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

    def _run(self): 
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
        model_predicted = model.predict(self.X_test)
        model_conf_matrix = confusion_matrix(self.y_test, model_predicted)
        model_acc_score = accuracy_score(self.y_test, model_predicted)
        model_f1_score = f1_score(self.y_test,model_predicted)
        print(f"confussion matrix for {model_name}: \n{model_conf_matrix}")
        print(f"Accuracy of {model_name}: {model_acc_score*100}")
        print(f"F1 Score of {model_name}: {model_f1_score*100}\n" )
        # print(classification_report(y_test,model_predicted))

    def best_neighbors_finder(self):
        y = self.dataset[self.predicted_attr]
        X = self.dataset.drop(self.predicted_attr,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
        accuracy = []
        f1 = []
        error_rate = []
        max_n = 300#self.X_train.shape[0] + 1
        for i in range(1,max_n):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(self.X_train,self.y_train)
            model_predicted = model.predict(self.X_test)

            model_acc_score = accuracy_score(self.y_test, model_predicted) 
            model_f1_score = f1_score(self.y_test,model_predicted) 
            error_rate.append(np.mean(model_predicted != self.y_test))
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
