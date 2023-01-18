import pandas as pd
import pandas_profiling as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, f1_score

class BaseDataset():
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    def _run(self, dataset, predicted_attr): 
        pp.ProfileReport(dataset).to_file(type(self).__name__)
        y = dataset[predicted_attr]
        X = dataset.drop(predicted_attr,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

        print(self.y_test.value_counts())
        print(self.y_train.value_counts())
        # print(X_test)
        # print(X_train)

        # models
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)
        dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
        knn = KNeighborsClassifier(n_neighbors=20)
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

if __name__ == "__main__":
    BaseDataset._run(pd.read_csv("datasets/heart.csv"), "target")