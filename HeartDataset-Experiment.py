from HeartDataset import HeartDataset
import pandas as pd
import numpy as np


def gen_graph_for_sets(h: HeartDataset, name: str):
    full_dataset_train = pd.concat(
        [h.x_train_list[i] for i in range(len(h.x_train_list))])
    full_dataset_train['target'] = pd.concat(
        [h.y_train_list[i] for i in range(len(h.y_train_list))])

    full_dataset_test = pd.concat([h.x_test_list[i]
                                  for i in range(len(h.x_test_list))])
    full_dataset_test['target'] = pd.concat(
        [h.y_test_list[i] for i in range(len(h.y_test_list))])

    print("\nMetrics calculated over the train datasets (concatenanted from all 10 repetitions)")
    h.evaluate_metrics('sex', 1, 'cp', dataset=full_dataset_train)
    h.evaluate_metrics('sex', 1, 'thal', cddl_only=True,
                       dataset=full_dataset_train)
    h.gen_graph('sex', df_type=f'{name}-trainDataset',
                dataset=full_dataset_train)
    h.gen_graph('sex', df_type=f'{name}-testDataset',
                dataset=full_dataset_test)

# TO-DO: export "remove instances" as a method


def move_train_instances_to_test(x_train, x_test, target, percentage=None, value=None, sex=0):
    new_x = x_train.loc[(x_train['sex'] == sex) &
                        (x_train['target'] == target)]
    drop_indices = np.random.choice(
        new_x.index, round(len(new_x)*percentage if value is None else value), replace=False)
    new_xtest = pd.concat([x_test, x_train.loc[drop_indices]])
    new_xtrain = x_train.drop(drop_indices)
    return new_xtrain, new_xtest


def remove_instances_from_train_and_test(x_train, x_test, target, percentage=None, value_from_train=None, value_from_test=None, sex=0):
    new_x_train = x_train.loc[(x_train['sex'] == sex) &
                              (x_train['target'] == target)]
    new_x_test = x_test.loc[(x_test['sex'] == sex) &
                            (x_test['target'] == target)]
    drop_indices_train = np.random.choice(
        new_x_train.index, round(len(new_x_train)*percentage if value_from_train is None else value_from_train), replace=False)
    drop_indices_test = np.random.choice(
        new_x_test.index, round(len(new_x_test)*percentage if value_from_test is None else value_from_test), replace=False)
    new_xtrain = x_train.drop(drop_indices_train)
    new_xtest = x_test.drop(drop_indices_test)
    return new_xtrain, new_xtest


def original_dataset():
    h = HeartDataset()
    print('==========Original Dataset===========')

    h.execute_models()
    gen_graph_for_sets(h, "original-dataset")
    # print(h.model_conf_matrix['DecisionTreeClassifier'][0])


def high_imbalance():
    def perturbe(X_train, y_train, X_test, y_test):
        complete_x_train = X_train.reset_index()
        complete_x_train['target'] = y_train.reset_index()['target']

        complete_x_test = X_test.reset_index()
        complete_x_test['target'] = y_test.reset_index()['target']

        complete_x_train, complete_x_test = move_train_instances_to_test(
            complete_x_train, complete_x_test, 0, 0.8)
        new_x_train, new_x_test = move_train_instances_to_test(
            complete_x_train, complete_x_test, 1, 0.4)

        new_y_train = new_x_train['target']
        new_x_train = new_x_train.drop('target', axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        new_y_test = new_x_test['target']
        new_x_test = new_x_test.drop('target', axis=1)
        new_x_test = new_x_test.drop('index', axis=1)

        return new_x_train, new_y_train, new_x_test, new_y_test

    print("==========High Imbalance==========")
    print("Move 80% of woman with negative output and 40% with positive output from train to test set, respectively")
    h = HeartDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.execute_models()

    # df_out = h.x_train_list[0].reset_index()
    # df_out["Actual"] = h.y_train_list[0].reset_index()[h.predicted_attr]

    # df_out2 = h.x_test_list[0].reset_index()
    # y_hats = pd.DataFrame(h.predicted_list['DecisionTreeClassifier'][0])
    # df_out2["Actual"] = h.y_test_list[0].reset_index()[h.predicted_attr]
    # df_out2["Prediction"] = y_hats.reset_index()[0]

    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    def perturbe(X_train, y_train, X_test, y_test):
        complete_x_train = X_train.reset_index()
        complete_x_train['target'] = y_train.reset_index()['target']

        complete_x_test = X_test.reset_index()
        complete_x_test['target'] = y_test.reset_index()['target']

        positive_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train['target'] == 1)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train['target'] == 1)])
        negative_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train['target'] == 0)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train['target'] == 0)])
        positive_out_in_test = len(complete_x_test.loc[(
            complete_x_test['sex'] == 1) & (complete_x_test['target'] == 1)]) - len(complete_x_test.loc[(
                complete_x_test['sex'] == 0) & (complete_x_test['target'] == 1)])
        negative_out_in_test = len(complete_x_test.loc[(
            complete_x_test['sex'] == 1) & (complete_x_test['target'] == 0)]) - len(complete_x_test.loc[(
                complete_x_test['sex'] == 0) & (complete_x_test['target'] == 0)])

        complete_x_train, complete_x_test = remove_instances_from_train_and_test(
            complete_x_train, complete_x_test, target=0, value_from_train=negative_out_in_train, value_from_test=negative_out_in_test, sex=1)
        new_x_train, new_x_test = remove_instances_from_train_and_test(
            complete_x_train, complete_x_test, target=1, value_from_train=positive_out_in_train, value_from_test=positive_out_in_test, sex=1)

        new_y_train = new_x_train['target']
        new_x_train = new_x_train.drop('target', axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        new_y_test = new_x_test['target']
        new_x_test = new_x_test.drop('target', axis=1)
        new_x_test = new_x_test.drop('index', axis=1)

        return new_x_train, new_y_train, new_x_test, new_y_test

    print("==========Equally Balanced==========")
    print("Move 80% of woman with negative output and 40% with positive output from train to test set, respectively")
    h = HeartDataset()
    h.dropper = True
    h.gen_graph()
    h.perturbe = perturbe
    h.execute_models()

    for i in range(h.num_repetitions):
        df_out = h.x_train_list[i].reset_index()
        df_out["target"] = h.y_train_list[i].reset_index()[h.predicted_attr]
        h.gen_graph(dataset=df_out, file_name=f"balanced-{i}-train-dataset")
    h.gen_graph()
    # df_out2 = h.x_test_list[0].reset_index()
    # y_hats = pd.DataFrame(h.predicted_list['DecisionTreeClassifier'][0])
    # df_out2["Actual"] = h.y_test_list[0].reset_index()[h.predicted_attr]
    # df_out2["Prediction"] = y_hats.reset_index()[0]

    gen_graph_for_sets(h, "equal-balance")


# original_dataset()
# high_imbalance()
equal_balance()
