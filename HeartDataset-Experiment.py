from HeartDataset import HeartDataset
import pandas as pd
import numpy as np
#detalhar escolha de hiper parâmetros dos algoritimos de treino
#criar uma figura com os 10 datasets de treino e teste. Variar só o treino (e mostrar pra cada)
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
# calcular as métricas em cima dos train sets pra balanceamentos/desbalanceamentos
def move_train_instances_to_test(x_train, x_test, target, percentage=None, value=None, sex=0):
    ### FAZER SÓ NO TREINO
    new_x = x_train.loc[(x_train['sex'] == sex) &
                        (x_train['target'] == target)]
    drop_indices = np.random.choice(
        new_x.index, round(len(new_x)*percentage if value is None else value), replace=False)
    new_xtest = pd.concat([x_test, x_train.loc[drop_indices]])
    new_xtrain = x_train.drop(drop_indices)
    return new_xtrain, new_xtest


def remove_instances(x, target, value, sex=0):
    new_x = x.loc[(x['sex'] == sex) & (x['target'] == target)]
    drop_indices = np.random.choice(new_x.index, value if value>=1 else round(len(new_x)*value), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def original_dataset():
    h = HeartDataset()
    print('==========Original Dataset===========')

    h.execute_models()
    gen_graph_for_sets(h, "original-dataset")
    # print(h.model_conf_matrix['DecisionTreeClassifier'][0])


def high_imbalance():
    def perturbe(X_train, y_train):
        complete_x_train = X_train.reset_index()
        complete_x_train['target'] = y_train.reset_index()['target']

        complete_x_train = remove_instances(complete_x_train, 0, 0.8)
        new_x_train = remove_instances(complete_x_train, 1, 0.4)

        new_y_train = new_x_train['target']
        new_x_train = new_x_train.drop('target', axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print("Move 80% of woman with negative output and 40% with positive output from train to test set, respectively")
    h = HeartDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.execute_models()

    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    def perturbe(X_train, y_train):
        complete_x_train = X_train.reset_index()
        complete_x_train['target'] = y_train.reset_index()['target']

        positive_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train['target'] == 1)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train['target'] == 1)])
        negative_out_in_train = len(complete_x_train.loc[(
            complete_x_train['sex'] == 1) & (complete_x_train['target'] == 0)]) - len(complete_x_train.loc[(
                complete_x_train['sex'] == 0) & (complete_x_train['target'] == 0)])

        complete_x_train = remove_instances(complete_x_train, 1, positive_out_in_train, sex=1)
        new_x_train = remove_instances(complete_x_train, 0, negative_out_in_train, sex=1)
        new_y_train = new_x_train['target']
        new_x_train = new_x_train.drop('target', axis=1)
        new_x_train = new_x_train.drop('index', axis=1)


        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = HeartDataset()
    h.dropper = True
    h.gen_graph()
    h.perturbe = perturbe
    h.execute_models()

    gen_graph_for_sets(h, "equal-balance")


original_dataset()
high_imbalance()
equal_balance()
