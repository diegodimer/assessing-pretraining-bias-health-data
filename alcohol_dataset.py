import numpy as np
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
import os

import pandas as pd

from aif360.datasets import StandardDataset
            #     return '>35'
            # elif x <= 20:
            #     return '<=20'
            # elif x > 20 and x <= 25:
            #     return '>20, <=25'
            # elif x > 25 and x <= 30:
            #     return '>25, <=30'
            # else:
            #     return '>30, <=35'


default_mappings = {
    'label_maps': [{1: 'Risk', 0: 'No Risk'}],
    # 'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
    #                              {1.0: 'Male', 0.0: 'Female'}]
}


class AlcoholDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='Risk_Stratification',
                 favorable_classes=[0],
                 protected_attribute_names=[  # se age entrar aqui, quais idades são privilegiadas?
                     'cotas', 'gender', 'gender_id'],
                 # rodar as métricas de pretraining pra saber quem é o privilegiado?
                 privileged_classes=[[0], [2], [2]],
                 instance_weights_name=None,
                 categorical_features=['cotas', 'gender', 'sexual_orientation', 'gender_id',  # categóricos incluem binários
                                       'marital_status', 'children', 'family_income', 'live_with', 'tobacco',
                                       'alcohol', 'alcohol_dose', 'alcohol_binge', 'Audit_Total',
                                       'Risk_Stratification', 'cannabis', 'cocaine', 'family_relations',
                                       'friends_relations', 'physical_activities', 'Sleep', 'workload',
                                       'colleagues_relations', 'professors_relations', 'bullying_school',
                                       'bullying_university', 'bullying_yes', 'repeat_university',
                                       'satisfaction', 'change_giveup', 'child_abuse', 'adult_abuse', 'phq_1',
                                       'phq_2', 'phq_total', 'phq_diagnosis', 'suicide_ideation_life',
                                       'suicide_ideation_month', 'suicide_family', 'suicidal_attempt',
                                       'suicide_attempt_dic', 'suicide_teach'],
                 features_to_keep=[], features_to_drop=[],  # which features to drop?
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments."""
        train_path = "banco_flavia.csv"

        try:
            train = pd.read_csv(train_path, header=0, na_values=na_values)
            # test = pd.read_csv(test_path, header=0, names=column_names,
            #                    skipinitialspace=True, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            import sys
            sys.exit(1)

        df = train
        
        super(AlcoholDataset, self).__init__(df=df, label_name=label_name,
                                             favorable_classes=favorable_classes,
                                             protected_attribute_names=protected_attribute_names,
                                             privileged_classes=privileged_classes,
                                             instance_weights_name=instance_weights_name,
                                             categorical_features=categorical_features,
                                             features_to_keep=features_to_keep,
                                             features_to_drop=features_to_drop, na_values=na_values,
                                             custom_preprocessing=custom_preprocessing, metadata=metadata)


def load_preproc_data_alcohol(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        # df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def age_group(x):
            if x <= 20:
                return 0 #'<=20'
            elif x > 20 and x <= 25:
                return 1 #'>20, <=25'
            elif x > 25 and x <= 30:
                return 2 # '>25, <=30'
            else:
                return 3 #'>30, <=35'

        # 1 -> Feminino
        # 2 -> Masculino
        def risk_labels(risk, gender):
            if risk <= 2 and gender == 1 or risk <= 3 and gender == 2:
                return 0
            else:
                return 1

        # Change age to categories
        df['age'] = df['age'].apply(lambda x: age_group(x))

        df['Risk'] = df.apply(lambda x: risk_labels(
            x.Risk_Stratification, x.gender), axis=1)

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Risk'] == 0]
            df_1 = df[df['Risk'] == 1]
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['age', 'year', 'cotas', 'gender', 'sexual_orientation',
                   'marital_status', 'children', 'family_income', 'live_with', 'tobacco',
                   'alcohol', 'alcohol_dose', 'alcohol_binge', 'cannabis', 'physical_activities', 'Sleep', 'workload',
                   'satisfaction', 'change_giveup', 'child_abuse', 'suicide_ideation_life', 'suicide_family']
    D_features = [
        'cotas', 'gender'] if protected_attributes is None else protected_attributes
    Y_features = ['Risk']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = [ 'year', 'sexual_orientation',
                            'marital_status', 'children', 'family_income', 'live_with', 'tobacco',
                            'alcohol', 'alcohol_dose', 'alcohol_binge', 'cannabis', 'physical_activities', 'Sleep', 'workload',
                            'satisfaction', 'change_giveup', 'child_abuse', 'suicide_ideation_life',
                            'suicide_family']

    # privileged classes
    all_privileged_classes = {"cotas" : lambda x: x == 0, "gender": lambda x: x == 2} # ?

    # protected attribute maps
    # all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
    #                                 "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AlcoholDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        # metadata={'protected_attribute_maps': [all_protected_attribute_maps[x]
        #                                        for x in D_features]},
        custom_preprocessing=custom_preprocessing)
