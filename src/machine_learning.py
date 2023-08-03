import numpy as np


def set_target_column_in_df(df, target, df_id_column, target_id_column, target_value_column):
    """
    Create and fill the target column in a dataframe according to target values.

    :param df: Pandas dataframe to fill.
    :param target: Pandas dataframe containing target values.
    :param df_id_column: Identifiers column in df.
    :param target_id_column: Identifiers column in target.
    :param target_value_column: Values column in target.

    :return:

    """
    df['target'] = np.NaN
    for patient_id in set(target[target_id_column]):
        df.loc[df[df_id_column] == patient_id, 'target'] = \
            target.loc[target[target_id_column] == patient_id][target_value_column].item()
