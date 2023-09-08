def convert_categorical_column_to_int(df, column_name):
    """
    Convert the categorical column in the dataframe to integer values.

    :param df: Pandas DataFrame.
    :param column_name: Name of the categorical column to be converted.

    :return:

    """
    categories = df[column_name].unique()
    numbers = [_ for _ in range(len(categories))]
    df[column_name].replace(categories, numbers, inplace=True)


def set_time_point(df, class_name, time_point):
    """
    Set a specific time point for rows in a Pandas DataFrame based on the presence of class-related columns.

    :param df: Pandas DataFrame containing the data.
    :param class_name: Prefix of class-related columns.
    :param time_point: Time point to assign to rows with class-related data.

    :return:

    """
    # Get columns associated to class
    columns = []
    for column in df.columns:
        if column.startswith(class_name):
            columns.append(column)

    # Set time points in df
    for i in range(1, len(df)):
        if any(element in list(df.iloc[i].dropna().keys()) for element in columns):
            df.iloc[i]['Time_point'] = time_point


def generate_static_csv_from_df(df, path):
    """
    Generate static CSV files from a DataFrame based on different time points.

    :param df: Pandas DataFrame containing the data.
    :param path: Path where the generated CSV files will be saved.

    :return: None
    """
    if 'Days_from_relative_date' in df.columns:
        df.drop('Days_from_relative_date', axis=1, inplace=True)
    if 'Relative_date' in df.columns:
        df.drop('Relative_date', axis=1, inplace=True)
    for time_point in set(df.iloc[1:]['Time_point'].dropna()):
        df[df['Time_point'] == time_point].dropna(axis=1, how='all').drop('Time_point', axis=1).to_csv(
            path + str(int(time_point)) + '.csv', index=False)
