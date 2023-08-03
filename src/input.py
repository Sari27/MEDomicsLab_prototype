def convert_categorical_column_to_int(df, column_name):
    categories = df[column_name].unique()
    numbers = [_ for _ in range(len(categories))]
    df[column_name].replace(categories, numbers, inplace=True)


def set_time_point(df, class_name, time_point):
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
    for time_point in set(df.iloc[1:]['Time_point'].dropna()):
        df[df['Time_point'] == time_point].dropna(axis=1, how='all').drop('Time_point', axis=1).to_csv(
            path + str(int(time_point)) + '.csv', index=False)
