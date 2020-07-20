import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def replace_notes(row):
    """Replace NU_NOTA_MT if NU_NOTA_LC is zero."""
    if row['NU_NOTA_LC'] == 0: return np.nan
    return row['NU_NOTA_MT']

def split_dataset(dataframe):
    """Split feature from target."""
    target = dataframe['NU_NOTA_MT']
    features = dataframe.drop(['NU_NOTA_MT'], axis=1)
    return features, target

def main():
    df_train = pd.read_csv('train.csv', index_col=0)
    df_validation = pd.read_csv('test.csv')

    feat_columns = [
        'NU_INSCRICAO',
        'NU_NOTA_MT',
        'NU_NOTA_CH',
        'NU_NOTA_CN',
        'NU_NOTA_LC',
        'NU_NOTA_REDACAO',
        'NU_NOTA_COMP1',
        'NU_NOTA_COMP2',
        'NU_NOTA_COMP3',
        'NU_NOTA_COMP4',
        'NU_NOTA_COMP5',
    ]

    df_train = df_train.loc[:, feat_columns]
    df_train = df_train.drop(['NU_INSCRICAO'], axis=1)

    feat_columns.remove('NU_NOTA_MT')
    df_validation = df_validation.loc[:, feat_columns]

    # Replace missing data with 0
    df_train.fillna(value=0, inplace=True)
    df_validation.fillna(value=0, inplace=True)

    # Separate feature from target
    train_x, train_y = split_dataset(df_train)

    # y_data = df_train['NU_NOTA_MT']
    # x_data = df_train.drop(['NU_NOTA_MT'], axis=1)

    # Split dataset to train and test
    # train_x, x_test, train_y, y_test = train_test_split(x_data, y_data, test_size=0.25)

    # insc_train = train_x.loc[:, ['NU_INSCRICAO', 'NU_NOTA_LC']].copy()

    validation_data = df_validation.drop(['NU_INSCRICAO'], axis=1)

    regressor = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01).fit(train_x, train_y)
    validation_target = regressor.predict(validation_data)

    df_answer = pd.DataFrame({
        'NU_INSCRICAO': df_validation['NU_INSCRICAO'],
        'NU_NOTA_LC': df_validation['NU_NOTA_LC'],
        'NU_NOTA_MT': validation_target.clip(0)
    })

    df_answer['NU_NOTA_MT'] = df_answer.apply(replace_notes, axis=1)

    df_answer = df_answer.loc[:, ['NU_INSCRICAO', 'NU_NOTA_MT']]

    df_answer.to_csv('answer.csv', index=False)

if __name__ == '__main__':
    # pd.set_option('display.max_columns', 35)
    main()
