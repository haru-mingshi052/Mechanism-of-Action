import pandas as pd

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train_features.csv')
    test = pd.read_csv(data_folder + '/test_features.csv')

    return train, test

def read_target(data_folder):
    target = pd.read_csv(data_folder + '/train_targets_scored.csv')

    return target

def read_submission(data_folder):
    sub = pd.read_csv(data_folder + '/sample_submission.csv')

    return sub

def categorize(train, test):
    df = pd.concat([train, test])

    #cp_typeのone_hot_encoder
    type_dummy = pd.get_dummies(df['cp_type'], prefix = 'type')
    train = pd.concat([train, type_dummy.iloc[:len(train),:]], axis = 1)
    test = pd.concat([test, type_dummy.iloc[len(train):,:]], axis = 1)

    #cp_timeのone_hot_encoder
    time_dummy = pd.get_dummies(df['cp_time'], prefix = 'time')
    train = pd.concat([train, time_dummy.iloc[:len(train),:]], axis = 1)
    test = pd.concat([test, time_dummy.iloc[len(train):,:]], axis = 1)

    #cp_doseのone_hot_encoder
    dose_dummy = pd.get_dummies(df['cp_dose'], prefix = 'dose')
    train = pd.concat([train, dose_dummy.iloc[:len(train),:]], axis = 1)
    test = pd.concat([test, dose_dummy.iloc[len(train):,:]], axis = 1)

    drop_list = ['cp_type', 'cp_time', 'cp_dose']
    train.drop(drop_list, axis = 1, inplace = True)
    test.drop(drop_list, axis = 1, inplace = True)

    return train, test

def make_sub(sub, pred):
    target_cols = [c for c in sub.columns]
    sub = pd.concat([sub['sig_id'], pred], axis = 1)
    sub.columns = target_cols

    return sub

def processing_df(data_folder):
    train, test = read_data(data_folder)
    target = read_target(data_folder)
    train, test = categorize(train, test)

    return train, test, target