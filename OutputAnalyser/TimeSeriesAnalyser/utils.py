
from OutputAnalyser.TimeSeriesAnalyser import phq2int_dict
from seglearn.base import TS_Data
import numpy as np
from seglearn.util import check_ts_data
from seglearn.datasets import load_watch
import pandas as pd
from OutputAnalyser.TimeSeriesAnalyser import artifacts_dir


def get_demographics_str(folder, subject_ids):
    ages = []
    genders = []
    for subj in subject_ids:
        subj_folder = folder + "/" + subj
        fn = subj_folder + "/personalDetails.txt"
        with open(fn, "r") as f:
            dets = f.read()
            age, gender = int(dets.split("\n")[1].split("\t")[1]), dets.split("\n")[1].split("\t")[2].lower()
        ages.append(age)
        genders.append(gender)
    demographics_str = "{} males , {} females , {:.2f} mean age ({:.2f} std)".format(len([x for x in genders if x == "m"]), len([x for x in genders if x == "f"]),
                                                                                     np.mean(ages), np.std(ages))
    print(demographics_str)
    with open(artifacts_dir + "/demographics_str.txt", "w") as f:
        f.write(demographics_str)
    print("Saved demographics_str to ", artifacts_dir + "/demographics_str.txt")
    return demographics_str


def onehot_encode_categorical(df, c_cols):
    c = []
    for col in c_cols:
        one_hot_encoded_columns_df = pd.get_dummies(df[col])
        new_cols = [col + "_" + x for x in list(one_hot_encoded_columns_df.columns)]
        one_hot_encoded_columns_df.columns = new_cols
        df = pd.concat([df, one_hot_encoded_columns_df ], axis=1)
        c += new_cols
    return df, c

def df2seglearn_format(df, context_cols):

    ts = df.x_gaze_location_segment.apply(lambda x : np.array(x))
    Xts = list(ts)
    # Xts = list(df.x_gaze_location)#.apply(lambda x : np.array(eval(x))))
    Xc = df[context_cols].values
    t = TS_Data(Xts, Xc)
    check_ts_data(t)
    return t , Xts, ts


def df2sktime_format(df):
    pass

def split_df(df, X, y_col, dictify_y = False):
    train_idx = df[df.df_type == 'train'].index
    test_idx = df[df.df_type.isin(['val','test'])].index

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]

    y_train = df.loc[train_idx][y_col]
    y_test = df.loc[test_idx][y_col]
    if dictify_y:
        y_train = y_train.apply(lambda x : phq2int_dict[x])
        y_test = y_test.apply(lambda x : phq2int_dict[x])

    return X_train, X_test, y_train, y_test, train_idx, test_idx