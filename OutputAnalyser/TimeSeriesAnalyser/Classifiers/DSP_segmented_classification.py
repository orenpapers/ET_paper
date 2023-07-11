from seglearn.base import TS_Data
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, Segment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

from OutputAnalyser.TimeSeriesAnalyser import split_df


def _classify_contextual(df, series_of_timeseries, contextual_cols, y_col ='binary_label'):
    #https://dmbee.github.io/seglearn/auto_examples/plot_feature_rep.html#sphx-glr-auto-examples-plot-feature-rep-py
    clf = Pype([('segment', Segment()),
                ('features', FeatureRep()),
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(n_estimators=20))])

    X_train , X_test, y_train, y_test, train_idx, test_idx = split_df(df, series_of_timeseries, y_col)

    if contextual_cols:
        contextual_data_df = df[contextual_cols]

        X_train_C = contextual_data_df.loc[train_idx]
        X_test_C = contextual_data_df.loc[test_idx]

        X_train = TS_Data(np.array(X_train), np.array(X_train_C))
        X_test = TS_Data(np.array(X_test), np.array(X_test_C))
    else:
        #todo implement
        return {},{}

    print("DSP - N series in train: ", len(X_train), len(y_train))
    print("DSP - N series in test: ", len(X_test), len(y_test))

    start = time.time()

    # return X_train, X_train_C, X_train_S
    clf.fit(X_train, np.array(y_train))
    end = time.time()

    score = clf.score(X_test, np.array(y_test))

    print("DSP - N segments in train: ", clf.N_train)
    print("DSP - N segments in test: ", clf.N_test)
    print("DSP - Accuracy score: ", score)
    print("DSP - Classification time : " , end - start)
    return score , clf


def dsp_classification_manager(sen_type_df, series_of_timeseries, contextual_cols):
    sc_dict = {}
    models_dict = {}

    sc, dsp_model_non_contextual = _classify_contextual(df = sen_type_df, series_of_timeseries = series_of_timeseries,
                                          contextual_cols = None)
    sc_contextual, dsp_model_contextual = _classify_contextual(df = sen_type_df, series_of_timeseries = series_of_timeseries,
                              contextual_cols = contextual_cols)

    sc_dict["dsp_segmented"] = sc
    sc_dict["dsp_segmented_contextual"] = sc_contextual
    models_dict["dsp_segmented_no_context"] = dsp_model_non_contextual
    models_dict["dsp_segmented_contextual"] = dsp_model_contextual
    return sc_dict, models_dict