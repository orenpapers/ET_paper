from OutputAnalyser.TimeSeriesAnalyser import split_df
from OutputAnalyser.TimeSeriesAnalyser import phq2int_dict
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import numpy as np
import pandas as pd
import time


def _scale_ts(dataset):
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
    dataset = scaler.fit_transform(dataset)
    return dataset

def _early_classification(df, formatted_dataset, contextual_cols = None, y_col = "phq_group"):
    #https://tslearn.readthedocs.io/en/stable/auto_examples/classification/plot_early_classification.html#sphx-glr-auto-examples-classification-plot-early-classification-py
    X = np.where(np.isnan(formatted_dataset), 0, formatted_dataset)
    X_train , X_test , y_train , y_test , _ , _= split_df(df, X, y_col, dictify_y = False)

    early_clf = NonMyopicEarlyClassifier(n_clusters=5,
                                         cost_time_parameter=1e-3,
                                         lamb=1e2,
                                         random_state=0)
    start = time.time()

    X_train_vals = np.concatenate(X_train , axis=1).transpose()
    early_clf.fit(X_train_vals, y_train.values)
    end = time.time()
    score = early_clf.score(X_test, y_test)
    print("RAW (early) - Accuracy score: ", score)
    print("RAW (early) - Classification time : " , end - start)
    preds, times = early_clf.predict_class_and_earliness(X_test)
    return preds, times, score , early_clf

def _kneighbours_classification(df, formatted_dataset, contextual_cols = None, y_col = "binary_label"):
    # https://tslearn.readthedocs.io/en/stable/auto_examples/neighbors/plot_neighbors.html#sphx-glr-auto-examples-neighbors-plot-neighbors-py
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=5)
    X = np.where(np.isnan(formatted_dataset), 0, formatted_dataset)
    X_train , X_test , y_train , y_test , _ , _= split_df(df, X, y_col, dictify_y = False)
    start = time.time()
    X_train_vals = np.concatenate(X_train , axis=1).transpose()
    knn.fit(X_train_vals, y_train.values)
    end = time.time()
    # knn.kneighbors(X2)  # Search for neighbors using series from `X2` as queries
    score = knn.score(X_test, y_test)

    print("RAW (KN) - Accuracy score: ", score)
    print("RAW (KN) - Classification time : " , end - start)
    # y_pred_probas = knn.predict_proba(X_test)
    return knn, score


def raw_classification_manager(sen_type_df, series_of_timeseries, contextual_cols, algs):
    from seglearn.transform import PadTrunc
    from tslearn.utils import to_time_series_dataset
    y = [phq2int_dict[x] for x in list(sen_type_df.phq_group)]
    formatted_dataset = to_time_series_dataset(list(series_of_timeseries))

    formatted_dataset_padded_X, y, _ = PadTrunc(width = 500).fit_transform(formatted_dataset, y)
    # formatted_dataset_padded_X, y = formatted_dataset_padded

    formatted_dataset_padded_scaled = _scale_ts(formatted_dataset_padded_X)
    sc_dict = {}
    m_dict = {}
    sentype_df_reindexed = sen_type_df.reset_index().rename({"index" : "orig_index"}, axis=1)

    if "kn" in algs:
        kn_model, kn_score = _kneighbours_classification(df = sentype_df_reindexed, formatted_dataset = formatted_dataset_padded_scaled,
                                         contextual_cols = contextual_cols)
        sc_dict["kn"] = kn_score
        m_dict["kn"] = kn_model

    if "early" in algs:
        early_preds, early_times, early_score , early_model = _early_classification(df = sentype_df_reindexed,
                                                                  formatted_dataset = formatted_dataset_padded_scaled,
                                                                  contextual_cols = None)
        sc_dict["early"] = early_score
        m_dict["early"] = early_model


    if "dtw" in algs:
        #todo implement https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html
        #https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3
        #https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
        pass

    if "hmm" in algs:
        pass
        #todo add DTW , HMM
    return sc_dict, m_dict