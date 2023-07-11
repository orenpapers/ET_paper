import sys, time
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
import os
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.feature_selection import SelectKBest, f_classif
from OutputsAnalyser.run_timepoints_utils import get_timecols_df
from params import analysis_artifacts_dir
import random
from sklearn.decomposition import FastICA

n_CV = 25
np.random.seed(42)
Y_COL = "phq_score"
random.seed(42)
list_of_snapshot_trans = [
    FastICA(n_components=60, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=80, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=30, tol=0.3, max_iter=500, random_state=42),
    # FastICA(n_components=120, tol=0.3, max_iter=500, random_state=42),
    # FastICA(n_components=25, tol=0.3, max_iter=500, random_state=42),
    SelectKBest(k=150), SelectKBest(k=60), SelectKBest(k=90)#, SelectKBest(k=120), SelectKBest(k=30),
    #     SelectKBest(k=150),

]
list_of_snapshot_models = [
    RandomForestRegressor(n_estimators=300, random_state=42),
    # RandomForestRegressor(n_estimators=250, random_state=42),
    RandomForestRegressor(n_estimators=200, random_state=42),
    # RandomForestRegressor(n_estimators=150, random_state=42),
    # RandomForestRegressor(n_estimators=100, random_state=42),
    RandomForestRegressor(n_estimators=50, random_state=42),
]

SCALERS = [None, StandardScaler(), MinMaxScaler(), Normalizer()]#, MinMaxScaler(), QuantileTransformer(n_quantiles=60)] #todo add no scaler
ET_SCALE_COLS = ["x_gaze_location_normalized","x_gaze_location_standard_scaled"]#,"x_gaze_location_minmax_scaled","x_gaze_location_q_scaled","x_gaze_location_r_scaled"]
METHODS = ["mean_all"]#,"mean_not_F"]

DEMO = True if "DEMO" in sys.argv else False
PERM = True if "PERM" in sys.argv else False

if DEMO:
    list_of_snapshot_models = list_of_snapshot_models[:2]
    list_of_snapshot_trans = list_of_snapshot_trans[:2]
    SCALERS = SCALERS[:2]
    SCALE_COLS = ET_SCALE_COLS[:2]
    METHODS = METHODS[:2]
    n_CV = 3

def train_regression(regression_pipeline, df, train_subjects, test_subjects,  X_cols, y_col):

    df_train = df[df.Subject.isin(train_subjects)].reset_index(drop=True).dropna()
    df_test = df[df.Subject.isin(test_subjects)].reset_index(drop=True).dropna()

    X_train, X_test = df_train[X_cols], df_test[X_cols]
    y_train, y_test = df_train[y_col], df_test[y_col]
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    regression_pipeline.fit(X_train.astype('float32'), y_train.astype('float32'))
    y_pred = [x for x in regression_pipeline.predict(X_test)]
    test_acc = regression_pipeline.score(X_test, y_test)
    train_acc = regression_pipeline.score(X_train, y_train)
    # y_proba = [x[0] for x in regression_pipeline.predict_proba(X_test)]

    return regression_pipeline, y_pred, y_test, test_acc, train_acc


def train_subject_level_classifier_per_cond(df, trs , clfs, c, verbose = 25):

    res_cols = ["cv_i","scaler", "transformations", "classifier", "sentence_type","test_acc","train_acc","y_pred",
                "y_true", "test_subjects", "train_subjects", "time_to_train"]
    pipeline_idx = 0
    res_rows = []
    total_pipelines = len(SCALERS) * len(trs) * len(clfs)
    total_configs =  len(ET_SCALE_COLS) * len(METHODS)
    for sc in SCALERS:
        for tr in trs:
            for est in clfs:
                for cv_i in range(n_CV):

                    test_subjects = sorted(list(random.sample(list(all_subjects), k=int(len(all_subjects) * 0.2))))
                    train_subjects = [x for x in all_subjects if x not in test_subjects]
                    if (cv_i % verbose) == 0:
                        print(f"{datetime.datetime.now()},pipeline #{pipeline_idx}/{total_pipelines} (of config #{c}/{total_configs}):"
                              f"  sc = {sc}, tr={tr}, est={est}, fold = {cv_i}/{n_CV}")

                    for sen_type, sen_df in df.groupby(["Sentence_type"]):
                        classification_pipeline = make_pipeline(sc , tr, est)
                        try:
                            b = time.time()
                            fitted_classification_pipeline, y_pred, y_true, test_acc, train_acc = train_regression(classification_pipeline, sen_df,
                                                                                                                               train_subjects,  test_subjects,
                                                                                                                               X_cols = timepoint_cols, y_col = Y_COL,
                                                                                                                               )
                            e = time.time()
                            row = [cv_i, str(sc).split("()")[0], str(tr).split("()")[0] , str(est).split("()")[0],
                                   sen_type, test_acc, train_acc, list(y_pred), list(y_true),
                                   test_subjects, train_subjects, round(e - b, 2)]

                            res_rows.append(row)
                        except Exception as e:
                            print(f"Failed!! <{sc},{tr},{est}> - {e}")
                pipeline_idx +=1

    per_fold_per_cond_per_config_res_df = pd.DataFrame(data = res_rows, columns = res_cols). \
        sort_values(by=['scaler','transformations','classifier','cv_i','sentence_type']).reset_index(drop=True)
    print("Shape of per_fold_per_cond_per_config_res_df is ", per_fold_per_cond_per_config_res_df.shape)
    return per_fold_per_cond_per_config_res_df


def plot_subject_level_regression_results(analysis_artifacts_output_dir,  y_true, y_pred):


    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'ground truth', 'y': 'prediction'})
    fig.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y_true.min(), y0=y_true.min(),
    x1=y_true.max(), y1=y_true.max()
    )
    # fig.show()
    fig.write_image(analysis_artifacts_output_dir + "actual_VS_predicted.png")
    return

def analyse_config_results(analysis_artifacts_output_dir, config_df, df, pipeline_idx, method, et_scale_col, title_suf):

    def combine_conditions_to_unified_predictions_subject_prediction(cv_df , method):
        if method=='mean_all':
            return list(pd.DataFrame(cv_df.y_pred.tolist()).mean())
        if method=='mean_not_F':
            return list(pd.DataFrame(cv_df[cv_df.sentence_type != 'F'].y_pred.tolist()).mean())

    config_rows = []
    per_fold_acc = []

    for cv_num, cv_df in config_df.groupby(["cv_i"]):
        per_subject_unified_prediction = combine_conditions_to_unified_predictions_subject_prediction(cv_df, method)

        cv_test_subjects = list(cv_df.test_subjects.iloc[0])
        mses = []
        for subj_i, subj_num in enumerate(cv_test_subjects):
            subj_label = df[df.Subject == subj_num][Y_COL].iloc[0]
            subj_pred =  per_subject_unified_prediction[subj_i]
            subj_mse = MSE([subj_label], [subj_pred])

            mses.append(subj_mse)
            row = [cv_num, subj_num, per_subject_unified_prediction[subj_i], subj_label, subj_mse]
            config_rows.append(row)

        fold_acc = np.mean(mses)
        per_fold_acc.append(fold_acc)

    subj_agg_res_df = pd.DataFrame(data = config_rows, columns = ["cv_num", "test_subject", "prediction","true_label", "subj_MSE"])

    y_true = subj_agg_res_df.true_label
    y_pred = subj_agg_res_df.prediction

    plot_subject_level_regression_results(analysis_artifacts_output_dir,  y_true, y_pred)

    return subj_agg_res_df, per_fold_acc


def get_per_config_metrics(analysis_artifacts_output_dir, subject_level_res_df, df, method, et_scale_col,
                           ):
    per_model_subj_res_rows = []
    pipeline_idx = 0
    for config, config_df in subject_level_res_df.groupby(["scaler","transformations","classifier"]):

        subj_agg_res_df,  per_fold_acc = analyse_config_results(
            analysis_artifacts_output_dir, config_df , df, pipeline_idx, method, et_scale_col, title_suf = ",".join(config))

        per_model_subj_res_rows.append([pipeline_idx, method,
                                        config[0], config[1],config[2], np.mean(per_fold_acc), np.std(per_fold_acc),
                                        per_fold_acc])

        pipeline_idx += 1

    per_config_metrics_df = pd.DataFrame(data = per_model_subj_res_rows,
                                         columns = ["model_idx","method", "local_scaler",
                                                    "tr","cls",
                                                    "MSE_mean_across_fold", "MSE_std_across_fold",
                                                    "per_fold_score"])


    return per_config_metrics_df


perm_benchmark_df = pd.read_csv("/Users/orenkobo/Desktop/PhD/Aim1/Analysis_artifacts/1637007626_all_res_df_iterall_REG_PERM.csv",
                                converters={"per_fold_score" : eval})
demo_suffix = "_DEMO" if DEMO else ""
perm_suffix = "_PERM" if PERM else ""
desc_suffix = "_iterall"
suffix = f"{desc_suffix}_REG{demo_suffix}{perm_suffix}"
print("SUFFIX IS ", suffix)
print(f"=====\nRun: \n\t{list_of_snapshot_trans} , \n\t{list_of_snapshot_models}, \n\t{ET_SCALE_COLS}, \n\t{METHODS}, \n\t{SCALERS}=====\n")
timestamp = int(time.time())
analysis_artifacts_dir = f"{analysis_artifacts_dir}"
os.system(f"mkdir {analysis_artifacts_dir}/{timestamp}{suffix}")
all_res_dfs = []
all_res_per_cv  = []

df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL.csv"
c = 0

for et_scale_col in ET_SCALE_COLS:
    scale_prefix = et_scale_col[16:]

    print(f"{datetime.datetime.now()} Read df from {df_fn} col is {et_scale_col}")
    all_subjs_per_subject_settype_mean_df, timepoint_cols = get_timecols_df(fn =df_fn, scale_col = et_scale_col)
    print(f"{datetime.datetime.now()} Got df with shape {all_subjs_per_subject_settype_mean_df.shape} , {len(timepoint_cols)} time cols")
    for method in METHODS:
        print(f"{datetime.datetime.now()}: ================= Work on{method}, {et_scale_col} ================= ")
        df = all_subjs_per_subject_settype_mean_df.copy()
        if PERM:
            df[Y_COL] = df[Y_COL].sample(frac=1, random_state = 42).reset_index(drop=True)

        all_subjects = df.Subject.unique()

        analysis_artifacts_output_dir = f"{analysis_artifacts_dir}/{timestamp}{suffix}/{suffix}/"

        os.system(f"mkdir {analysis_artifacts_output_dir}")

        mode_artifacts_output_dir = analysis_artifacts_output_dir + "snapshot/"
        os.system(f"mkdir {mode_artifacts_output_dir}")
        os.system(f"mkdir {mode_artifacts_output_dir}/pngs")

        snapshot_per_fold_per_cond_per_confid_res_df = train_subject_level_classifier_per_cond(df,
                                                                                               list_of_snapshot_trans ,
                                                                                               list_of_snapshot_models,
                                                                                               c,
                                                                                               )


        snapshot_per_model_subj_res_df = get_per_config_metrics(mode_artifacts_output_dir,
                                                                snapshot_per_fold_per_cond_per_confid_res_df,
                                                                df, method, scale_prefix,
                                                                )



        snapshot_per_fold_per_cond_per_confid_res_df["mode"] = "snapshot"
        snapshot_per_fold_per_cond_per_confid_res_df["ET_scale_col"] = scale_prefix
        snapshot_per_model_subj_res_df["mode"] = "snapshot"
        snapshot_per_model_subj_res_df["ET_scale_col"] = scale_prefix
        snapshot_per_model_subj_res_df["num_folds"] = n_CV

        snapshot_per_fold_per_cond_per_confid_res_df.to_csv(f"{mode_artifacts_output_dir}/{suffix}{method}_snapshot_per_fold_per_cond_per_confid_res_df.csv", index = False)
        snapshot_per_model_subj_res_df.to_csv(f"{mode_artifacts_output_dir}/{suffix}{method}_snapshot_per_model_subj_res_df.csv", index = False)

        print(f"{datetime.datetime.now()}: Saved <{method},{et_scale_col}> results to {mode_artifacts_output_dir}")

        all_res_dfs.append(snapshot_per_model_subj_res_df)
        all_res_per_cv.append(snapshot_per_fold_per_cond_per_confid_res_df)

        c += 1

all_res_df = pd.concat(all_res_dfs, axis=0)
all_res_per_cv_df = pd.concat(all_res_per_cv, axis=0)
all_res_df.to_csv(f"{analysis_artifacts_dir}/{timestamp}_all_res_df{suffix}.csv", index = False)
all_res_per_cv_df.to_csv(f"{analysis_artifacts_dir}/{timestamp}_all_res_per_cv_df{suffix}.csv", index = False)

if not PERM:
    try:
        per_config_metrics_df_with_perm_benchmark_df = all_res_df.merge(right = perm_benchmark_df,
                                                                        on = ['local_scaler','tr','cls',
                                                                              'method','ET_scale_col'],
                                                                        suffixes = ("", "_PERM"))

        per_config_metrics_df_with_perm_benchmark_df["temp"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : x.per_fold_score_PERM + [x.actual_mean_success_across_folds], axis=1)
        per_config_metrics_df_with_perm_benchmark_df["classifier_rank"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : sorted(x.temp, reverse=True).index(x.actual_mean_success_across_folds), axis=1)
        per_config_metrics_df_with_perm_benchmark_df["classifier_pval"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : x.classifier_rank / x.num_folds, axis=1)
        merged_out_fn = f"{analysis_artifacts_dir}/{timestamp}_all_res_df_with_perm{suffix}.csv"
        per_config_metrics_df_with_perm_benchmark_df.to_csv(merged_out_fn, index = False)
        print("Merged with benchmark and saved to ", merged_out_fn)
    except Exception as e:
        print(f"Error {e} when merging with benchmark , cant get p val (probably some configuration missing from benmark file in {perm_benchmark_df}")
#