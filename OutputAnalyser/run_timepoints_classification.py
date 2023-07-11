import sys, time
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import datetime
import plotly.io as pio
from tslearn.utils import to_time_series
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, auc
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from OutputsAnalyser.run_timepoints_utils import get_timecols_df, apply_phq_cutoff, get_timecols_df_for_DL

from params import analysis_artifacts_dir, perm_benchmark_df_fn, df_fn
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import brier_score_loss

import random
n_CV = 500
np.random.seed(42)
Y_COL = "phq_binary_label"
random.seed(42)
# print(random.random())
list_of_snapshot_trans = [
    # FastICA(n_components=80, tol=0.3, max_iter=500, random_state=42),
    # FastICA(n_components=60, tol=0.3, max_iter=500, random_state=42),
    # FastICA(n_components=30, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=25, tol=0.3, max_iter=500, random_state=42),
    # SelectKBest(k=150),
    # SelectKBest(k=120),
    # SelectKBest(k=90),
    # SelectKBest(k=60),
    # SelectKBest(k=30),

]
list_of_snapshot_models = [
    # CatBoostClassifier(random_state=42, n_estimators=100),
    #
    # XGBClassifier(random_state=42),
    # LGBMClassifier(verbose=0, random_state=42),
    ExtraTreesClassifier(n_estimators=300, random_state=42),
    # ExtraTreesClassifier(n_estimators=250, random_state=42),
    # ExtraTreesClassifier(n_estimators=200, random_state=42),
    # ExtraTreesClassifier(n_estimators=150, random_state=42),
    # ExtraTreesClassifier(n_estimators=100, random_state=42),
    # ExtraTreesClassifier(n_estimators=50, random_state=42),
    RandomForestClassifier(n_estimators=300, random_state=42),
    # RandomForestClassifier(n_estimators=250, random_state=42),
    # RandomForestClassifier(n_estimators=200, random_state=42),
    # RandomForestClassifier(n_estimators=150, random_state=42),
    # RandomForestClassifier(n_estimators=100, random_state=42),
    # RandomForestClassifier(n_estimators=50, random_state=42),
]
# list_of_snapshot_trans = [
#                         SelectKBest(k=120)
#                           # FastICA(n_components=25, tol=0.3, max_iter=500, random_state=42),
#                         #  SelectKBest(k=120), SelectKBest(k=90), SelectKBest(k=60), SelectKBest(k=30),
#                      #     SelectKBest(k=150),
#
#                           ]
#
# list_of_snapshot_models = [RandomForestClassifier(n_estimators=200, random_state=42)
#                            #BaggingClassifier(random_state=42),
#                            # ExtraTreesClassifier(n_estimators=150, random_state=42),
#     #                        RandomForestClassifier(n_estimators=250, random_state=42),
#     # RandomForestClassifier(n_estimators=200, random_state=42), RandomForestClassifier(n_estimators=150, random_state=42),
#     # RandomForestClassifier(n_estimators=100, random_state=42), RandomForestClassifier(n_estimators=50, random_state=42)

#                             ]


SCALERS = [None, MinMaxScaler()]#, StandardScaler()]#, Normalizer()]#, MinMaxScaler()]#, QuantileTransformer(n_quantiles=60)]
# SCALERS = [None, StandardScaler()]
ET_SCALE_COLS = ["x_gaze_location_standard_scaled", "x_gaze_location_normalized"]#,"x_gaze_location_standard_scaled","x_gaze_location_minmax_scaled"]#,"x_gaze_location_q_scaled","x_gaze_location_r_scaled"]
# METHODS = ["mean_all","mean_not_A", "take_max_cond", "take_violation"]
# METHODS = ["mean_all","take_violation"]
METHODS = ["mean_all"]#,'mean_not_F','take_max_cond']
CUTOFFS = [[7,8]]#,[6,7]]#,[4,14],[5,8],[5,7],[4,8]]:
N_SUBJ_OUT = 0.2
DEMO = True if "DEMO" in sys.argv else False
PERM_RUN = True if "PERM" in sys.argv else False
MULTILABEL = True if "multilabel" in sys.argv else False

if PERM_RUN:
    n_CV *= 10
if MULTILABEL:
    Y_COL = "phq_group"
    METHODS = ["max_vote","mean_vote"]
if DEMO:
    list_of_snapshot_models = list_of_snapshot_models[:2]
    list_of_snapshot_trans = list_of_snapshot_trans[:2]
    SCALERS = SCALERS[:2]
    SCALE_COLS = ET_SCALE_COLS[:2]
    METHODS = METHODS[:2]
    CUTOFFS = CUTOFFS[:2]
    n_CV = 3

def train_classifiction(classification_pipeline, df, train_subjects, test_subjects,  X_cols, y_col, mode):

    df_train = df[df.Subject.isin(train_subjects)].reset_index(drop=True).dropna()
    df_test = df[df.Subject.isin(test_subjects)].reset_index(drop=True).dropna()

    X_train, X_test = df_train[X_cols], df_test[X_cols]
    y_train, y_test = df_train[y_col], df_test[y_col]
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    if mode == 'timeseries':
        X_train = to_time_series(X_train)
        X_test = to_time_series(X_test)


    classification_pipeline.fit(X_train.astype('float32'), y_train.astype('float32')) #X shape is 81
    y_pred = [x for x in classification_pipeline.predict(X_test)]
    test_acc = classification_pipeline.score(X_test, y_test)
    train_acc = classification_pipeline.score(X_train, y_train)
    probs = classification_pipeline.predict_proba(X_test)
    y_proba = probs if MULTILABEL else [x[0] for x in classification_pipeline.predict_proba(X_test)]

    return classification_pipeline, y_pred, y_test, y_proba, test_acc, train_acc


def train_subject_level_classifier_per_cond(df, trs , clfs, c, mode , verbose = 25):

    res_cols = ["cv_i","scaler", "transformations", "classifier", "sentence_type","test_acc","train_acc","y_pred","y_pred_proba",
                "y_true", "test_subjects", "train_subjects", "time_to_train"]
    pipeline_idx = 0
    res_rows = []
    total_pipelines = len(SCALERS) * len(trs) * len(clfs)  #(2 is for is_perm in [True,False]
    total_configs =  len(ET_SCALE_COLS) * len(METHODS) * len(CUTOFFS) * 2
    print(f"{datetime.datetime.now()}: OVERALL {total_configs * total_pipelines} iters")
    for sc in SCALERS:
        for tr in trs:
            for est in clfs:
                for cv_i in range(n_CV):

                    test_subjects = sorted(list(random.sample(list(all_subjects), k=int(len(all_subjects) * N_SUBJ_OUT))))
                    train_subjects = [x for x in all_subjects if x not in test_subjects]
                    if (cv_i % verbose) == 0:
                        print(f"{datetime.datetime.now()},pipeline #{pipeline_idx}/{total_pipelines} (of config #{c}/{total_configs}):"
                              f"  sc = {sc}, tr={tr}, est={est}, fold = {cv_i}/{n_CV}")

                    for sen_type, sen_df in df.groupby(["Sentence_type"]):
                        classification_pipeline = make_pipeline(sc , tr, est)
                        try:
                            bt = time.time()
                            fitted_classification_pipeline, y_pred, y_true, y_proba, test_acc, train_acc = train_classifiction(classification_pipeline, sen_df,
                                                                                                                               train_subjects,  test_subjects,
                                                                                                                               X_cols = timepoint_cols, y_col = Y_COL,
                                                                                                                               mode = mode
                                                                                                                               )
                            et = time.time()
                            row = [cv_i, str(sc).split("()")[0], str(tr).split("()")[0] , str(est).split("()")[0],
                                   sen_type, test_acc, train_acc, list(y_pred), y_proba, list(y_true),
                                   test_subjects, train_subjects, round(et - bt, 2)]

                            res_rows.append(row)
                        except Exception as e:
                            print(f"Failed!! <{sc},{tr},{est}> - {e}")
                pipeline_idx +=1

    per_fold_per_cond_per_config_res_df = pd.DataFrame(data = res_rows, columns = res_cols).\
        sort_values(by=['scaler','transformations','classifier','cv_i','sentence_type']).reset_index(drop=True)
    print("Shape of per_fold_per_cond_per_config_res_df is ", per_fold_per_cond_per_config_res_df.shape)
    return per_fold_per_cond_per_config_res_df

def plot_subject_level_classification_results(analysis_artifacts_output_dir, method, et_scale_col, y_true, y_pred,
                                              cv_scores, chance_scores, pipeline_idx, title_suf):

    desc = f"{method}_{et_scale_col}"
    try:
        cm = confusion_matrix(y_true, [round(x) for x in y_pred])
        f = sns.heatmap(cm, annot=True, fmt='d')
        f.get_figure().savefig(f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__confusionmatrix.png")
    except Exception as e:
        print("Cant plot CM - ",e)
    if MULTILABEL:
        return np.nan, np.nan

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(4, 3))
    plt.boxplot([cv_scores])
    plt.axhline(np.mean(chance_scores))
    plt.xticks([1], ['Actual'])
    plt.title('Prediction: accuracy score')
    plt.savefig(f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__acc.png")
    print(f"{datetime.datetime.now()} Saved graph to {analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}")
    plt.cla()


    plt.figure(figsize=(4, 3))
    plt.boxplot([cv_scores, chance_scores])
    # plt.scatter([null_cv_scores], color = 'blue')
    plt.xticks([1, 2], ['Actual', 'chance'])
    plt.title('Prediction: accuracy score VS. chance level')
    plt.savefig(f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__accVSchance.png")
    print(f"{datetime.datetime.now()} Saved graph to {analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}")
    plt.cla()
    # Evaluating model performance at various thresholds
    df = pd.DataFrame({'False Positive Rate': fpr,'True Positive Rate': tpr}, index=roc_thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"

    plt.cla()
    sns.violinplot(x=y_true, y=y_pred, inner='points',bw=.05)
    plt.title("Distribution of prediction per actual label")
    plt.savefig(f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__prediction_per_label.png")
    plt.cla()


    fig = px.area( x=recall, y=precision, title=f'Precision-Recall Curve : ' + title_suf, labels=dict(x='Recall', y='Precision'),)
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=1, y1=0)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    pio.write_image(fig, f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__PR.png")
    plt.cla()

    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)

    fig.add_trace(go.Scatter(x=fpr, y=tpr,  mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        # width=700, height=500
    )
    pio.write_image(fig, f"{analysis_artifacts_output_dir}/pngs/pipeline#{pipeline_idx}_{desc}__FPR_TPR.png")
    plt.cla()


    return roc_auc, pr_auc

def combine_conditions_to_unified_predictions_subject_prediction(cv_df , method):
    if method=='mean_all':
        return list(pd.DataFrame(cv_df.y_pred_proba.tolist()).mean())
    if method=='mean_not_A':
        return list(pd.DataFrame(cv_df[cv_df.sentence_type != 'A'].y_pred_proba.tolist()).mean())
    if method=='only_F':
        return list(pd.DataFrame(cv_df[cv_df.sentence_type == 'F'].y_pred_proba.tolist()).mean())
    if method=='mean_not_F':
        return list(pd.DataFrame(cv_df[cv_df.sentence_type != 'F'].y_pred_proba.tolist()).mean())
    if method=='take_max_cond':
        return list(pd.DataFrame(cv_df[cv_df.sentence_type != 'F'].y_pred_proba.tolist()).max())
    if method=='take_violation':
        return list(pd.DataFrame(cv_df[cv_df.sentence_type.isin(['B','D'])].y_pred_proba.tolist()).mean())
    #todo try max vote also and weighted mean and mean_notA

def combine_conditions_to_unified_predictions_subject_prediction_MULTILABEL(cv_df , method):
    if method=='max_vote':
        return list(pd.DataFrame(cv_df.y_pred.tolist()).max())
    if method=='mean_vote':
        return [round(x) for x in list(pd.DataFrame(cv_df.y_pred.tolist()).mean())]


def analyse_config_results(analysis_artifacts_output_dir, config_df, df, pipeline_idx, method, et_scale_col, title_suf,
                           is_perm_iter):

    config_rows = []
    per_fold_acc = []
    chance_scores = []

    combine_func = combine_conditions_to_unified_predictions_subject_prediction_MULTILABEL if MULTILABEL else combine_conditions_to_unified_predictions_subject_prediction
    # method = "mean_all" if take_f else "mean_not_F"
    for cv_num, cv_df in config_df.groupby(["cv_i"]):
        per_subject_unified_prediction = combine_func(cv_df, method)

        cv_test_subjects = list(cv_df.test_subjects.iloc[0])
        num_success_subj = 0
        for subj_i, subj_num in enumerate(cv_test_subjects):
            subj_label = df[df.Subject == subj_num][Y_COL].iloc[0]
            subj_success = True if round(per_subject_unified_prediction[subj_i]) == subj_label else False
            if subj_success:
                num_success_subj +=1
            row = [cv_num, subj_num, per_subject_unified_prediction[subj_i], subj_label, subj_success] + \
                                             [cv_df[cv_df.sentence_type==x]['test_acc'].iloc[0] for x in ['A','B','C','D',"F"]]
            config_rows.append(row)

        fold_acc = num_success_subj / len(cv_test_subjects)
        per_fold_acc.append(fold_acc)
        chance_scores.append(sum([df[df.Subject == x][Y_COL].iloc[0] for x in cv_test_subjects]) / len(cv_test_subjects))

    subj_agg_res_df = pd.DataFrame(data = config_rows, columns = ["cv_num", "test_subject", "prediction","true_label",
                                                                  "subj_success"] + [f"{x}_success" for x in ['A','B','C','D',"F"]])

    y_true = subj_agg_res_df.true_label
    y_pred = subj_agg_res_df.prediction

    mean_acc = subj_agg_res_df.subj_success.mean()
    std_acc = subj_agg_res_df.subj_success.std()
    var_acc = subj_agg_res_df.subj_success.var()
    A_acc = subj_agg_res_df.A_success.mean()
    B_acc = subj_agg_res_df.B_success.mean()
    C_acc = subj_agg_res_df.C_success.mean()
    D_acc = subj_agg_res_df.D_success.mean()
    F_acc = subj_agg_res_df.F_success.mean()

    if MULTILABEL:
        f1 = np.nan
        br = np.nan
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

    else:
        f1 = f1_score(y_true, [round(x) for x in y_pred])
        br = brier_score_loss(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, [round(x) for x in y_pred]).ravel()

    if is_perm_iter:
        roc_auc, pr_auc = np.nan, np.nan
    else:
        roc_auc, pr_auc = plot_subject_level_classification_results(analysis_artifacts_output_dir, method, et_scale_col,
                                                                y_true, y_pred, per_fold_acc, chance_scores, pipeline_idx,
                                                                title_suf)

    return subj_agg_res_df, f1, br, roc_auc, pr_auc, mean_acc, std_acc, var_acc, tn, fp, fn, tp, chance_scores, per_fold_acc, \
           A_acc, B_acc, C_acc, D_acc, F_acc


def get_per_config_metrics(analysis_artifacts_output_dir, subject_level_res_df, df, method, et_scale_col,
                           override_cutoff, is_perm_iter):
    per_model_subj_res_rows = []
    pipeline_idx = 0
    labeled_0 = len(df[df[Y_COL] == 0]) / 5
    labeled_1 = len(df[df[Y_COL] == 1]) / 5
    for config, config_df in subject_level_res_df.groupby(["scaler","transformations","classifier"]):

        subj_agg_res_df, f1, br, roc_auc, pr_auc, mean_acc, std_acc, var_acc, tn, fp, fn, tp, chance_scores, per_fold_acc, A_acc, B_acc, C_acc, D_acc,F_acc, \
            = analyse_config_results(analysis_artifacts_output_dir, config_df , df, pipeline_idx, method, et_scale_col,
                                     title_suf = ",".join(config) , is_perm_iter=is_perm_iter)

        per_model_subj_res_rows.append([pipeline_idx, method, override_cutoff[0], override_cutoff[1],labeled_0, labeled_1,
                                        config[0], config[1],config[2], f1, br, roc_auc,
                                        tn, tp, fn, fp, pr_auc, mean_acc, std_acc, var_acc, np.mean(chance_scores),
                                        per_fold_acc, chance_scores, A_acc, B_acc, C_acc, D_acc, F_acc])

        pipeline_idx += 1

    per_config_metrics_df = pd.DataFrame(data = per_model_subj_res_rows,
                                         columns = ["model_idx","method", "cutoff_low","cuttoff_high","labeled_0","labeled_1","local_scaler",
                                                    "tr","cls", "f1", "brier", "roc_auc", "tn", "tp" , "fn","fp",
                                                    "pr_auc","actual_mean_success_across_folds", "std_success_across_folds",
                                                    "var_success_across_folds",
                                                    "chance_success_across_folds", "per_fold_score", "per_fold_chance_score",
                                                    "A_acc", "B_acc", "C_acc", "D_acc", "F_acc"])


    return per_config_metrics_df

if not PERM_RUN:
    perm_benchmark_df = pd.read_csv(perm_benchmark_df_fn,
                                #"/Users/orenkobo/Desktop/PhD/Aim1/Analysis_artifacts/PERM_benchmark_100_df.csv",
                                converters={"per_fold_score" : eval})
train_types = ["snapshot"]
demo_suffix = "_DEMO" if DEMO else ""
perm_suffix = "_PERMRUN" if PERM_RUN else ""
mult_suffix = "_MULTILABEL" if MULTILABEL else ""
desc_suffix = "7-8_iterall"
test_suffix = f"_{int(N_SUBJ_OUT*100)}subjectsout_{n_CV}iters_"
suffix = f"{desc_suffix}{mult_suffix}{demo_suffix}{perm_suffix}{test_suffix}"
print("SUFFIX IS ", suffix)
str_output = f"=====\nRun: \n\t{list_of_snapshot_trans} , \n\t{list_of_snapshot_models}, \n\t{ET_SCALE_COLS}, " \
             f"\n\t{METHODS}, \n\t{SCALERS}\n" \
             f"Total {len(list_of_snapshot_trans)*len(list_of_snapshot_models)*len(METHODS)*len(SCALERS)}=====\n"
print(str_output)
script_start_timestamp = int(time.time())
analysis_artifacts_dir = f"{analysis_artifacts_dir}"
all_res_dfs = []
all_res_per_cv  = []
#TODO try time series again


c = 0
from sklearn import preprocessing

for is_perm_iter in [False, True]:
    for et_scale_col in ET_SCALE_COLS:
        scale_prefix = et_scale_col[16:]
        is_perm_iter_prefix = "perm_iter" if is_perm_iter else "regr_iter"
    # run_multiprocess = True
    # if run_multiprocess:
        # et_scale_col = sys.argv[-1]
        # scale_prefix = et_scale_col
        suffix += "_" + scale_prefix + "_" + is_perm_iter_prefix
        os.system(f"mkdir {analysis_artifacts_dir}/{script_start_timestamp}")
        os.system(f"mkdir {analysis_artifacts_dir}/{script_start_timestamp}/{suffix}")

        print(f"{datetime.datetime.now()} Read df from {df_fn} col is {et_scale_col}")
        all_subjs_per_subject_settype_mean_df, timepoint_cols = get_timecols_df(fn =df_fn, scale_col = et_scale_col)
        print(f"{datetime.datetime.now()} Got df with shape {all_subjs_per_subject_settype_mean_df.shape} , {len(timepoint_cols)} time cols")
        for override_cutoff in CUTOFFS:
            for method in METHODS:
                p = "_".join([str(x) for x in override_cutoff])
                print(f"{datetime.datetime.now()}: ================= (c={c}) Work on {p} , {method}, {et_scale_col}, {is_perm_iter_prefix} ================= ")
                df = apply_phq_cutoff(all_subjs_per_subject_settype_mean_df,
                                      neg_phq_cutoff = override_cutoff[0],
                                      pos_phq_cutoff = override_cutoff[1])
                if is_perm_iter:
                    df[Y_COL] = df[Y_COL].sample(frac=1, random_state = 42).reset_index(drop=True)
                if MULTILABEL:

                    le = preprocessing.LabelEncoder()
                    df[Y_COL] = le.fit_transform(df[Y_COL])

                all_subjects = df.Subject.unique()

                analysis_artifacts_output_dir = f"{analysis_artifacts_dir}/{script_start_timestamp}/{suffix}/{p}{suffix}/"

                os.system(f"mkdir {analysis_artifacts_output_dir}")

                mode_artifacts_output_dir = analysis_artifacts_output_dir + "snapshot/"
                os.system(f"mkdir {mode_artifacts_output_dir}")
                os.system(f"mkdir {mode_artifacts_output_dir}/pngs")

                snapshot_per_fold_per_cond_per_confid_res_df = train_subject_level_classifier_per_cond(df,
                                                                                                       list_of_snapshot_trans ,
                                                                                                       list_of_snapshot_models,
                                                                                                       c,
                                                                                                       mode = 'snapshot',
                                                                                                       )


                snapshot_per_model_subj_res_df = get_per_config_metrics(mode_artifacts_output_dir,
                                                                        snapshot_per_fold_per_cond_per_confid_res_df,
                                                                        df, method, scale_prefix,
                                                                        override_cutoff,is_perm_iter )



                snapshot_per_fold_per_cond_per_confid_res_df["mode"] = "snapshot"
                snapshot_per_fold_per_cond_per_confid_res_df["ET_scale_col"] = scale_prefix
                snapshot_per_model_subj_res_df["mode"] = "snapshot"
                snapshot_per_model_subj_res_df["ET_scale_col"] = scale_prefix
                snapshot_per_model_subj_res_df["num_folds"] = n_CV

                snapshot_per_fold_per_cond_per_confid_res_df.to_csv(f"{mode_artifacts_output_dir}/{p}{suffix}{method}_snapshot_per_fold_per_cond_per_config_res_df.csv", index = False)
                snapshot_per_model_subj_res_df.to_csv(f"{mode_artifacts_output_dir}/{p}{suffix}{method}_snapshot_per_model_subj_res_df.csv", index = False)

                print(f"{datetime.datetime.now()}: Saved <{override_cutoff},{method},{et_scale_col}> results to {mode_artifacts_output_dir}")

                snapshot_per_model_subj_res_df['is_perm_iter'] = is_perm_iter
                snapshot_per_fold_per_cond_per_confid_res_df['is_perm_iter'] = is_perm_iter
                all_res_dfs.append(snapshot_per_model_subj_res_df)
                all_res_per_cv.append(snapshot_per_fold_per_cond_per_confid_res_df)

                c += 1

all_res_df = pd.concat(all_res_dfs, axis=0)
all_res_per_cv_df = pd.concat(all_res_per_cv, axis=0)
all_res_df.to_csv(f"{analysis_artifacts_dir}/{script_start_timestamp}/{script_start_timestamp}_all_res_df{suffix}.csv", index = False)
all_res_per_cv_df.to_csv(f"{analysis_artifacts_dir}/{script_start_timestamp}/{script_start_timestamp}_all_res_per_cv_df{suffix}.csv", index = False)
with open(f"{analysis_artifacts_dir}/{script_start_timestamp}_description.txt", "w") as text_file:
    text_file.write(str_output)

if PERM_RUN:
    print(f"{datetime.datetime.now()}: Saved new perm benchmark to {perm_benchmark_df_fn}")
    all_res_df.to_csv(perm_benchmark_df_fn, index = False)
else:
    try:
        print(f"{datetime.datetime.now()}: merge with perm in {perm_benchmark_df_fn}")
        per_config_metrics_df_with_perm_benchmark_df = all_res_df.merge(right = perm_benchmark_df,
                                                                                            on = ['local_scaler','tr','cls',
                                                                                                  'method','ET_scale_col',
                                                                                                  'cutoff_low','cuttoff_high'],
                                                                                            suffixes = ("", "_PERM"))

        per_config_metrics_df_with_perm_benchmark_df["temp"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : x.per_fold_score_PERM + [x.actual_mean_success_across_folds], axis=1)
        per_config_metrics_df_with_perm_benchmark_df["classifier_rank"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : sorted(x.temp, reverse=True).index(x.actual_mean_success_across_folds), axis=1)
        per_config_metrics_df_with_perm_benchmark_df["classifier_pval"] = per_config_metrics_df_with_perm_benchmark_df.apply(lambda x : x.classifier_rank / 100, axis=1)
        merged_out_fn = f"{analysis_artifacts_dir}/{script_start_timestamp}/{script_start_timestamp}_all_res_df_with_perm{suffix}.csv"
        per_config_metrics_df_with_perm_benchmark_df.to_csv(merged_out_fn, index = False)
        print(f"{datetime.datetime.now()}: Merged with benchmark {perm_benchmark_df} and saved to {merged_out_fn}")
    except Exception as e:
        print(f"Error {e} when merging with benchmark , cant get p val (probably some configuration missing from benmark file in {perm_benchmark_df}")
#
# if "time_series" in train_types:
#     mode_artifacts_output_dir = analysis_artifacts_output_dir + "timeseries/"
#     os.system(f"mkdir {mode_artifacts_output_dir}")
#     os.system(f"mkdir {mode_artifacts_output_dir}/Fitted_pipelines")
#     os.system(f"mkdir {mode_artifacts_output_dir}/pngs")
#     ts_trs = [TimeSeriesScalerMinMax(), TimeSeriesScalerMeanVariance()]
#     ts_clf = [TimeSeriesSVC(), NonMyopicEarlyClassifier(n_clusters=2), TimeSeriesMLPClassifier(),
#               #LearningShapelets()
#               ]
#
#     timeseries_subject_level_classifier_per_cond_res_df = train_subject_level_classifier_per_cond(mode_artifacts_output_dir,
#                                                                                                   df,
#                                                                                                   ts_trs ,
#                                                                                                   ts_clf,
#                                                                                                   mode = 'timeseries',
#                                                                                                   verbose = 1)
#
#
#     # timeseries_subject_level_classifier_per_cond_res_df.head(4)
#     timeseries_per_model_subj_res_df = get_per_config_metrics(mode_artifacts_output_dir,
#                                                               timeseries_subject_level_classifier_per_cond_res_df,
#                                                               df, take_f,
#                                                               override_cutoff, viz = True)
#
#     print("Saving...")
#
#     timeseries_subject_level_classifier_per_cond_res_df["mode"] = "timeseries"
#     timeseries_per_model_subj_res_df["mode"] = "timeseries"
#     timeseries_per_model_subj_res_df["scaler"] = scale_col
#     timeseries_subject_level_classifier_per_cond_res_df.to_csv(f"{mode_artifacts_output_dir}/{p}{demo_suffix}_timeseries_subject_level_classifier_per_cond_res_df.csv", index = False)
#     timeseries_per_model_subj_res_df.to_csv(f"{mode_artifacts_output_dir}/{p}{demo_suffix}_timeseries_per_model_subj_res_df.csv", index = False)
#     print("Saved to analysis_artifacts_dir")
#     all_res_dfs.append(timeseries_per_model_subj_res_df)
#     all_res_per_cv.append(timeseries_subject_level_classifier_per_cond_res_df)