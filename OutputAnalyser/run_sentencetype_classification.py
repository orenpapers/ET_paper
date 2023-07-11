import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import joblib

list_of_snapshot_trans = [
    FastICA(n_components=80, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=60, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=30, tol=0.3, max_iter=500, random_state=42),
    FastICA(n_components=25, tol=0.3, max_iter=500, random_state=42),
    SelectKBest(k=150),
    SelectKBest(k=120),
    SelectKBest(k=90),
    SelectKBest(k=60),
    SelectKBest(k=30),

]
list_of_snapshot_models = [
    # CatBoostClassifier(random_state=42, n_estimators=100),

    # XGBClassifier(random_state=42,use_label_encoder=False),
    # LGBMClassifier(verbose=0, random_state=42),
    ExtraTreesClassifier(n_estimators=300, random_state=42),
    ExtraTreesClassifier(n_estimators=250, random_state=42),
    ExtraTreesClassifier(n_estimators=200, random_state=42),
    # ExtraTreesClassifier(n_estimators=150, random_state=42),
    # ExtraTreesClassifier(n_estimators=100, random_state=42),
    # ExtraTreesClassifier(n_estimators=50, random_state=42),
    RandomForestClassifier(n_estimators=300, random_state=42),
    RandomForestClassifier(n_estimators=250, random_state=42),
    RandomForestClassifier(n_estimators=200, random_state=42),
    RandomForestClassifier(n_estimators=150, random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    RandomForestClassifier(n_estimators=50, random_state=42),
]
list_of_scalers = [None, StandardScaler(), MinMaxScaler(), Normalizer()]
ET_SCALE_COLS = ["x_gaze_location_normalized","x_gaze_location_standard_scaled","x_gaze_location_minmax_scaled","x_gaze_location"]#,"x_gaze_location_q_scaled","x_gaze_location_r_scaled"]
def main():
    Y_COL = 'is_plausible'
    df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL.csv"
    # scale_col = "x_gaze_location"
    print(f"{datetime.now()} Reading from {df_fn}")
    df = pd.read_csv(df_fn,
                     index_col=None,
                     converters={#'alephbert_enc': eval,
                         # scale_col : eval,
                         # 'x_gaze_location_minmax_scaled' : eval,
                         'x_gaze_location_normalized' : eval,
                         'x_gaze_location_standard_scaled' : eval,
                         'x_gaze_location' : eval,
                         'x_gaze_location_minmax_scaled' : eval,
                         'x_gaze_location_orig' : eval,
                         # 'target_word_x_range' : eval
                         # 'phq_label': bool
                     })
    print(f"{datetime.now()} df.shape")

    cond_one_hot_cols = ["Sentence_type_A","Sentence_type_B","Sentence_type_C","Sentence_type_D"]
    df['is_experimental'] = [False if x=='F' else True for x in df.Sentence_type]
    # df = df[df['is_experimental'] == True].reset_index(drop=True)
    df['is_plausible'] = [True if x in ['A','C'] else False for x in df.Sentence_type]
    # df['target_word_x_range'] = [(0,0) if pd.isna(x) else eval(x) for x in df['target_word_x_range']]
    #[len(set(range(a, b+1)).intersection(y))  for (a, b), y in df[['t','l']].to_numpy()]
    # df['num_fixations_in_target_word'] = [sum(a <= i <= b for i in y) for (a, b), y in df[['target_word_x_range','x_gaze_location_orig']].to_numpy()]
    timepoint_cols_all = [f"timepoint#{i}" for i in range(875)]
    id_cols = ["is_plausible", 'is_experimental','Sentence_type','Subject'] + cond_one_hot_cols

    scores = []
    i = 0
    t = len(ET_SCALE_COLS) * len(list_of_scalers) * len(list_of_snapshot_trans) * len(list_of_snapshot_models)

    scores_dict = {}
    for scale_col in ET_SCALE_COLS:
        timeseries_df = pd.DataFrame(data = df[scale_col].to_list() , columns = timepoint_cols_all).ffill(axis=1).astype(float)
        timeseries_df = timeseries_df.iloc[:,200:]
        timeseries_df[id_cols] = df[id_cols]

        timepoint_cols = [x for x in list(timeseries_df.columns) if "timepoint" in x]
        per_subject_settype_mean_df = timeseries_df.groupby(["Subject","Sentence_type"]).mean()
        per_subject_settype_mean_df = per_subject_settype_mean_df.reset_index()
        per_subject_settype_mean_df = per_subject_settype_mean_df.ffill()
        per_subject_settype_mean_df = per_subject_settype_mean_df[per_subject_settype_mean_df['is_experimental'] == True].reset_index(drop=True)
        X = per_subject_settype_mean_df[timepoint_cols]
        y = per_subject_settype_mean_df[Y_COL]
        for tr in list_of_snapshot_trans:
            for est in list_of_snapshot_models:
                for scl in list_of_scalers:
                    try:
                        pipeline = make_pipeline(scl , tr, est)

                        print(f"{datetime.now()} : Running {scale_col},{scl}, {tr}, {est} (i={i}/{t})")
                        res_row = [tr,est,scl]

                        a1 = cross_val_score(estimator=pipeline, X=X, y=per_subject_settype_mean_df['is_plausible'], cv=20)
                        a2 = cross_val_score(estimator=pipeline, X=per_subject_settype_mean_df[timepoint_cols + ['is_experimental']], y=per_subject_settype_mean_df['is_plausible'], cv=20)
                        a3 = cross_val_score(estimator=pipeline, X=X, y=per_subject_settype_mean_df['Sentence_type'], cv=20)
                        a4 = cross_val_score(estimator=pipeline, X=per_subject_settype_mean_df[timepoint_cols + ['is_experimental']], y=per_subject_settype_mean_df['Sentence_type'], cv=20)

                        res_row += [np.mean(x) for x in [a1, a2, a3, a4]]

                        for cond in ['A','B','C','D']:
                            cond_acc = cross_val_score(estimator=pipeline, X=X, y=per_subject_settype_mean_df['Sentence_type'] == cond, cv=10)
                            res_row.append(np.mean(cond_acc))


                        print(f"{datetime.now()}: Got the scores, mean={[round(np.mean(x),3) for x in [a1,a2,a3,a4]]}")
                        scores.append(res_row)
                        i+=1
                    except Exception as e:
                        print(f"{datetime.now()} Failed {scale_col},{scl}, {tr}, {est} - {e}")

    print(f"{datetime.now()} Creating df... ")
    res_df = pd.DataFrame(data = scores, columns=["tr","est","scl",
                                                  "plausibility_cond_detection_acc1","plausibility_cond_detection_acc2",
                                                  "Sentence_type_detection_acc1","Sentence_type_detection_acc2",
                                                  "A_VS_all_acc","B_VS_all_acc","C_VS_all_acc","D_VS_all_acc"])
    print(f"{datetime.now()} Saving df... ")
    res_df.to_csv("/Users/orenkobo/Downloads/1_sentencestype_classification_res_dict.csv", index=False)
    # joblib.dump("/Users/orenkobo/Downloads/sentence_classification_res_dict.jbl")
    print(f"{datetime.now()} Done ")

if __name__ == "__main__":
    main()