import time
from datetime import datetime
import numpy as np
import pandas as pd
from OutputAnalyser.TimeSeriesAnalyser import main_dir

def _get_phq_group(phq_score):

    if phq_score in range(0,5): #0 to 4 is Minimal
        return "Minimal"

    if phq_score in range(5,10): #5 to 9 is Mild
        return "Mild"

    if phq_score in range(10,15): #10 to 14 is Moderate
        return "Moderate"

    if phq_score in range(15,20): #15 to 19 is Moderately severe
        return "Moderately Severe"

    if phq_score in range(20,28): #20 to 27 is severe
        return "Severe"

    return np.nan

def get_subject_phq(subject_id, res_type):
    from params import phq_fn
    phq_df = pd.read_csv(phq_fn)

    if len(phq_df[phq_df["sid"] == subject_id]) == 0:
        print("Cant get PHQ for " , subject_id)
        return np.nan

    phq_score = sum(phq_df[phq_df["sid"] == subject_id].iloc[:, 2:-1].values[0])
    phq_group = _get_phq_group(phq_score)
    if res_type == "score":
        return phq_score
    if res_type == "group":
        return phq_group

def add_label_column(df, id_col = "Subject"):
    print("{} add_label_column".format(datetime.now().time()))
    df.loc[ : , "phq_score"] = df.apply(lambda x : get_subject_phq(x[id_col], res_type ="score"), axis=1)
    df.loc[ : , "phq_group"] = df.apply(lambda x : get_subject_phq(x[id_col], res_type ="group"), axis=1)
    return df

def add_x_vec_len(df):
    df.loc[ : , "x_gaze_len"] = df.x_gaze_location.apply(lambda x : len(eval(x)))
    return df

def _split_train_test_by_subject():
    import random
    u = random.random()
    if u < 0.2:
        return "test"
    if u < 0.4:
        return "val"

    return "train"

def add_train_test_split(df, id_col = "Subject"):
    print(time.time() , "add_train_test_split")
    for sid, sdf in df.groupby([id_col]):
        df_type = _split_train_test_by_subject()
        df.loc[sdf.index, "df_type"] = df_type
    print(time.time() , "added")
    return df

def add_videosegementnum_col(df, frames_per_segment = 5):
    print(time.time(), "add_videosegementnum_col")
    def _get_video_segment_from_frame_num(x, frames_per_segment):
        return int(x.frame_num / 100 / 5)

    df.insert(3,"video_segment_num", df.apply(lambda x : _get_video_segment_from_frame_num(x,frames_per_segment) , axis = 1))
    df.video_segment_num = df.video_segment_num.astype(float)
    df.sort_values(by=['video_segment_num'])
    print(time.time() , "added")
    return df