import imageio
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import datetime
import os
import joblib

def apply_phq_cutoff(df , neg_phq_cutoff, pos_phq_cutoff):
    df["phq_binary_label"] = [0.0 if x <= neg_phq_cutoff else 1.0 if x >= pos_phq_cutoff else "other" for x in df.phq_score]
    df = df[df.phq_binary_label!= 'other']
    return df


def get_image_df(fn, apply_mean, use_existing = True):

    numeric_cols_to_keep = ["trial_total_distance_covered", "sentence_pupil_diameter_mean","phq_score","is_experimental_sentence"]
    df = pd.read_csv(fn,
                     index_col=None,
                     converters={'alephbert_enc': eval,
                                 '1d_vgg_featuremap' : eval,
                                 #'x_gaze_location' : eval,
                                 #'x_gaze_location_normalized' : eval,
                                 #'sentence_pupil_diameter_vec_normalized' : eval
                                 # 'phq_label': bool
                                 })
    print(df.shape)
    df['is_experimental_sentence'] = [False if x=='F' else True for x in df.Sentence_type]
    print(f"{datetime.datetime.now()} Got shape {df.shape} from {fn}, go to agg data")
    arrs = []
    if apply_mean:
        new_col = "mean_img"
        for g, gdf in df.groupby(["Subject","Sentence_type"]):

            im_list = list(gdf.heatmap_fn)
            # w, h,_ = img_to_array(load_img(im_list[0])).shape #this is a PIL image
            w, h = Image.open(im_list[0]).size

            arr = np.zeros((h,w,3), np.float)

            for im in im_list:
                imarr = img_to_array(load_img(im))
                arr = arr + (imarr / len(im_list))

            arr = np.array(np.round(arr), dtype=np.uint8) # Round values in array and cast as 8-bit integer
            arrs.append(arr)

        mdf = df.groupby(["Subject","Sentence_type", "phq_group"])[numeric_cols_to_keep].mean().reset_index()
        mdf[new_col] = arrs
        print(f"{datetime.datetime.now()} Got data with shape {mdf.shape}")
        return mdf , new_col

    else:
        new_col = "img"
        # new_fn = fn+ "_no_agg.csv"
        # if use_existing and os.path.isfile(new_fn):
        #     df = pd.read_csv(new_fn, index_col=False)
        #     print(f"{datetime.datetime.now()} Got data with shape {df.shape} from {new_fn}")
        # else:
        df[new_col] = df.heatmap_fn.apply(lambda x: img_to_array(load_img(x)))
        print(f"{datetime.datetime.now()} Created data with shape {df.shape}")
        # df.to_csv(new_fn, index_label=None)
        # print(f"{datetime.datetime.now()} Saved data to {new_fn}")
        return df, new_col


def get_timecols_df_for_DL(fn ="/Users/orenkobo/Desktop/PhD_new/repos/HebLingStudy/notebooks/df.csv",
                           scale_col = "x_gaze_location_orig",
                           word_embedding_dict_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/word2alephbert_encodingdict.jbl"):

    import string, re
    print(f"{datetime.datetime.now()} Reading csv from {fn}")
    df = pd.read_csv(fn,
                     index_col=None,
                     converters={#'alephbert_enc': eval,
                         scale_col : eval,
                         # 'x_gaze_location_minmax_scaled' : eval,
                         'x_gaze_location_orig' : eval,
                         # 'target_word_x_range' : eval
                         # 'phq_label': bool
                     })
    print(df.shape)

    def exclude_char(s):
        exclude = set(string.punctuation)
        exclude.remove("-")
        return ''.join(ch for ch in s if ch not in exclude)

    words_embedding_dict = joblib.load("/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/word2alephbert_encodingdict.jbl")
    df = df[df.Sentence_type != 'F'].reset_index(drop=True)
    df["words_order"] = df["words_order"].apply(lambda x : [exclude_char(w) for w in eval(x)])
    df['word_embeddings_order'] = df["words_order"].apply(lambda x : [words_embedding_dict[w] for w in x])

    id_cols = ["phq_score","phq_group","Subject", "Sentence_type",
               "sentence_pupil_diameter_mean","set_num", "words_order","word_embeddings_order"]
    vec_size = 3500
    new_colname = f"x_gaze_location_{vec_size}"
    cols = [f"timepoint#{i}" for i in range(vec_size)]
    df[new_colname] = df["x_gaze_location_orig"].apply(lambda x : x[:vec_size])
    timeseries_df = pd.DataFrame(data = df[new_colname].to_list() , columns = cols)
    timeseries_df[id_cols] = df[id_cols]
    timeseries_df = timeseries_df.iloc[:,200:]
    cols = [x for x in timeseries_df.columns if "timepoint" in x]
    return timeseries_df, cols

def get_timecols_df(fn ="/Users/orenkobo/Desktop/PhD_new/repos/HebLingStudy/notebooks/df.csv", scale_col = "x_gaze_location_minmax_scaled",
                    return_df = False):
    print(f"{datetime.datetime.now()} Reading csv from {fn}")
    df = pd.read_csv(fn,
                     index_col=None,
                     converters={#'alephbert_enc': eval,
                                 scale_col : eval,
                                 # 'x_gaze_location_minmax_scaled' : eval,
                                 'x_gaze_location_orig' : eval,
                                 # 'target_word_x_range' : eval
                                 # 'phq_label': bool
                                 })
    print(df.shape)

    df['is_experimental_sentence'] = [False if x=='F' else True for x in df.Sentence_type]
    df['target_word_x_range'] = [(0,0) if pd.isna(x) else eval(x) for x in df['target_word_x_range']]
    #[len(set(range(a, b+1)).intersection(y))  for (a, b), y in df[['t','l']].to_numpy()]
    df['num_fixations_in_target_word'] = [sum(a <= i <= b for i in y) for (a, b), y in df[['target_word_x_range','x_gaze_location_orig']].to_numpy()]
    timepoint_cols = [f"timepoint#{i}" for i in range(875)]
    id_cols = ["phq_score","phq_group","Subject", "Sentence_type",
               "sentence_pupil_diameter_mean","set_num","is_experimental_sentence"]
    timeseries_df = pd.DataFrame(data = df[scale_col].to_list() , columns = timepoint_cols) 
    timeseries_df[id_cols] = df[id_cols]
    timeseries_df = timeseries_df.iloc[:,200:]
    timepoint_cols = [x for x in list(timeseries_df.columns) if "timepoint" in x]
    per_subject_settype_mean_df = timeseries_df.groupby(["Subject","Sentence_type","phq_group"]).mean()
    per_subject_settype_mean_df = per_subject_settype_mean_df.reset_index()
    per_subject_settype_mean_df = per_subject_settype_mean_df.ffill()

    # from TimeSeriesAnalyser.utils import get_demographics_str
    # from TimeSeriesAnalyser.ts_params import exp_output_dir
    # s = get_demographics_str(exp_output_dir, [str(x).zfill(3) for x in per_subject_settype_mean_df.Subject.unique()])
    # print(s)
    if return_df:
        return per_subject_settype_mean_df, timepoint_cols , df
    return per_subject_settype_mean_df, timepoint_cols
