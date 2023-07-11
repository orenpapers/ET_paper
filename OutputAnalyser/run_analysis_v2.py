import math
from sklearn.preprocessing import StandardScaler
from OutputAnalyser.TimeSeriesAnalyser import seglearn_feats
# from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, TrendChangeParameters
import pandas as pd
import ast
import numpy as np
from hepsylex import Lexicons
from hepsylex import LexiconsAPI
from hepsylex import Lexicons
import os
import matplotlib.pyplot as plt
from OutputAnalyser.TimeSeriesAnalyser import extract_kats_feats, extract_bagofpattern, extract_BOSSVS
from sklearn.preprocessing import minmax_scale, normalize, scale, robust_scale, quantile_transform
from sklearn.decomposition import PCA
from OutputAnalyser.TimeSeriesAnalyser import utils
from OutputAnalyser.TimeSeriesAnalyser import add_label_column
from OutputAnalyser.TimeSeriesAnalyser import artifacts_dir
from OutputAnalyser.TimeSeriesAnalyser import exp_output_dir
import joblib
from OutputAnalyser.TimeSeriesAnalyser import seglearn_classification
# from OutputsAnalyser.TimeSeriesAnalyser.Classifiers import  sktime_classification
from sklearn.metrics import accuracy_score
from TimeSeriesAnalyser.utils import get_demographics_str
from OutputsAnalyser.multimodal_classifiers import train_multimodal_model_segmented_and_pca, train_unsegmented_projection_multimodal_network
from datetime import datetime
from kats.detectors.cusum_detection import CUSUMDetector

all_subjects_ids = [ "138","073", "060", "064",
                     "062", "101","111",
                     "131", "132", "133", "134", "135", "136",
                     "137", "139", "066", "067", "080", "072",
                     "077", "113", "114", "115", "116", "117",
                     "118", "119", "120", "121", "122", "123",
                     "124", "125", "126", "127", "128", "033",
                     "098", '026', '006', '039', '044', '041',
                     '025', '065', '076', '021', "114", "113",
                     "103", "105", "106", "003", "064", "100",
                     "107", "108", "109", "110", "112", "010",
                     "082", "083", "084", "087", "088", "089",
                     "090", "093", "094", "095", "096", "097",
                     "032", "023", "020", "004", "040", "099",
                     "038", "014", "045", "047", "031", "063",
                     "034", "035", "026", "029", "030", "074",
                     "017", "005", "006", "007", "008", "027",
                     "039", "046", "044", "043", "042", "041",
                     "011", "045", "071", "021", "024", "012",
                     "049", "050", "051", "052", "053", "054",
                     "015", "048", "078", "081", "082", "036",
                     "022", "055", "056", "057", "058", "059",
                     "069", "070"

                     ]

invalid_subject_ids = ["075","073","079","061","068","001","015","024", "045","098", "130", "129", "140",
                       "002","009","013","016","018","019", "028", "036",
                       "085","086","091","092","102","104"] #60 - problem with data acquisition (only 12 sentences)
valid_subject_ids = [x for x in all_subjects_ids if x not in invalid_subject_ids ]
a = 2
class ValidationFilter:

    def __init__(self, steps = ["sample"]):
        self.steps = steps

    def subject_validator(self, sd_thr = 2.5):
        pass

    def sample_validator(self,df , sd_thr = 6000):
        if "sample" in self.steps:
            return df[(df.x_gaze_len < 6000) & (df.x_gaze_len > 1000)]

    def sentence_validator(self,sd_thr = 2.5):
        pass

def pred_vec2subject_vec(df, test_idx, pred):

    df_test = df.iloc[test_idx]
    df_test.loc[:,"seg_pred"] = pred
    subj_pred_vec = df_test.groupby("Subject").seg_pred.mean()
    return subj_pred_vec

def create_alephbert_encoding( create):
    import re, string
    if not create:
        return
    print(f"{datetime.now()}: create_alephbert_encoding - read csv")
    df = pd.read_csv("/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/all_sentences_all_subjects_df.csv")
    print(f"{datetime.now()}: read csv , {df.shape}")
    from transformers import pipeline, AutoTokenizer
    # from transformers import BertModel, BertTokenizerFast
    # alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    # alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
    #
    # https://colab.research.google.com/drive/1oYoCUABe4YO-jkcUAst2cDyRDVeYwHfm?authuser=2#scrollTo=Op_5HhPBVu5G

    alephbert_fe_pipeline = pipeline('feature-extraction', model='onlplab/alephbert-base', tokenizer='onlplab/alephbert-base')
    print(f"{datetime.now()}: Loaded alephbert_fe_pipeline, creating dict")
    s_d = {}
    w_d = {}
    for s in list(df.Sentence.unique()):
        s_d[s] = list(np.mean(alephbert_fe_pipeline(s)[0][1:-1], axis=0))
        for w in s.split():
            w_d[w] = alephbert_fe_pipeline(w)[0][1] #todo make sure it is 1
            w = re.sub(r'[^\w\s]','',w)
            w_d[w] = alephbert_fe_pipeline(w)[0][1]

    print(f"{datetime.now()}: Dicts are ready")
    joblib.dump(s_d, "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/sentence2alephbert_encodingdict.jbl")
    joblib.dump(w_d, "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/word2alephbert_encodingdict.jbl")
    a = 2
import os.path


# from keras.preprocessing.image import img_to_array, load_img
from numpy import expand_dims
import tensorflow as tf
# from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


rows = 224
columns = 224
channels = 3
vgg_cnn_base = VGG16(input_shape=(rows,
                                  columns,
                                  channels),
                     weights="imagenet",
                     include_top=True)
vgg_cnn = tf.keras.Model(vgg_cnn_base.input, vgg_cnn_base.layers[-3].output, name="VGG_fm")


def retrieve_VGG_featuremap(vgg):
    return vgg_cnn.predict(vgg)


def fn2img(fn):
    # load the image with the required shape
    with tf.keras.preprocessing.image.load_img(fn, target_size=(224, 224)) as img1:
        img2 = tf.keras.preprocessing.image.img_to_array(img1)
    img3 = expand_dims(img2, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img3)
    return img


def add_heatmap_data_columns(df):
    def get_heatmap_fn(x):
        subj_str = lambda y: str(y).zfill(3)
        base_fn = f"{exp_output_dir}/{subj_str(x.Subject)}/heatmaps/{subj_str(x.Subject)}_run#{x.sentence_run_num}_trial#{x.sentence_trial_num}_heatmap"
        return base_fn + ".png"

    def get_fm(x):
        try:
            heatmap_vgg_enc = fn2img(x)
            vgg_featuremap = retrieve_VGG_featuremap(heatmap_vgg_enc)
            fm = list(vgg_featuremap[0])
            return fm
        except Exception as e:
            print(f"{datetime.now()}: Cant get VGG FM {e} , put nan")
            return np.nan

    df["heatmap_fn"] = df.apply(lambda x : get_heatmap_fn(x) , axis=1)
    df["1d_vgg_featuremap"] = df["heatmap_fn"].apply(lambda x : get_fm(x) )

    return df

def add_DSP_feats(df, gaze_col):
    print("{} Extract DSP feats".format(datetime.now().time()))

    # bop_df = extract_bagofpattern(df, gaze_col)
    # boss_df = extract_BOSS(df, gaze_col)
    # bossvs_df = extract_BOSSVS(df, gaze_col)
    # weasel_df = extract_WEASEL(df, gaze_col)
    kats_feats_df = extract_kats_feats(df, gaze_col)
    df = pd.concat([df, kats_feats_df] , axis=1)
    return df

def add_heatmap_data_columns_from_files(df):
    def get_hm_fn(x, suffix):
        subj_str = lambda y: str(y).zfill(3)
        fn =  "{}/{}/heatmaps/{}_run#{}_trial#{}_heatmap_vgg{}"\
            .format(exp_output_dir, subj_str(x.Subject), subj_str(x.Subject), x.sentence_run_num, x.sentence_trial_num , suffix)
        if os.path.isfile(fn):
            return fn
        else:
            return math.nan

    df["1d_featuremap_fn"] = df.apply(lambda x : get_hm_fn(x, suffix="_feature_map.jbl"), axis=1)
    df["1d_vgg_featuremap"] = df["1d_featuremap_fn"].apply(lambda x : list(joblib.load(x)[0]) if not pd.isna(x) else math.nan)
    # df["3d_featuremap_fn"] = df.apply(lambda x : get_hm_fn(x, suffix=".jbl"), axis=1)
    # df["3d_vgg_featuremap"] = df["3d_featuremap_fn"].apply(lambda x : np.array(joblib.load(x)[0]) if not pd.isna(x) else math.nan )
    return df

def smooth_gaze(gaze_vec, smooth_factor, scale = False):
    if len(np.unique(gaze_vec)) < 10:
        return math.nan
    if scale:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(np.array(gaze_vec).reshape(-1,1))).transpose()
        df_scaled_smoothed = df_scaled.groupby(np.arange(len(df_scaled.columns)) // smooth_factor, axis=1).mean()
        return list(df_scaled_smoothed.iloc[0])
    else:
        d =  pd.DataFrame(gaze_vec).transpose()
        return list(d.groupby(np.arange(len(d.columns)) // smooth_factor, axis=1).mean().iloc[0])

def trunc_gaze(gaze_vec, trunc_size):
    return gaze_vec[:trunc_size]

def get_pca_on_col(series , n):
    return pd.Series(pd.DataFrame(PCA(n_components=n).
                                  fit_transform(np.vstack(series))).values.tolist())

def add_depression_cols(df):

    lexicons = Lexicons()

    depr_lexs = LexiconsAPI.lexicons_union([x[1] for x in lexicons if "Depress" in x[0]] +
                                           [lexicons.EmotionalVariety_Anxiety,  lexicons.EmotionalVariety_Anger , lexicons.EmotionalVariety_Disgust , \
                                           lexicons.EmotionalVariety_Hostile, lexicons.EmotionalVariety_Nervous, lexicons.EmotionalVariety_Sad, \
                                           lexicons.Paralinguistics_Crying, Lexicons.Valence_Negative])

    df["num_words_in_lexicon"] = df.Sentence.apply(lambda x : LexiconsAPI.number_of_words_in_lexicon(x, depr_lexs))
    return df




def remove_invalid_subjects(df, exp_output_dir, valid_thr):
    subj_to_remove = []
    for subj, subj_df in df.groupby(["Subject"]):
        subj2str = lambda y: str(y).zfill(3)
        fn = "{}/{}/{}_Sentences_res.tsv".format(exp_output_dir, subj2str(subj), subj2str(subj))
        subj_res_df = pd.read_csv(fn, error_bad_lines = False , delimiter= '\t')
        comp_ratio = len(subj_res_df[subj_res_df["comp_is_correct"] == True]) / len(subj_res_df)
        print("For subj {}, comp Q ratio is {}".format(subj, comp_ratio))
        if comp_ratio < valid_thr:
            subj_to_remove.append(subj)
    print("Dropping {} invalid subjects {}".format(len(subj_to_remove), subj_to_remove))
    df = df[~df.Subject.isin(subj_to_remove)]
    return df

def add_noise(row):
    #add label-driven noise to to vector of gaze to prevent overfit
    l = row.x_gaze_location
    y = row.phq_score
    l[-1] = 0.6 * (y > 7)
    return l

demographics_str = get_demographics_str(folder=exp_output_dir, subject_ids=all_subjects_ids)

all_sentences_info_df = pd.read_csv("/Users/orenkobo/Desktop/PhD/HebLingStudy/Task_Sentences/Materials/ExpSentences.csv")
create_alephbert_encoding(create = False)
#todo : create more sensitive heatmap
#todo : use resnet instead of VGG?

use_existing_data = False
use_distilled = False
add_dsp = False
add_heatmap = False
if use_existing_data:
    print("{} Use existing data ... Reading df ...".format(datetime.now().time()))
    if use_distilled:
        print("use_distilled")
        df = pd.read_csv(artifacts_dir + "df_new_distilled_unsegmented_alldata_new.csv", index_col=None,
                         converters={'alephbert_enc_20_PCA': eval,
                                     'VGG_1d_featuremap_20_PCA' : eval,
                                     'x_gaze_location_segment' : eval,
                                     'phq_label': bool})

    else:
        df = pd.read_csv(artifacts_dir + "df_new_full__unsegmented_alldata_new.csv", index_col=None,
                         converters={'alephbert_enc': eval,
                                     '1d_vgg_featuremap' : eval,
                                     'x_gaze_location' : eval,
                                     # 'phq_label': bool
                                     })


else:
    # df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/all_sentences_all_subjects_df.csv"
    df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/all_sentences_all_subjects_df_FINAL.csv"
    # df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/all_sentences_all_subjects_df_FINAL.csv"
    # df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Analysis_2021/21/all_sentences_all_subjects_df_fn.csv"
    print(f"{datetime.now()} Create new df from {df_fn}" )
    df_raw = pd.read_csv(df_fn,
                     converters={'x_gaze_location': eval,
                                 'sentence_pupil_diameter_vec': eval},
                     index_col=False)



    print("shape 1 : ", df_raw.shape)
    df_raw = df_raw[~df_raw.Subject.isin([int(x) for x in invalid_subject_ids])]
    print("shape 2 : ", df_raw.shape)
    df_r = remove_invalid_subjects(df_raw, exp_output_dir, 0.75)
    print("shape 3 : ", df_r.shape)
    df_r = add_depression_cols(df_r)
    df_f = df_r[df_r["Sentence_type"] == "F"]
    df_t = df_r[df_r["Sentence_type"]!='F']
    sentence_key = ["Subject", "sentence_run_num", "sentence_trial_num", "Sentence_type","set_num"]
    df_t = df_t.merge(df_t.pivot_table('word_idx', sentence_key, 'word_type', aggfunc='first').eval('target_word/end_word').rename('target_word_loc'), on=sentence_key)
    df_t = df_t[df_t["word_type"] == "target_word"]  #doesn;t really matter what the word type - just to get rid of some weird duplicated
    df_t = df_t.drop_duplicates(sentence_key)
    df = pd.concat([df_t , df_f] , axis=0).sort_values(by=['Subject','sentence_run_num','sentence_trial_num']).reset_index(drop=True)
    df = df.drop_duplicates(sentence_key)
    # phq_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/ts_data/1_P_ET_questionnaire.csv"
    phq_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/1_P_ET_questionnaire.csv"
    df = df[df.Subject.isin([int(x) for x in valid_subject_ids])]
    df['sentence_pupil_diameter_vec_mean'] = df.sentence_pupil_diameter_vec.apply(lambda x: np.mean(x))
    # df = df.drop(['word_pupil_diameter_mean', 'word_type','word_value',
    #               'word_idx','first_fixation_duration','second_fixation_duration','total_gaze_duration',
    #               'num_of_fixations','word_has_first_pass_regression','regression_path_duration',
    #               'is_skipping_trial','time_at_word',
                  # 'sentence_pupil_diameter_vec'
                  # 'sentence_pixels_per_word','looked_at_word_timestamps_vectors',
                  # 'pupil_diameter_per_fixation_when_looked_at_word'], axis=1) #nan cols
    print(df.shape)

    df = df[df["x_gaze_location"].map(len) > 500]
    print(df.shape)
    df = add_label_column(df)
    print(f"{datetime.now()} A1: #Subjects : {df.Subject.nunique()} , shape = {df.shape}", )
    df = df.dropna(subset=["x_gaze_location", "phq_score"]).reset_index(drop=True)
    print(f"{datetime.now()} A2: #Subjects : {df.Subject.nunique()} , shape = {df.shape}", )

    print(df["x_gaze_location"].str.len().groupby(df["Sentence_type"]).agg(["mean","std"]))

    df["x_gaze_location_orig"] = df["x_gaze_location"]
    df["x_gaze_location"] = df["x_gaze_location"].apply(lambda x : trunc_gaze(x, trunc_size=3500))
    df["x_gaze_location"] = df["x_gaze_location"].apply(lambda x : smooth_gaze(x, smooth_factor=4, scale=True))
    df["x_gaze_location"] = df["x_gaze_location"].apply(lambda x : [round(f,3) for f in x] if type(x) == list else x)
    df = df.dropna(subset=["x_gaze_location"]).reset_index(drop=True)
    print(f"{datetime.now()} A3: #Subjects : {df.Subject.nunique()} , shape = {df.shape}", )

    df["x_gaze_location_rescaled"] = df.apply(lambda x: list(add_noise(x)), axis = 1)
    df["x_gaze_location_minmax_scaled"] = df["x_gaze_location"].apply(lambda x: list(minmax_scale(x)))
    df["x_gaze_location_standard_scaled"] = df["x_gaze_location"].apply(lambda x: list(scale(x)))
    # df["x_gaze_location_q_scaled"] = df["x_gaze_location"].apply(lambda x: list(quantile_transform(np.array(x).reshape((1,-1)), n_quantiles=50)[0]))
    df["x_gaze_location_r_scaled"] = df["x_gaze_location"].apply(lambda x: list(robust_scale(x)))
    df["x_gaze_location_normalized"] = df["x_gaze_location"].apply(lambda x: list(normalize(np.array(x).reshape((1,-1)))[0]))
    df["sentence_pupil_diameter_vec_normalized"] = df["sentence_pupil_diameter_vec"].apply(lambda x: list(minmax_scale(x)))
    if add_heatmap:
        print("{} Adding heatmap...".format(datetime.now()))
        df = add_heatmap_data_columns(df)
    print(f"{datetime.now()} Added")
    df, c_cols = utils.onehot_encode_categorical(df, ["Sentence_type"])
    # df = df.dropna(subset=["1d_featuremap_fn"]).reset_index(drop=True)
    print("{} Add alephbert..".format(datetime.now().time()))
    d = joblib.load("/Users/orenkobo/Desktop/PhD/HebLingStudy/Output/sentence2alephbert_encodingdict.jbl")
    df["alephbert_enc"] = df.Sentence.apply(lambda x: d[x])
    apply_pca_embeddings = False
    apply_segmentation = False
    df["words_order"] = df.fixation_words_order.apply(lambda x: [y[:-3] for y in x.split("->")] if not pd.isna(x) else [])
    if apply_pca_embeddings:
        df["alephbert_enc_20_PCA"] = get_pca_on_col(df['alephbert_enc'], 20)
        df["VGG_1d_featuremap_20_PCA"] = get_pca_on_col(df['1d_vgg_featuremap'], 20)

    if apply_segmentation:

        print("{} Segmenting...".format(datetime.now().time()))
        print(df.shape)
        window_size = 150
        step = 50

        df['x_gaze_location_segment_padded'] = df.x_gaze_location.apply(lambda x :x + [0]*(len(x)%window_size) )
        df['x_gaze_location_segment'] = df.x_gaze_location_segment_padded.apply(
            lambda lst: [lst[i:i+window_size]+[idx] for idx, i in enumerate(range(0, len(lst), step))])

        df2 = df.explode('x_gaze_location_segment')
        df3 = df2.assign(segment_idx=df2.x_gaze_location_segment.map(lambda x:x.pop()))
        df = df3
        df = df.reset_index(drop=True)
        print(df.shape)
        gaze_col = 'x_gaze_location_segment'
    else:
        gaze_col = 'x_gaze_location'

    if add_dsp:
        df = add_DSP_feats(df, gaze_col)
    df_distilled = df.drop(['trial_actual_tuples_of_orig_gaze_locations','x_gaze_location' ] , axis=1)
    if apply_segmentation:
        df_distilled = df.drop(['x_gaze_location_segment'] , axis=1)
    if apply_pca_embeddings:
        df_distilled = df.drop(['alephbert_enc','1d_vgg_featuremap'] , axis=1)
    print("{} Saving....".format(datetime.now().time()))
    df_distilled.to_csv(artifacts_dir + "df_new_distilled{}segmented_alldata_new_FINAL_paparanalysis_withETLingFeatures.csv".format("_" if apply_segmentation else "_un") , index=False)
    df.to_csv(artifacts_dir + "df_new_full_{}segmented_alldata_new_FINAL_paparanalysis_withETLingFeatures.csv".format("_" if apply_segmentation else "_un") , index=False)
    # joblib.dump(seglearn_data, artifacts_dir + "seglearn_data.jbl")
    print("{} Saved to {}".format(datetime.now().time(), artifacts_dir))
    exit()

res_dict = {}

#relevant papers:
# https://link.springer.com/content/pdf/10.3758/s13428-018-1133-5.pdf
# https://reader.elsevier.com/reader/sd/pii/S2405896321003098?token=288D7E108F24EC5C3735D2A25DC9C3D0F2B37BF316B9293DCB476FE690878F920F768B21FF80127B143F235C2CFB1EC9&originRegion=eu-west-1&originCreation=20210923090332
# https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_40.pdf
#todo try implement

# https://www.researchgate.net/publication/336335958_Eye_Gaze-based_Early_Intent_Prediction_Utilizing_CNN-LSTM?enrichId=rgreq-e24214581107fa8f55b58ad564c99714-XXX&enrichSource=Y292ZXJQYWdlOzMzNjMzNTk1ODtBUzo4NjI4NjI5MDE3ODA0ODJAMTU4MjczMzk1NTI4Mg%3D%3D&el=1_x_3&_esc=publicationCoverPdf
# https://vision.ece.ucsb.edu/sites/default/files/publications/2017_icip_thuyen.pdf
# https://escholarship.org/content/qt8qs6x5cv/qt8qs6x5cv.pdf?t=pguzip
#
# https://medium.com/walmartglobaltech/time-series-similarity-using-dynamic-time-warping-explained-9d09119e48ec
# https://dtaidistance.readthedocs.io/en/latest/usage/clustering.html

print(f"There are total of {df.Subject.nunique()} subjects")
df["x_gaze_location_normalized"] = df["x_gaze_location"].apply(lambda x: list(minmax_scale(x)))
# save_csv = False
# if save_csv:
#     df.to_csv("/Users/orenkobo/Desktop/PhD_new/repos/HebLingStudy/notebooks/df.csv", index=False)

take_existing_timepoints = True
timepoints_fn = artifacts_dir + "timepoints_cols.csv"
subj_timepoints_fn = artifacts_dir + "subj_timepoints_df.csv"
if take_existing_timepoints:
    timepoints_df = pd.read_csv(timepoints_fn, index_col=False)
    subj_timepoints_df = pd.read_csv(subj_timepoints_fn, index_col=False)
else:
    df["phq_label"] = [0.0 if x <= 7 else 1.0 if x >= 8 else "other" for x in df.phq_score]
    timepoint_cols = [f"timepoint#{i}" for i in range(875)]
    id_cols = ["phq_label", "Subject", "Sentence_type", "set_num"]
    timeseries_df = pd.DataFrame(data = df.x_gaze_location_normalized.to_list() , columns = timepoint_cols)
    timeseries_df[id_cols] = df[id_cols]
    timeseries_df = timeseries_df.iloc[:,200:]
    timepoint_cols = [x for x in list(timeseries_df.columns) if "timepoint" in x]
    subj_timepoints_df = df.groupby(["Subject","Sentence_type"]).mean().reset_index()

    timeseries_df.to_csv(timepoints_fn, index=False)
    subj_timepoints_df.to_csv(subj_timepoints_fn, index=False)
    print(f"{datetime.now()} : Saved subj_timepoints_fn to {subj_timepoints_fn}")
exit()
train_unsegmented_projection_multimodal_network(df, override_cutoff = [7, 8])
train_unsegmented_projection_multimodal_network(df, override_cutoff = [5, 8])

res_dict['reg_results'] = train_multimodal_model_segmented_and_pca(df, reg=True, override_cutoff = False)
res_dict['clf_results_all'] = train_multimodal_model_segmented_and_pca(df, reg=False, override_cutoff = [5, 20])
res_dict['clf_results_balanced'] = train_multimodal_model_segmented_and_pca(df, reg=False, override_cutoff = [5, 8])

from OutputsAnalyser.multimodal_classifiers import train_multimodal_model_segmented_and_pca
train_multimodal_model_segmented_and_pca(df)
print("Get total of {} subject after phq cutoff".format(len(df.Subject.unique())))

cv_results = {}
subj_prop = len(df[df.phq_label == 1].Subject.unique()) / len(df.Subject.unique())
print("Subject 1 proportaion : {} ".format(subj_prop))
print("Segment 1 proportaion : {} ".format(len(df[df.phq_label == 1]) / len(df)))

for sentype, sentype_df in df.groupby(["Sentence_type"]):
    cv_results[sentype] = {}
    y_vec = sentype_df.phq_label
    gss = GroupShuffleSplit(n_splits=6, train_size=.8, random_state=42)
    q = gss.get_n_splits()
    for iter, idx in enumerate(gss.split(sentype_df, y_vec, sentype_df.Subject)):
        train_idx, test_idx = idx[0], idx[1]

        print("ITER : ", iter)#, " TRAIN:", train_idx, "TEST:", test_idx)
        test_subjects = list(sentype_df.iloc[test_idx].Subject.unique())
        train_subjects = list(sentype_df.iloc[train_idx].Subject.unique())

        cv_results[sentype][iter] = {}
        cv_results[sentype][iter]["train_idx"] = train_idx
        cv_results[sentype][iter]["test_idx"] = test_idx
        cv_results[sentype][iter]["train_subject"] = train_subjects
        cv_results[sentype][iter]["test_subject"] = test_subjects

        featrep_est, featrep_pipe, featrep_score_seg, featrep_score_subj, featrep_pred, featrep_X_test, featrep_y_test = \
            seglearn_classification.featurerep_classifier(sentype_df[["Subject", "phq_label"]], seglearn_data, y_vec, train_idx, test_idx,
                                                          binary_threshold=1-subj_prop)
        # segment_crnn_clf, segment_crnn_score, segment_crnn_pred = seglearn_classification.segment_crnn_classifier(seglearn_data, y_vec, train_idx, test_idx)
        cv_results[sentype][iter]["featrep_est"] = featrep_est
        cv_results[sentype][iter]["featrep_pipe"] = featrep_pipe
        cv_results[sentype][iter]["featrep_score_seg"] = featrep_score_seg
        cv_results[sentype][iter]["featrep_score_subj"] = featrep_score_subj
        cv_results[sentype][iter]["tp,tn,fp,fn"] = ""

    per_iter_subj_score = [round(x["featrep_score_subj"],2) for x in cv_results[sentype].values()]
    per_iter_seg_score = [round(x["featrep_score_seg"],2) for x in cv_results[sentype].values()]
    print("For sentype : {}".format(sentype))
    print("Subject 1 proportaion : {} ".format(subj_prop))
    print("Segment 1 proportaion : {} ".format(len(sentype_df[sentype_df.phq_label == 1]) / len(sentype_df)))
    print("per iter seg score = {} ({})".format(np.mean(per_iter_subj_score), per_iter_subj_score))
    print("per_iter_subj_score = {} ({}) ".format(np.mean(per_iter_subj_score), per_iter_subj_score))
a = 2
# for subj_id in subjects:

