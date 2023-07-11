import pandas as pd
from OutputAnalyser.TimeSeriesAnalyser import input_csv_fn, heatmap_input_csv_fn
from OutputAnalyser.TimeSeriesAnalyser import add_label_column, add_x_vec_len, add_train_test_split, add_videosegementnum_col
from OutputAnalyser.TimeSeriesAnalyser import samples_validation
from OutputAnalyser.TimeSeriesAnalyser import apply_mean_pooling, apply_smoothing_1d
from OutputAnalyser.TimeSeriesAnalyser import phq2int_dict
import joblib
from OutputAnalyser.TimeSeriesAnalyser import analysis_output_dir

# https://colab.research.google.com/drive/1oYoCUABe4YO-jkcUAst2cDyRDVeYwHfm?authuser=2#scrollTo=e-Pk9C-uTwxD

def read_rawdata_based_df(df_fn):
    df_orig = pd.read_csv(df_fn)
    df = df_orig[df_orig.word_type == "target_word"]
    df = add_label_column(df)
    df = add_x_vec_len(df)
    df = add_train_test_split(df)
    df = df.dropna(subset=["phq_group", "x_gaze_location"])
    df = samples_validation(df)
    return df

def read_heatmap_based_df(df_fn):
    print("Read data")
    df_orig = pd.read_csv(df_fn)
    print("data shape is " , df_orig.shape)
    print("Data cols are " , list(df_orig))
    df = df_orig.dropna(subset=["phq_group"])
    df = add_videosegementnum_col(df , frames_per_segment = 5)
    df = add_train_test_split(df, id_col='subject_id')
    return df

def process_labels(df , binary_cutoff = "med"):
    if binary_cutoff == "med":
        cutoff = df.phq_score.median()
    df['binary_label'] = df.phq_score.apply(lambda x : 1 if x > cutoff else 0)
    df['phq_level'] = df.phq_group.apply(lambda x : phq2int_dict[x])
    return df, cutoff


def train_heatmapbased_classifiers(df, steps):
    from OutputAnalyser.TimeSeriesAnalyser import video_classification_manager
    accuracy_dict = {}
    models_dict = {}
    if "DL_video" in steps:
        video_classification_dict, video_models_dict = video_classification_manager(df)

def train_rawdata_classifiers(df, df_of_nested_ts, contextual_cols, steps):
    from OutputAnalyser.TimeSeriesAnalyser import dsp_classification_manager
    from OutputAnalyser.TimeSeriesAnalyser import dl_classification_manager
    from OutputAnalyser.TimeSeriesAnalyser import raw_classification_manager

    accuracy_dict = {}
    models_dict = {}
    series_of_timeseries = df_of_nested_ts.pooled_data

    if "DL_timeseries" in steps:
        dl_classification_dict, dl_models_dict = dl_classification_manager(df, series_of_timeseries, contextual_cols)
        accuracy_dict["DL_classification"] = dl_classification_dict
        models_dict["DL_models"] = dl_models_dict

    if "raw" in steps:
        raw_classifictation_dict , raw_models_dict = raw_classification_manager(df, series_of_timeseries, contextual_cols,
                                                                                algs = ["early", "kn"])
        accuracy_dict["rawdata_classification"] = raw_classifictation_dict
        models_dict["rawdata_models"] = raw_models_dict

    if "DSP" in steps:
        dsp_classification_dict, dsp_models_dict = dsp_classification_manager(df, series_of_timeseries, contextual_cols)
        accuracy_dict["DSP_classification"] = dsp_classification_dict
        models_dict["DSP_models"] = dsp_models_dict

    #todo implement https://github.com/marcellacornia/sam
    return accuracy_dict, models_dict

def main():
    steps = ["raw", "DSP", "DL_timeseries","DL_heatmap_CNN","DL_video"]
    list_of_acc_dicts = []
    list_of_model_dicts = []
    sc_dict = {}
    df_heatmap_based = read_heatmap_based_df(heatmap_input_csv_fn)
    df_heatmap_based, cutoff = process_labels(df_heatmap_based)

    df = read_rawdata_based_df(input_csv_fn)
    df, cutoff = process_labels(df)


    for cutoff in [6]:
        sc_dict["value_counts"] = df.phq_group.value_counts()
        sc_dict["phq_levels"]= phq2int_dict
        sc_dict["binary_cuttoff"] = cutoff

        sc_dict["heatmap_based_classifiers"] = train_heatmapbased_classifiers(df_heatmap_based, steps=["DL_video"])
        sc_dict["raw_data_classifiers"] = {}
        for sen_type, sen_type_df in df.groupby(["Sentence_type"]):
            series_of_nesed_timeseries = pd.DataFrame(sen_type_df.x_gaze_location.apply(lambda x : eval(x)))
            df_of_nested_ts_raw = pd.DataFrame(data = series_of_nesed_timeseries.apply(lambda x: apply_mean_pooling(x[0]) , axis=1) , columns = ['pooled_data'])
            df_of_nested_ts = pd.DataFrame(data = df_of_nested_ts_raw.pooled_data.apply(lambda x: apply_smoothing_1d(x)) , columns = ['pooled_data'])


            sen_type_accuracy_dict , models_dict = train_rawdata_classifiers(df = sen_type_df, df_of_nested_ts = df_of_nested_ts,  #a,
                                                                             contextual_cols = ['first_fixation_duration','regression_path_duration','word_idx'],
                                                                             steps=steps)

            print("For sen type {} - {}".format(sen_type, sen_type_accuracy_dict))
            sc_dict["raw_data_based_classifiers"][sen_type]= sen_type_accuracy_dict
            list_of_model_dicts.append(models_dict)
            list_of_acc_dicts.append(sen_type_accuracy_dict)
    print(sc_dict)

    joblib.dump(list_of_acc_dicts , analysis_output_dir + "list_of_acc_dicts.jbl")
    joblib.dump(list_of_model_dicts , analysis_output_dir + "list_of_models_dicts.jbl")


if __name__ == '__main__':
    main()