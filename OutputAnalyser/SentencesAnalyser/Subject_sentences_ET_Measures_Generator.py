import math
import pandas as pd
import datetime, os
import numpy as np
import colorama
from colorama import Fore, Back, Style
from GLOBALS import params
from OutputsAnalyser.et_analysis.AscParser import AscParser
from OutputsAnalyser.et_analysis.ET_SentenceProcessing_DfsAdder import ET_SentenceProcessing_DfsAdder

from OutputsAnalyser.et_analysis.EyeTrackingLinguisticMeasuresFetcher import EyeTrackingLinguisticMeasuresFetcher
from OutputsAnalyser import Utils

from enum import Enum
class SentimentEnum(Enum):
    POS = "pos"
    NEG = "neg"

class Subject_sentences_ET_Measures_Generator:

    def __init__(self, subj_id, use_existing_asc_parser = True, use_existing_sentence_et = True, use_existing_subject_et_processed_df = True, use_existing_subject_et_with_lingmeasures_processed_df = True):
        self.subj_id = subj_id
        subj_dir = params["exp_params"]["exp_dir"] + "/Output_2021_Aug/" + str(self.subj_id) + "/"
        self.subject_base_path = subj_dir + "/" + str(self.subj_id)
        self.dominant_eye = Utils.get_dominant_eye(subj_dir)
        self.use_existing_asc_parser = use_existing_asc_parser
        self.use_existing_sentence_et = use_existing_sentence_et
        self.use_existing_subject_et_processed_df = use_existing_subject_et_processed_df
        self.use_existing_subject_et_with_lingmeasures_processed_df = use_existing_subject_et_with_lingmeasures_processed_df

    def generate_subject_measures_df(self):
        print("Go to ET OutputsAnalyser")

        exp_output_df_fn = self.subject_base_path + "_Sentences_res.tsv"
        material_sentences_df_fn = params["exp_params"]["material_sentences_df_fn"]
        print('Reading: \nexp_output_df_fn : {}\nmaterial_sentences_df_fn: {}'.format(exp_output_df_fn, material_sentences_df_fn))

        exp_output_df = pd.read_csv(exp_output_df_fn, sep = "\t")
        material_sentences_df = pd.read_csv(material_sentences_df_fn)
        et_processed_df_fn = self.subject_base_path + "_Sentence_et_processed.csv"
        et_with_lingmeasures_processed_df_fn = self.subject_base_path + "_Sentence_et_processed_withLingMeasures.csv"

        subject_et_with_lingmeasures_processed_df, subject_gaze_route_df = self.create_new_subject_measures_df(exp_output_df, material_sentences_df, et_processed_df_fn, et_with_lingmeasures_processed_df_fn)
        subject_et_with_lingmeasures_processed_df.to_csv(et_processed_df_fn, index=False)
        print("{} Creaated new subject_et_with_lingmeasures_processed_df and saved it to {}".format(datetime.datetime.now(), et_processed_df_fn))
        return subject_et_with_lingmeasures_processed_df, subject_gaze_route_df


    def create_new_subject_measures_df(self, exp_output_df, material_sentences_df, et_processed_df_fn, et_with_lingmeasures_processed_df_fn):
        print("create new!")
        sentence_et_csv_fn = self.subject_base_path + "_Sentence_et.csv"

        if self.use_existing_sentence_et and os.path.exists(sentence_et_csv_fn):
            print("Read all_runs_merged_et_df from " , sentence_et_csv_fn)
            subject_all_runs_merged_et_df = pd.read_csv(sentence_et_csv_fn, index_col=None)
        else:
            task_ap = AscParser(self.subject_base_path, dominant_eye = self.dominant_eye, task_name="Sentences", use_existing=  self.use_existing_asc_parser)
            task_ap.set_exp_output_df(exp_output_df)
            subject_all_runs_merged_et_df = task_ap.asc2df()
            subject_all_runs_merged_et_df.to_csv(sentence_et_csv_fn, index = False)

        subject_gaze_route_df = subject_all_runs_merged_et_df[["Dominant_Eye_X", "Dominant_Eye_Y"]]
        if self.use_existing_subject_et_processed_df and os.path.exists(et_processed_df_fn):
            print("Read et_processed_df_fn from ", et_processed_df_fn)
            subject_et_processed_df = pd.read_csv(et_processed_df_fn, encoding="utf-8", index_col=False)
        else:
            merger = ET_SentenceProcessing_DfsAdder(subject_all_runs_merged_et_df, exp_output_df)
            subject_et_processed_df = merger.add_et_sentence_processing_data_per_row()
            print("Saving merged df (shape {})  to {} ".format(subject_et_processed_df.shape, et_processed_df_fn))
            subject_et_processed_df.to_csv(et_processed_df_fn, encoding="utf-8", index = False)

        print("Analyse linguistic et measures!")
        if self.use_existing_subject_et_with_lingmeasures_processed_df and os.path.exists(et_with_lingmeasures_processed_df_fn):
            print("Read subject_et_with_lingmeasures_processed_df from ", et_with_lingmeasures_processed_df_fn)
            subject_et_with_lingmeasures_processed_df = pd.read_csv(et_with_lingmeasures_processed_df_fn, index_col=False)
        else:
            measures_fetcher = EyeTrackingLinguisticMeasuresFetcher(self.subj_id, self.subject_base_path, subject_et_processed_df, exp_output_df, material_sentences_df)
            subject_et_with_lingmeasures_processed_df = measures_fetcher.fetch_linguistic_measures()
            subject_et_with_lingmeasures_processed_df.to_csv(et_with_lingmeasures_processed_df_fn, index = False)

        return subject_et_with_lingmeasures_processed_df, subject_gaze_route_df


    def generate_wordprobe_subject_row(self):
        try:

            wordprobe_output_df_fn = params["exp_params"]["exp_dir"] + "/Output_2021_Aug/" + str(self.subj_id) + "/" + str(self.subj_id) + "_Word_Probe_Task_res.tsv"

            wordprobe_df = pd.read_csv(wordprobe_output_df_fn, sep="\t")
            pos_df = wordprobe_df[wordprobe_df["target_sentiment"] == SentimentEnum.POS.value]
            neg_df = wordprobe_df[wordprobe_df["target_sentiment"] == SentimentEnum.NEG.value]

            positive_sentiment_RT = pos_df["pressed_RT"].mean()
            positive_sentiment_success_ratio = len(pos_df[pos_df["target_sentiment"] == True]) / len(pos_df)
            negative_sentiment_RT = neg_df["pressed_RT"].mean()
            negative_sentiment_success_ratio = len(neg_df[neg_df["target_sentiment"] == True]) / len(neg_df)

        except:
            colorama.init()
            print(Fore.YELLOW + "No {} file for subject {} ".format(wordprobe_output_df_fn, self.subj_id))
            print(Style.RESET_ALL)
            return [np.nan, np.nan, np.nan, np.nan]

        return [positive_sentiment_RT, positive_sentiment_success_ratio, negative_sentiment_RT, negative_sentiment_success_ratio]

    def generate_subject_pupil_vecs(self, subject_analysis_df, sentence_types, subject_id, mental_scores ):
        measures = []
        for sentence_type in sentence_types:

            set_num   = subject_analysis_df[subject_analysis_df["Sentence_type"] == sentence_type]["set_num"].iloc[0]
            try:
                pupil_vec = [float(x) for x in list(subject_analysis_df[subject_analysis_df["Sentence_type"] == sentence_type]["sentence_pupil_diameter_vec"].iloc[0][1:-1].split(","))]
            except AttributeError:
                pupil_vec = list(subject_analysis_df[subject_analysis_df["Sentence_type"] == sentence_type]["sentence_pupil_diameter_vec"].iloc[0])
            except ValueError:
                pupil_vec = []

            row = [subject_id] +  mental_scores + [set_num, sentence_type, pupil_vec]
            measures.append(row)

        return measures

    def get_subject_per_sentype_scores(self, subject_analysis_df, measures_keys, word_types, sentence_types):

        measures = []
        for sentence_type in sentence_types:
            for word_type in word_types:
                for measure_key in measures_keys:
                    sentence_type_word_type_rows = subject_analysis_df[(subject_analysis_df["word_type"] == word_type) & (subject_analysis_df["Sentence_type"] == sentence_type)]

                    if measure_key == 'skipping_probability':
                        is_skipped_trial_vec = sentence_type_word_type_rows["is_skipping_trial"]
                        val = is_skipped_trial_vec.mean()

                    else:
                        val = sentence_type_word_type_rows[measure_key].mean()

                    measures.append(val)

        return measures

    def generate_subject_row(self, measures, wordprobe_row, mental_scores):

        return [self.subj_id] + mental_scores + wordprobe_row + measures

    def generate_subject_gazeroute_vecs(self, gazeroute_df, mental_scores):

        x_vec = [float(x) for x in gazeroute_df["Dominant_Eye_X"] if x not in ["...","."]]
        y_vec = [float(x) for x in gazeroute_df["Dominant_Eye_Y"] if x not in ["...","."]]
        return [x_vec, y_vec]
