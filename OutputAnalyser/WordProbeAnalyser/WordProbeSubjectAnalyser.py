import random, glob
import pandas as pd
import numpy as np
import codecs, os
from GLOBALS import params
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from OutputsAnalyser.Utils import phq_analysis, state_analysis, trait_analysis
from OutputsAnalyser import Utils

def rt2float(x):
    if x == "None":
        return np.nan
    return float(x)

class WordProbeSubjectAnalyser:

    def __init__(self, analysis_output_dir_path, subject_id,phq_responses_fn, state_responses_fn, trait_responses_fn, debug_mode = False):
        self.subject_id = subject_id
        self.analysis_output_dir_path = analysis_output_dir_path

        self.phq_responses_fn = phq_responses_fn
        self.state_responses_fn = state_responses_fn
        self.trait_responses_fn = trait_responses_fn
        self.subject_dir = params["exp_params"]["exp_dir"] + "/Output_2021_Aug/" + str(subject_id) + "/"
        self.dominant_eye = Utils.get_dominant_eye(self.subject_dir) ##

        wordprobe_output_asc_fn =  self.subject_dir + str(subject_id) + "etW_0.asc"

        self.asc_file = codecs.open(wordprobe_output_asc_fn, encoding='utf-8-sig')
        self.asc_data = self.asc_file.readlines()
        self.run_asc_df = pd.DataFrame(self.asc_data, columns=['data'])

        if not debug_mode:
            print("parser et df : {}".format(wordprobe_output_asc_fn))
            self.run_asc_df = self.run_asc_df['data'].apply(lambda x: pd.Series(x.split('\t')))
            self.run_asc_df = self.run_asc_df[[0, 1, 2]]
            print("Done parsing")

        self.behav_csv_fn = self.subject_dir + str(subject_id) + "_Word_Probe_Task_res.tsv"
        self.behav_df = pd.read_csv(self.behav_csv_fn, sep = "\t")

        self.num_trials = len(self.behav_df)

    def get_mental_scores(self):
        phq = phq_analysis(self.phq_responses_fn, self.subject_id)

        if len(phq) == 1:
            phq_level = np.nan
            phq_score = np.nan
        else:
            phq_level = phq[0]
            phq_score = phq[1]

        state_score = state_analysis(self.state_responses_fn, self.subject_id)
        trait_score = trait_analysis(self.trait_responses_fn, self.subject_id)
        return phq_level, phq_score, state_score, trait_score

    def draw_single_subject_mean_RT(self, draw = False):
        df = self.behav_df[self.behav_df["pressed_RT"].apply(lambda x : str(x) != "None")]
        pos_rt = df[df["target_sentiment"] == "pos"][["pressed_RT", "target_sentiment"]]
        neg_rt = df[df["target_sentiment"] == "neg"][["pressed_RT", "target_sentiment"]]
        pos_rt["pressed_RT"] = pos_rt["pressed_RT"].apply(lambda x : float(x))
        neg_rt["pressed_RT"] = neg_rt["pressed_RT"].apply(lambda x: float(x))

        if draw:
            t,p = stats.ttest_ind([float(x) for x in pos_rt["pressed_RT"].values], [float(x) for x in neg_rt["pressed_RT"].values])

            df = pd.DataFrame(data=pd.concat([pos_rt, neg_rt]) , columns=["pressed_RT","target_sentiment"])
            plot_box = sns.boxplot(x = "target_sentiment", y = "pressed_RT", data=df)
            box_output_fn = "{}WordProbe_RT_box_plot_subject#{}.png".format(self.analysis_output_dir_path, self.subject_id)
            plot_box.get_figure().text(0.05, 0.95, "\nt = {}\np = {}".format(round(t,2), round(p,2)))
            plot_box.figure.savefig(box_output_fn)
            print("Saved RT boxplot to {}".format(box_output_fn))
            plt.close()

        return pos_rt["pressed_RT"].mean(), neg_rt["pressed_RT"].mean(), list(pos_rt["pressed_RT"].values), list(neg_rt["pressed_RT"].values)

    def draw_single_subject_mean_gazeduration(self, draw = False):

        print("Analyse gaze duration for subject #{}, go over {} trials".format(self.subject_id, self.num_trials))
        df = pd.DataFrame(columns=["Word_type" ,"Gaze_duration"])
        for trial_num in range(self.num_trials):
            trial_behav_row = self.behav_df[self.behav_df.trial_num == trial_num]

            trial_pos_time, trial_neg_time, na_time = Utils.get_trial_per_sentiment_gazeduration(self.run_asc_df, trial_num, trial_behav_row)

            df = df.append({"Word_type" : "Pos", "Gaze_duration" : trial_pos_time} , ignore_index=True)
            df = df.append({"Word_type" : "Neg", "Gaze_duration" : trial_neg_time} , ignore_index=True)
            df = df.append({"Word_type" : "NA" , "Gaze_duration" : na_time} , ignore_index=True)

        if draw:
            df["Gaze_duration_float"] = [float(x) for x in df["Gaze_duration"]]
            plot_box = sns.boxplot(x = "Word_type", y = "Gaze_duration_float", data=df)
            box_output_fn = "{}WordProbe_GazeDuration_box_plot_subject#{}.png".format(self.analysis_output_dir_path, self.subject_id)
            plot_box.figure.savefig(box_output_fn)
            print("Saved GazeDuration boxplot to {}".format(box_output_fn))
            plt.close()
            plt.clf()
            plt.cla()

        pos_df_gd = df[df["Word_type"] == "Pos"]["Gaze_duration"]
        neg_df_gd = df[df["Word_type"] == "Neg"]["Gaze_duration"]
        na_df_gd = df[df["Word_type"] == "NA"]["Gaze_duration"]
        return pos_df_gd.mean(), neg_df_gd.mean(), na_df_gd.mean() , list(pos_df_gd.values), list(neg_df_gd.values), list(na_df_gd.values)

