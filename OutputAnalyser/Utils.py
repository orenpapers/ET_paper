import pandas as pd
from behav_runner.Task.SentimentEnum import SentimentEnum
# from behav_runner.Task.Task_Executors.WordProbeTask_Executor import SentimentEnum
from behav_runner.Utilities.ETUtilities import Events_Enum, sendMessage, task_msg
from behav_runner.Task.Task_Executors.TaskExecutorsInterface import TaskTypes
import numpy as np
from colorama import Fore, Style

def get_dominant_eye(subject_dir):

    personal_details_fn = subject_dir + "/personalDetails.txt"
    with open(personal_details_fn,  "r") as pers_dets:
        lines = pers_dets.readlines()
        eye = lines[1].split("\t")[-1].replace("\n","")
        return eye

def get_stai_group(score):

    if score in range(0, 37): #0 to 4 is Minimal
        return "no or low anxiety"

    if score in range(37, 44): #5 to 9 is Mild
        return "moderate anxiety"

    if score in range(44, 80): #10 to 14 is Moderate
        return "high anxiety"


    return np.nan

def get_phq_group(phq_score):

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

def phq_analysis(phq_fn, subj_id):

    phq_df = pd.read_csv(phq_fn)
    if len(phq_df[phq_df["sid"] == int(subj_id.replace("00",""))]) == 0 and len(phq_df[phq_df["sid"] == int(subj_id)]) == 0:
        print(Fore.YELLOW + "NO PHQ SCORE FOR ", subj_id)
        print(Style.RESET_ALL)
        return [np.nan, np.nan]

    phq_score = sum(phq_df[phq_df["sid"] == int(subj_id.lstrip("0"))].iloc[:, 2:-1].values[0])
    phq_group = get_phq_group(phq_score)
    return [phq_score, phq_group]


def state_analysis(state_fn, subj_id, only_score = True):
    state_df = pd.read_csv(state_fn)
    if len(state_df[state_df["sid"] == int(subj_id.replace("00",""))]) == 0:
        print(Fore.YELLOW + "NO STATE SCORE FOR ", subj_id)
        print(Style.RESET_ALL)
        return [np.nan]
    state_score = sum(state_df[state_df["sid"] == int(subj_id.replace("00",""))].iloc[:, 2:].values[0])
    if only_score:
        return [state_score]

    state_group = get_stai_group(state_score)
    return [state_score, state_group]


def trait_analysis(trait_fn, subj_id, only_score = True):
    trait_df = pd.read_csv(trait_fn)
    if len(trait_df[trait_df["sid"] == int(subj_id.replace("00",""))]) == 0:
        print(Fore.YELLOW + "NO Trait SCORE FOR ", subj_id)
        print(Style.RESET_ALL)
        return [np.nan]
    trait_score = sum(trait_df[trait_df["sid"] == int(subj_id.replace("00",""))].iloc[:, 2:].values[0])
    if only_score:
        return [trait_score]
    trait_group = get_stai_group(trait_score)
    return [trait_score, trait_group]

def get_duo_df(run_df, trial_num):

    WordProbeTaskStartStr = task_msg(Events_Enum.Duo, TaskTypes.WORD_PROBE) + "_displayduo#{}".format(trial_num)
    WordProbeTaskEndStr = task_msg(Events_Enum.Duo, TaskTypes.WORD_PROBE) + "_removedduo#{}".format(trial_num)

    start_idx = run_df[run_df[1].str.contains(WordProbeTaskStartStr, na=False)].index[0] + 1
    end_idx = run_df[run_df[1].str.contains(WordProbeTaskEndStr, na=False)].index[0] - 1

    duo_df = run_df[start_idx:end_idx]

    # print("After taking only sentence-reading rows, down to {} lines".format(duo_df.shape))
    duo_df = duo_df[duo_df[1].str.strip() != "."]
    # print("After Filterting unfixations, rows, down to {} lines".format(duo_df.shape))

    duo_df = duo_df[duo_df[0].str.contains("MSG") == False]
    duo_df = duo_df[duo_df[0].str.contains("L") == False]
    return duo_df


def get_trial_per_sentiment_gazeduration(run_asc_df, trial_num, trial_row):
    from GLOBALS import params
    width = params["exp_params"]["width"]
    mid_screen = width / 2
    gap = 100
    left_x_d, right_x_d = params["Word_Probe_Task"]["x_tuple"]
    left_x, right_x = mid_screen + left_x_d, mid_screen + right_x_d
    pos_x = left_x if trial_row["left_sentiment"].iloc[0] == SentimentEnum.POS.value else right_x
    neg_x = left_x if trial_row["left_sentiment"].iloc[0] == SentimentEnum.NEG.value else right_x

    duo_df = get_duo_df(run_asc_df, trial_num)
    Gaze_L_X = duo_df[1].dropna().apply(lambda x: float(x.strip()))

    trial_pos_timestamps = [x for x in Gaze_L_X if x < pos_x + gap and x > pos_x - gap]
    trial_neg_timestamps = [x for x in Gaze_L_X if x < neg_x + gap and x > neg_x - gap]

    trial_pos_time = len(trial_pos_timestamps) * params["et_rate"]
    trial_neg_time = len(trial_neg_timestamps) * params["et_rate"]
    na_time = float(len(Gaze_L_X)) - float(len(trial_pos_timestamps)) - float(len(trial_neg_timestamps))

    return trial_pos_time, trial_neg_time, na_time