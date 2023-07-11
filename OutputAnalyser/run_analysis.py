import OutputsAnalyser.SentencesAnalyser.Heatmap_Visualizer as HM_VIS
import glob, random
import pandas as pd
import numpy as np
import shutil  #
from tensorflow.keras import models
import tensorflow as tf
from OutputsAnalyser.IbexFarm_OnlineStudy_ResultsAnalyser.questionnaires_csvs_parser import phq_analysis, state_analysis, trait_analysis
from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use('TkAgg')
#     from matplotlib import pyplot as plt
#     plt.plot(range(10))
#     plt.close()


import joblib, os

from OutputsAnalyser.SentencesAnalyser.GroupSentencesETAnalyser import GroupSentencesETAnalyser
from OutputsAnalyser.MentalAnalyser.MentalScoresAnalyser import MentalScoresAnalyser

OUTPUT_PATH = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/"

phq_responses_fn = f"{OUTPUT_PATH}/1_P_ET_questionnaire.csv"
state_responses_fn = f"{OUTPUT_PATH}/2_St_ET_questionnaire.csv"
trait_responses_fn = f"{OUTPUT_PATH}/3_Tr_ET_questionnaire.csv"

print("Make sure you download PHQ/State/Trait files - from "
      "https://docs.google.com/forms/d/1mv4AgGoilvAdMfVqBlCSi0wR8a2UIf8P5iCLKX5OO8E/edit#responses ,"
      "https://docs.google.com/forms/d/1R6oAmUv5FuvYtassqVtNUNPIffB3CMo_TJlMXKGZTUA/edit#responses ,"
      "https://docs.google.com/forms/d/1O2DXtmUWC9glDkkSvOY2af__MLc4_db_gSIaFXDru4o/edit#responses "
      "Dowload, take only relevant subjects, and put it in /Users/orenkobo/Desktop/PhD/HebLingStudy/Output/ under the name 1/2/3_P/St/Tr_questionnaire.csv ")



def get_analysis_output_dir_path(new_folder):
    base_analysis_path = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Analysis_2021_Aug/"
    analysis_output_dir_suffix = max([eval(name) for name in os.listdir(base_analysis_path) if
                                      name != '.DS_Store' and os.path.isdir(base_analysis_path + name)]) + new_folder

    analysis_output_dir_path = base_analysis_path + str(analysis_output_dir_suffix) + "/"
    if not os.path.exists(analysis_output_dir_path):
        os.makedirs(analysis_output_dir_path)
    print("Group analysis path: {} ({})".format(analysis_output_dir_path , "Created new folder" if new_folder else "Used last folder"))
    return analysis_output_dir_path

def get_run_num(saved_frame_filename, subject_id):
    if int(subject_id) > 16:
        return saved_frame_filename.split("run#")[1].split(".")[0]
    else:
        return 'X'

def sentype_to_vec(sentence_type_char):
    if sentence_type_char == "A":
        return [1,0,0,0]
    if sentence_type_char == "B":
        return [0,1,0,0]
    if sentence_type_char == "C":
        return [0,0,1,0]
    if sentence_type_char == "D":
        return [0,0,0,1]

# def retrieve_VGG_featuremap(vgg):
# 
#     return vgg_cnn.predict(vgg)

def main():

    analysis_output_dir_path = get_analysis_output_dir_path(new_folder = True)
    sentence_processing_group_df_fn = analysis_output_dir_path + "phq_level_to_scores_group_unified_analysis_df.csv"
    gaze_route_group_df_fn = analysis_output_dir_path + "phq_level_to_scores_group_gazeroute_df.csv"
    all_subjects_sentence_processing_df_fn = analysis_output_dir_path + "all_subjects_unified_sentence_processing_df.csv"
    all_subjects_pupil_vec_df_fn = analysis_output_dir_path + "all_subjects_pupil_vec_df.csv"
    all_subjects_gazeroute_df_fn = "all_subject_gaze_route_df.csv"
    subjects_word_probe_df_fn = analysis_output_dir_path + "subjects_word_probe_df.csv"
    all_sentences_all_subjects_df_fn = analysis_output_dir_path + "all_sentences_all_subjects_df.csv"
    output_for_DL_df_fn = f"{OUTPUT_PATH}/output_for_DL_df_ALL_new.csv"


    should_analyse_wordprobe = False
    should_analyse_sentence_processing = True
    should_draw_fixations_on_sentence = True
    remove_outliers_from_measures_graphs = False
    draw_slow_graphs = False
    draw_graphs = True
    draw_phq_analysis = False
    run_gaze_classification = False
    copy_videos_to_dir = False
    if remove_outliers_from_measures_graphs:
        analysis_output_dir_path += "/with_outliers/"
    else:
        analysis_output_dir_path += "/without_outliers/"



    subjects_to_disqualify = ["002","009","013","016","018","019", "028", "036",
                               "085","086","091","092","102","104"]
    # video_failed_subjects = ['026', '006', '039', '044', '041', '021',# '075',
    #                         '025', '065', '076' , '020']
    et_failed_subjects = ["098", "130", "129", "140"]
    # heatmap_failed_subjects = ["073", "060", "064"]
    no_phq_subjects = ["001","015","024", "045"]
    missing_data = ["075","073"]
    parse_code_failed_subjects = ["079","061","068"]

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

    # all_subjects_ids = [ "098", '026']#, '006', '039', '044', '041']

    valid_subjects_ids = [x for x in all_subjects_ids if x
                          not in subjects_to_disqualify
                          and x not in et_failed_subjects
                          and x not in no_phq_subjects]
    # subject_to_run_fixations = [x for x in valid_subjects_ids if x not in video_failed_subjects]
    subject_to_run_fixations = valid_subjects_ids
    demo = False
    if demo:
        valid_subjects_ids = all_subjects_ids[:4]


    print("Go over subjects {}. \nTotal {} valid and {} invalid".format(" , ".join(valid_subjects_ids), len(valid_subjects_ids), len(subjects_to_disqualify) ) )

    if draw_phq_analysis:
        ms_analyser = MentalScoresAnalyser(phq_responses_fn, state_responses_fn, trait_responses_fn, analysis_output_dir_path, valid_subjects_ids)
        ms_analyser.state_train_hist()
        ms_analyser.draw_phq_hist()
        ms_analyser.draw_phq_state_trait_correlation()

        exit()

    if should_analyse_wordprobe:
        from OutputsAnalyser.WordProbeAnalyser.WordProbeAnalyser import WordProbeAnalyser
        wp_analyser = WordProbeAnalyser(analysis_output_dir_path, phq_responses_fn, state_responses_fn, trait_responses_fn, valid_subjects_ids)
        wp_analyser.analyse_word_probe(subjects_word_probe_df_fn,  draw_graphs = draw_graphs)


    if should_analyse_sentence_processing:
        sp_analyser = GroupSentencesETAnalyser(analysis_output_dir_path, phq_responses_fn, state_responses_fn, trait_responses_fn, valid_subjects_ids)

        per_subject_sentence_unified_all_measures_processing_df, per_subject_gaze_route_group_df, per_subject_pupil_vec_per_sentence_type_df,\
            all_sentences_all_subjects_df = sp_analyser.generate_per_subject_dfs(all_subjects_sentence_processing_df_fn, all_subjects_gazeroute_df_fn,
                                             all_subjects_pupil_vec_df_fn, all_sentences_all_subjects_df_fn,
                                             use_existing_asc_parser = True, use_existing_sentence_et = True,
                                             use_existing_subject_et_processed_df = True,
                                             use_existing_subject_et_with_lingmeasures_processed_df = True )

        sp_analyser.generate_group_sentence_processing_df(sentence_processing_group_df_fn)
        group_gazeroute_df = sp_analyser.generate_group_gazeroute_df(gaze_route_group_df_fn)
        print("FINISHED!!")
        if draw_graphs:
            print("draw_graphs")
            sp_analyser.draw_sentence_graphs(draw_sentences=True, draw_ling_measures = True, draw_feats_corr = True,
                                             graph_types=["box", "violin"],
                                             remove_outliers= remove_outliers_from_measures_graphs,
                                             draw_slow_graphs = draw_slow_graphs)
            sp_analyser.draw_pupil_graphs(per_subject_pupil_vec_per_sentence_type_df, draw_slow_graphs)

        if run_gaze_classification:
            sp_analyser.gaze_classifiers(group_gazeroute_df, per_subject_gaze_route_group_df, all_sentences_all_subjects_df)


        all_sentences_all_subjects_df.to_csv(f"{OUTPUT_PATH}/all_sentences_all_subjects_df_FINAL.csv", index=False)
        saved_subjects = list(all_sentences_all_subjects_df.Subject.unique())
        print(" {} Saved subjs are {}".format(len(saved_subjects), saved_subjects))
        print("Saved all_sentences_all_subjects_df")
        # exit()
    if copy_videos_to_dir:
        from shutil import copy2
        import os
        import cv2

        # vid_dst_dir = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_semi_processed/videos/"
        scanpath_dst_dir = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_semi_processed/scanpaths/"

        for sid in subject_to_run_fixations:
            subject_videos_dir = f"{OUTPUT_PATH}/{sid}/videos"
            subject_scanpath_dir = f"{OUTPUT_PATH}/{sid}/scanpaths"
            # video_files = glob.glob(subject_videos_dir + "/*")
            scanpath_dirs = glob.glob(subject_scanpath_dir + "/*")

            for scanpath_dir in scanpath_dirs:
                for scanpath_file in glob.glob(scanpath_dir + "/*"):
                    if "video.avi" not in scanpath_file:
                        if int(scanpath_file.split("_")[-1].split(".j")[0]) % 100 == 0:
                            copy2(scanpath_file, scanpath_dst_dir)
                            print("Copied {} to {}".format(scanpath_file, scanpath_dst_dir))

            # for subj_vid_file in video_files:
            #     copy2(subj_vid_file, vid_dst_dir)
            #     print("Copied {} to {}".format(subj_vid_file, vid_dst_dir))

    vgg_dict = {}
    from datetime import datetime
    if should_draw_fixations_on_sentence:


        failed_subjects = []
        all_visualised_frames = []
        output_for_DL_rows = []
        errors = []

        measures_cols =  ["Sentence","sentence_pupil_diameter_mean", "word_pupil_diameter_mean", "word_has_first_pass_regression",
                              "trial_total_distance_covered", "is_skipping_trial", "regression_path_duration",
                              "num_of_fixations",  "total_gaze_duration", "second_fixation_duration",  "first_fixation_duration"
                                ]

        output_for_DL_cols = ["dst_video_fn", "num_video_frames", "frame_num", "frame_fn", "heatmap_dst_fn",
                                 "subject_id", "set_num", "sentence_idx", "run_num",
                                 "trial_num", "sentence_type", "sentence_type_isA","sentence_type_isB","sentence_type_isC","sentence_type_isD"] \
                             + measures_cols + ["phq_score","phq_group", "frame_vgg_3d_data" , "sentence_encoding"]

        draw_cou = 0

        all_sentences_info_df = pd.read_csv("/Users/orenkobo/Desktop/PhD/HebLingStudy/Task_Sentences/Materials/ExpSentences.csv")
        dir_of_all_raw_sentences_print_screens = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Task_Sentences/Materials/printscreens_all_sentences/"
        import os
        print("Run fixations on " , subject_to_run_fixations)
        for subject_id in subject_to_run_fixations:
            if 1==1:
            # try:
                print("{} ==== Subject : {} ====".format(datetime.now().time() , subject_id))

                subject_output_dir = f"{OUTPUT_PATH}/{subject_id}/"

                subject_videos_dir = subject_output_dir + "videos"
                subject_scanpaths_dir = subject_output_dir + "scanpaths"
                subject_heatmap_dir = subject_output_dir + "heatmaps"
                subject_ps_from_exp = subject_output_dir + "existing_frames_ps"
                # import os
                for folder in [subject_videos_dir, subject_scanpaths_dir, subject_heatmap_dir, subject_ps_from_exp]:
                    if not os.path.exists(folder):
                        print("create folder ", folder)
                        os.mkdir(folder)

                subject_sentenecs = pd.read_csv(f"{OUTPUT_PATH}/{subject_id}/{subject_id}_Sentences_res.tsv", sep = "\t")
                measures_df = pd.read_csv(f"{OUTPUT_PATH}/{subject_id}/{subject_id}_Sentence_et_processed_withLingMeasures.csv")
                et_df = pd.read_csv(f"{OUTPUT_PATH}/{subject_id}/{subject_id}_Sentence_et.csv")

                move_subject_existing_frames(subject_id, subject_ps_from_exp)
                subject_sentenecs.loc[subject_sentenecs['set_num'] == 99 , 'set_num'] = \
                    100 + subject_sentenecs.reset_index()["index"]

                for set_num in subject_sentenecs.set_num.unique():

                    sentence_info = subject_sentenecs[subject_sentenecs.set_num == set_num]
                    sentence_type = sentence_info.sentence_type.iloc[0]
                    sentence_type_vec = sentype_to_vec(sentence_type)
                    # sentence_encoding = sent2enc_dict[sentence_info.sentence_text.iloc[0]]
                    set_num_to_check = 99 if set_num >= 100 else set_num
                    sentence_idx = list(all_sentences_info_df[(all_sentences_info_df.set_num == set_num_to_check) & (all_sentences_info_df.sen_type == sentence_type)].index)[0]
                    assert (len(sentence_info) == 1)
                    presented_sentence_type_from_set = sentence_info.sentence_type.iloc[0]
                    sentence_printscreen_fn = "{}sentence_set#{}_type{}.png".format(dir_of_all_raw_sentences_print_screens,
                                                                                    set_num_to_check, presented_sentence_type_from_set)
                    trial_num = sentence_info.trial_num.iloc[0]
                    run_num = sentence_info.run_num.iloc[0]

                    text = sentence_info.sentence_text.iloc[0]
                    if set_num_to_check == 99 :

                        sentence_measures_df = measures_df[(measures_df.Subject == int(subject_id)) &
                                                    (measures_df.sentence_trial_num == trial_num) &
                                                    (measures_df.sentence_run_num == run_num)]
                    else:
                        sentence_measures_df = measures_df[(measures_df.Subject == int(subject_id)) &
                                                           (measures_df.sentence_trial_num == trial_num) &
                                                           (measures_df.sentence_run_num == run_num) &
                                                           (measures_df.word_type == "source_word")]


                    assert (len(sentence_measures_df) == 1)
                    print("{} : {}: Work on subject {}, set {} , trial {}, run {} (set {} with type {} : {})".format(datetime.now().time() , draw_cou, subject_id, set_num, trial_num,
                                                                                                    run_num, set_num_to_check, presented_sentence_type_from_set,
                                                                                                    text))

                    trial_et_df = et_df[et_df["trial_num"] == trial_num]
                    x = trial_et_df["Dominant_Eye_X"]
                    y = trial_et_df["Dominant_Eye_Y"]
                    create_scanpath = False
                    if create_scanpath:

                        scandir = HM_VIS.draw_scanpath_on_image(np.column_stack((list(trial_et_df["Dominant_Eye_X"]),
                                                                                 list(trial_et_df["Dominant_Eye_Y"]))), sentence_printscreen_fn,
                                                                sentence_idx, run_num, subject_scanpaths_dir, subject_id, trial_num,
                                                                rescale_image_size = True)

                        video_fn = HM_VIS.scanpath_to_vid(scandir, trial_num, run_num, subject_id, sentence_idx)
                        dst_video_fn = subject_videos_dir + "/subject#{}_run#{}_trial#{}_sentence#{}_scan_path_video.avi".format(subject_id, run_num, trial_num, sentence_idx)
                        cp_cmd = "cp {} {}".format(video_fn, dst_video_fn)
                        os.system(cp_cmd)
                        print("{} : Copied video from {} to {}".format(datetime.now().time() , video_fn , subject_videos_dir))

                    heatmap_dst_fn = "{}/{}_run#{}_trial#{}_heatmap.png".format(subject_heatmap_dir, subject_id, run_num, trial_num)
                    vgg_heatmap_dst_fn = heatmap_dst_fn.replace(".png","") + "_vgg.jbl"
                    vgg_featuremap_fn = vgg_heatmap_dst_fn.replace(".jbl","") + "_feature_map.jbl"

                    print("{} Saved heatmap to {}".format(datetime.now(), heatmap_dst_fn))
                    HM_VIS.draw_heatmap_on_image(x, y, sentence_printscreen_fn, dest_fn=heatmap_dst_fn)
                    all_visualised_frames.append(sentence_printscreen_fn)

                    # heatmap_vgg_enc = fn2img(heatmap_dst_fn)
                    # vgg_featuremap = retrieve_VGG_featuremap(heatmap_vgg_enc)
                    # joblib.dump(heatmap_vgg_enc, vgg_heatmap_dst_fn)
                    # joblib.dump(vgg_featuremap, vgg_featuremap_fn)
                    # print("{} Saved VGG 224X224X3 enc to {}".format(datetime.now(), vgg_heatmap_dst_fn))
                    # print("{} Saved VGG 1D 4096*1 feature map to {}".format(datetime.now() , vgg_featuremap_fn))


                    draw_cou += 1
                    # except Exception as e:
                    #     print("Failed for {} : {}".format(saved_frame_filename,e))

                    # create_vgg_dict = False
                    # all_frames = glob.glob(scandir + "*")

                    # if create_vgg_dict:
                    #
                    #     for frame_fn in all_frames:
                    #         if not "video.avi" in frame_fn:
                    #             frame_num = int(frame_fn.split("_path_")[1].split(".jpg")[0])
                    #             if frame_num % 50 == 0:
                    #                 vgg_dict[frame_fn] = fn2img(frame_fn)
                    #             a = 2
                    # else:
                    #     phq_group, phq_score = phq_analysis("/Users/orenkobo/Desktop/PhD/HebLingStudy/Output/1_P_ET_questionnaire.csv", int(subject_id))
                    #     for frame_fn in all_frames:
                    #         if not "video.avi" in frame_fn:
                    #             frame_num = int(frame_fn.split("_path_")[1].split(".jpg")[0])
                    #             if frame_num % 100 == 0:
                    #                 frame_vgg_3d_data = fn2img(frame_fn)
                    #                 new_output_row = [dst_video_fn, num_video_frames, frame_num, frame_fn, heatmap_dst_fn,
                    #                                   subject_id, set_num, sentence_idx, run_num, trial_num, sentence_type]  + \
                    #                                  sentence_type_vec + \
                    #                      list(sentence_measures_df[measures_cols].iloc[0].values) + \
                    #                                  [phq_group, phq_score, frame_vgg_3d_data] + [sentence_encoding]
                    #
                    #                 output_for_DL_rows.append(new_output_row)


                print("Finished subject ", subject_id)
            # except Exception as e:
            #     failed_subjects.append(subject_id)
            #     errors.append(e)
        print("CV Failed subjects are : ", failed_subjects)
        print("CV Errors are : ", errors)
        # print("Saving dict")
        # joblib.dump(vgg_dict , "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output/framename2vgg_dict_ALL.pkl")
        # print("Dict saved")
        print("SAVING!!!")
        output_for_DL_df = pd.DataFrame(data=output_for_DL_rows, columns=output_for_DL_cols)
        output_for_DL_df = output_for_DL_df.rename({"Sentence": "sentence_text"})
        output_for_DL_df.to_csv(output_for_DL_df_fn, index=False)
        print("Saved output_for_DL_df to ", output_for_DL_df_fn)


def get_num_frames(scandir):
    all_files = [x for x in glob.glob(scandir + "*") if not "video.avi" in x]
    max_frame = sorted(all_files, key = lambda x : int(x.split("_path_")[1].split(".jpg")[0]))[-1].split("_path_")[1].split(".jpg")[0]
    return int(max_frame)


def move_subject_existing_frames(subject_id, dst_dir, task_type = "sentence"):

    if int(subject_id) > 16:
        saved_sentences_frames_filenames = [x for x in glob.glob(f"{OUTPUT_PATH}/{subject_id}/{task_type}#*_frame_run*.png") if "heatmap" not in x]
    else:
        saved_sentences_frames_filenames = [x for x in glob.glob(f"{OUTPUT_PATH}/{subject_id}/{task_type}#*_frame*.png") if "heatmap" not in x]

    for f in saved_sentences_frames_filenames:
        shutil.move(f, dst_dir)
    print("Moved frames ps to " , dst_dir)


if __name__ == '__main__':
    main()