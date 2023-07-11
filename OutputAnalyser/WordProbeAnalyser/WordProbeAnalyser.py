import pandas as pd
import numpy as np
import datetime, os
from OutputsAnalyser.WordProbeAnalyser.WordProbeSubjectAnalyser import WordProbeSubjectAnalyser
from OutputsAnalyser.WordProbeAnalyser.WordProbeGroup_Analyser import WordProbeGroup_Analyser
from GLOBALS import params
import pprint

class WordProbeAnalyser:

    def __init__(self,  analysis_output_dir_path, phq_responses_fn, state_responses_fn, trait_responses_fn ,subjects_ids ):
        self.analysis_output_dir_path = analysis_output_dir_path
        self.subjects_ids = subjects_ids
        self.trait_responses_fn = trait_responses_fn
        self.phq_responses_fn = phq_responses_fn
        self.state_responses_fn = state_responses_fn

        print("Analyse Word Probe to ", analysis_output_dir_path)

    def analyse_word_probe(self, subjects_word_probe_df_fn,  draw_graphs = False):

        debug_mode = False
        subjects_word_probe_df = pd.DataFrame(columns=["subject_id", "word_type", "mean_RT", "mean_GD"])
        print("{} analyse_word_probe".format(datetime.datetime.now()))
        failed_subjects_dict = {}
        for subject_id in self.subjects_ids:

            btime = datetime.datetime.now()

            print("subject_id is ", subject_id)


            if not os.path.exists(params["exp_params"]["exp_dir"] + "/Output_2021_Aug/" + str(subject_id) + "/" + str(subject_id) + "etW_0.asc") or not os.path.exists(params["exp_params"]["exp_dir"] + "/Output_2021_Aug/" + str(subject_id) + "/" + str(subject_id) + "_Word_Probe_Task_res.tsv"):
                err = "No etW_0.asc or Word_Probe_Task_res file for {}".format(subject_id)
                failed_subjects_dict[subject_id] = err
                continue

            word_probe_subject_analyser = WordProbeSubjectAnalyser(self.analysis_output_dir_path, subject_id = subject_id,
                                                                   phq_responses_fn = self.phq_responses_fn,
                                                                   state_responses_fn = self.state_responses_fn,
                                                                   trait_responses_fn = self.trait_responses_fn,
                                                                   debug_mode = debug_mode)

            mean_pos_rt, mean_neg_rt, pos_rt_vec, neg_rt_vec = word_probe_subject_analyser.draw_single_subject_mean_RT(draw = False)
            mean_pos_gd, mean_neg_gd, mean_na_gd, pos_gd_vec, neg_gd_vec, nan_gd_vec = word_probe_subject_analyser.draw_single_subject_mean_gazeduration(draw = False)

            phq_score, phq_level, state_score, trait_score = word_probe_subject_analyser.get_mental_scores()

            subject_pos_row = {"subject_id": subject_id, "word_type": "Pos", "mean_RT": mean_pos_rt,
                               "mean_GD": mean_pos_gd, "RT_vec": pos_rt_vec, "GD_vec": pos_gd_vec,
                               "phq_score" : phq_score, "phq_level" : phq_level, "state_score" : state_score, "trait_score": trait_score}
            subject_neg_row = {"subject_id": subject_id, "word_type": "Neg", "mean_RT": mean_neg_rt,
                               "mean_GD": mean_neg_gd, "RT_vec": neg_rt_vec, "GD_vec": neg_gd_vec,
                               "phq_score": phq_score, "phq_level": phq_level, "state_score": state_score,"trait_score": trait_score}
            subject_na_row = {"subject_id": subject_id, "word_type": "NA", "mean_RT": np.nan, "mean_GD": mean_na_gd,
                              "RT_vec": np.nan, "GD_vec": nan_gd_vec,
                              "phq_score": phq_score, "phq_level": phq_level, "state_score": state_score[0], "trait_score": trait_score[0]}

            subjects_word_probe_df = subjects_word_probe_df.append(subject_pos_row, ignore_index=True)
            subjects_word_probe_df = subjects_word_probe_df.append(subject_neg_row, ignore_index=True)
            subjects_word_probe_df = subjects_word_probe_df.append(subject_na_row, ignore_index=True)
            etime = datetime.datetime.now()
            print("Added subject {} to subjects_word_probe_df ({} time)".format(subject_id, etime-btime))

        print("{} : Done creating subjects_word_probe_df per subject".format(datetime.datetime.now()))
        subjects_word_probe_df.to_csv(subjects_word_probe_df_fn, index=False)
        print("Saved subjects_word_probe_df to ", subjects_word_probe_df_fn)

        print("{} Group analysis of word probe!! {}".format(datetime.datetime.now() , " , ".join(self.subjects_ids)))

        word_probe_group_analyser = WordProbeGroup_Analyser(self.analysis_output_dir_path, subjects_word_probe_df, debug_mode=debug_mode)

        if draw_graphs:
            print("{} Draw word_probe_group_analyser graphs".format(datetime.datetime.now()))
            word_probe_group_analyser.draw_per_subject_facet_barplot(graph_type="GD")
            word_probe_group_analyser.draw_per_subject_facet_barplot(graph_type="RT")
            word_probe_group_analyser.draw_grouplevel_effect()

        print("Finished analyse_word_probe!!")
        print("==========================")
        print("{} Failed subject :".format(len(failed_subjects_dict)))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(failed_subjects_dict)