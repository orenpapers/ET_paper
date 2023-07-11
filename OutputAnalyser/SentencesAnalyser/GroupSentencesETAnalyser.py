import pandas as pd
import numpy as np
import os
from OutputsAnalyser.SentencesAnalyser.PupilProcessing_GraphGenerator import PupilProcessing_GraphGenerator

from OutputsAnalyser.SentencesAnalyser.sentenceprocessing_unified_df_Generator import sentenceprocessing_unified_df_Generator
from OutputsAnalyser.SentencesAnalyser.ET_ReadingMeasuresFetcher import ET_ReadingMeasuresFetcher
from OutputsAnalyser.SentencesAnalyser.SequenceClassifier import SequenceClassifier
from OutputsAnalyser.SentencesAnalyser.SentencesAnalyser import SentencesAnalyser
import datetime

def load_existing_dfs(all_subjects_df_fn, group_df_fn):
    group_df = pd.read_csv(group_df_fn)
    sentence_processing_all_subjects_df = pd.read_csv(all_subjects_df_fn)
    return group_df, sentence_processing_all_subjects_df


class GroupSentencesETAnalyser:

    def __init__(self,  analysis_output_dir_path, phq_responses_fn, state_responses_fn, trait_responses_fn ,subjects_ids):
        self.analysis_output_dir_path = analysis_output_dir_path
        self.subjects_ids = subjects_ids
        self.trait_responses_fn = trait_responses_fn
        self.phq_responses_fn = phq_responses_fn
        self.state_responses_fn = state_responses_fn
        print("Analyse Sentence Processing to ", analysis_output_dir_path)

    def generate_per_subject_dfs(self, all_subjects_sentence_processing_df_fn, all_subjects_gazeroute_df_fn,
                                 all_subjects_pupil_vec_df_fn, all_sentences_all_subjects_df_fn,
                                 use_existing_asc_parser=False, use_existing_sentence_et=False,
                                 use_existing_subject_et_processed_df=False,
                                 use_existing_subject_et_with_lingmeasures_processed_df=False):

        print("{} analyse_sentences_group_ET_pattern".format(datetime.datetime.now()))

        if not os.path.exists(self.analysis_output_dir_path):
            print("Create directory ", self.analysis_output_dir_path)
            os.makedirs(self.analysis_output_dir_path)

        print("{} GroupSentencesETAnalyser for subjects {}:".format(datetime.datetime.now() , " , ".join(self.subjects_ids)))
        unified_All_Measures_Group_DF_Generator = sentenceprocessing_unified_df_Generator(self.subjects_ids, self.phq_responses_fn, self.state_responses_fn, self.trait_responses_fn,
                                                                                          use_existing_asc_parser, use_existing_sentence_et, use_existing_subject_et_processed_df, use_existing_subject_et_with_lingmeasures_processed_df)

        self.per_subject_sentence_unified_all_measures_processing_df, self.per_subject_gaze_route_group_df, \
        self.per_subject_pupil_vec_per_sentence_type_df, self.all_sentences_all_subjects_df = \
            unified_All_Measures_Group_DF_Generator.create_per_subject_unified_sentenceprocessing_df()

        self.per_subject_sentence_unified_all_measures_processing_df.to_csv(all_subjects_sentence_processing_df_fn, index=False)
        self.per_subject_gaze_route_group_df.to_csv(all_subjects_gazeroute_df_fn, index=False)
        self.per_subject_pupil_vec_per_sentence_type_df.to_csv(all_subjects_pupil_vec_df_fn, index=False)
        self.all_sentences_all_subjects_df.to_csv(all_sentences_all_subjects_df_fn, index=False)

        print("{} Created new per subject sentence_processing and saved it to {}".format(datetime.datetime.now(), all_subjects_sentence_processing_df_fn))
        print("{} Created new per subject gaze_route and saved it to {}".format(datetime.datetime.now(), all_subjects_gazeroute_df_fn))
        print("{} Created new per subject pupil_vec (per set) and saved it to {}".format(datetime.datetime.now(), all_subjects_pupil_vec_df_fn))
        print("{} Created new ALL subject ALL sentences  and saved it to {}".format(datetime.datetime.now(), all_sentences_all_subjects_df_fn))
        return self.per_subject_sentence_unified_all_measures_processing_df, self.per_subject_gaze_route_group_df, self.per_subject_pupil_vec_per_sentence_type_df, self.all_sentences_all_subjects_df

    def generate_group_sentence_processing_df(self, sentence_processing_group_df_fn):
        sentence_processing_group_raw_measures_analyser = ET_ReadingMeasuresFetcher(self.per_subject_sentence_unified_all_measures_processing_df)
        self.sentence_processing_group_df_with_reading_measures = sentence_processing_group_raw_measures_analyser.add_et_reading_measures()
        self.sentence_processing_group_df_with_reading_measures.to_csv(sentence_processing_group_df_fn , index=False)
        return self.sentence_processing_group_df_with_reading_measures


    def generate_group_gazeroute_df(self, gaze_route_group_df_fn):

        columns = ["level", "list_of_subject_ids","n", "trait", "state" , "list_of_subjects_x_vec" , "list_of_subjects_y_vec", "group_mean_x_vec", "group_mean_y_vec"]
        rows = []
        for phq_level, phq_level_rows_df in self.per_subject_gaze_route_group_df.groupby("phq_level"):
            row = [phq_level, " , ".join(list(phq_level_rows_df["subject_id"])), len(phq_level_rows_df), phq_level_rows_df["trait_score"].mean(), phq_level_rows_df["state_score"].mean()]
            list_of_subjects_x_vec = ' , '.join([str(x) for x in phq_level_rows_df["x_gaze_vec"] if sum(x) > 0])
            list_of_subjects_y_vec = ' , '.join([str(x) for x in phq_level_rows_df["y_gaze_vec"] if sum(x) > 0])
            group_mean_x_vec = 0
            group_mean_y_vec = 0
            new_row = row + [list_of_subjects_x_vec] + [list_of_subjects_y_vec] + [group_mean_x_vec] + [group_mean_y_vec]
            rows.append(new_row)

        self.group_gazeroute_df = pd.DataFrame(data=rows, columns=columns)
        self.group_gazeroute_df.to_csv(gaze_route_group_df_fn, index=False)
        return self.group_gazeroute_df


    def draw_pupil_graphs(self, per_subject_pupil_vec_per_sentence_type_df, draw_slow_graphs):
        pupil_graph_generator = PupilProcessing_GraphGenerator(self.per_subject_sentence_unified_all_measures_processing_df, analysis_output_dir_path=self.analysis_output_dir_path)
        pupil_graph_generator.generate_pupil_diameter_graph(per_subject_pupil_vec_per_sentence_type_df, draw=True)
        pupil_graph_generator.generate_pupil_diameter_graph(per_subject_pupil_vec_per_sentence_type_df, draw=draw_slow_graphs, add_ci=True)
        print("=========")


    def draw_sentence_graphs(self, graph_types , draw_sentences = True, draw_ling_measures = True, draw_feats_corr = True,
                             remove_outliers = True, draw_slow_graphs = False):

        sentence_analyser = SentencesAnalyser(self.per_subject_sentence_unified_all_measures_processing_df,
                                              self.analysis_output_dir_path,
                                              remove_outliers = remove_outliers)

        if draw_sentences:
            sentence_analyser.draw_sentence_processing_graphs(graph_types)

        if draw_feats_corr:
            sentence_analyser.draw_feature_corr_graphs()
            
        if draw_ling_measures:
            sentence_analyser.draw_ling_measures_dist_graphs(draw_slow_graphs)



    def gaze_classifiers(self, group_gazeroute_df, per_subject_gaze_route_group_df, all_sentences_all_subjects_df, should_draw_evaluations = False):

        sequence_analyser = SequenceClassifier(self.per_subject_sentence_unified_all_measures_processing_df, group_gazeroute_df, per_subject_gaze_route_group_df, all_sentences_all_subjects_df, self.analysis_output_dir_path)
        sequence_analyser.classify_sentence_type_by_raw_sequence(should_draw_evaluations)
        sequence_analyser.classify_PHQ_level_by_raw_sequence(should_draw_evaluations)
        sequence_analyser.classify_sentence_type_by_measures(all_sentences_all_subjects_df, should_draw_evaluations)






