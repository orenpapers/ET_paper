import pandas as pd
import itertools

from OutputsAnalyser.Utils import phq_analysis, state_analysis, trait_analysis
from OutputsAnalyser.SentencesAnalyser.Subject_sentences_ET_Measures_Generator import Subject_sentences_ET_Measures_Generator
import datetime

class sentenceprocessing_unified_df_Generator:

    def __init__(self, list_of_ids, phq_responses_fn, state_responses_fn, trait_responses_fn, use_existing_asc_parser = True, use_existing_sentence_et = True, use_existing_subject_et_processed_df = True, use_existing_subject_et_with_lingmeasures_processed_df = True):
        self.list_of_ids = list_of_ids
        self.phq_responses_fn = phq_responses_fn
        self.state_responses_fn = state_responses_fn
        self.trait_responses_fn = trait_responses_fn
        self.use_existing_asc_parser = use_existing_asc_parser
        self.use_existing_sentence_et = use_existing_sentence_et
        self.use_existing_subject_et_processed_df = use_existing_subject_et_processed_df
        self.use_existing_subject_et_with_lingmeasures_processed_df = use_existing_subject_et_with_lingmeasures_processed_df


    def create_per_subject_unified_sentenceprocessing_df(self):
        print("Analyse {} ids ({}) ".format(len(self.list_of_ids), self.list_of_ids))

        sep = "_"
        linguistic_measures_keys = ["first_fixation_duration","second_fixation_duration","total_gaze_duration","num_of_fixations","word_has_first_pass_regression","regression_path_duration","skipping_probability","trial_total_distance_covered", "word_pupil_diameter_mean","sentence_pupil_diameter_mean"]
        word_probe_cols = ["wordrprobe_positive_sentiment_RT", "wordrprobe_positive_sentiment_success_ratio", "wordrprobe_negative_sentiment_RT", "wordrprobe_negative_sentiment_success_ratio"]
        word_types = ["source_word","target_word"]
        sentence_types = ["A","B","C","D"]

        sentencesprocessing_columns = ["subject_id" , "phq_score" , "phq_level" , "state_score" , "trait_score"] + word_probe_cols + [sep.join(x) for x in list(itertools.product(sentence_types, word_types, linguistic_measures_keys))]
        gazeroutes_columns = ["subject_id" , "phq_score" , "phq_level" , "state_score" , "trait_score", "x_gaze_vec","y_gaze_vec"]
        pupil_vec_columns = ["subject_id" , "phq_score" , "phq_level" , "state_score" , "trait_score", "set_num", "sentence_type", "sentence_pupil_diameter_vec" ]

        sentencesprocessing_rows = []
        gazeroute_rows = []
        pupil_rows = []
        all_sentences_all_subjects_df = pd.DataFrame(columns= ['Subject', 'sentence_run_num', 'sentence_trial_num', 'Sentence', 'Sentence_type', 'set_num', 'fixation_words_order', 'word_type', 'word_value', 'word_idx', 'first_fixation_duration', 'second_fixation_duration', 'total_gaze_duration', 'num_of_fixations', 'word_has_first_pass_regression', 'regression_path_duration', 'is_skipping_trial', 'trial_total_distance_covered', 'word_pupil_diameter_mean', 'sentence_pupil_diameter_mean', 'sentence_pupil_diameter_vec', 'trial_actual_tuples_of_orig_gaze_locations'])

        for subject_id in self.list_of_ids:

            print("{} create_unified_group_sentenceprocessing_df - Subject num is {}".format(datetime.datetime.now(), subject_id))
            subject_et_analyser = Subject_sentences_ET_Measures_Generator(subject_id, self.use_existing_asc_parser, self.use_existing_sentence_et, self.use_existing_subject_et_processed_df, self.use_existing_subject_et_with_lingmeasures_processed_df)
            subject_et_with_lingmeasures_processed_df, subject_gaze_route_df = subject_et_analyser.generate_subject_measures_df()

            measures = subject_et_analyser.get_subject_per_sentype_scores(subject_et_with_lingmeasures_processed_df, linguistic_measures_keys, word_types, sentence_types)
            mental_scores = phq_analysis(self.phq_responses_fn, subject_id) + \
                            state_analysis(self.state_responses_fn, subject_id) + \
                            trait_analysis(self.trait_responses_fn, subject_id)

            word_probe_row = subject_et_analyser.generate_wordprobe_subject_row()
            subject_overall_unified_row = subject_et_analyser.generate_subject_row(measures,word_probe_row, mental_scores)
            gazeroute_vecs = subject_et_analyser.generate_subject_gazeroute_vecs(subject_gaze_route_df, mental_scores)

            pupil_subj_rows = subject_et_analyser.generate_subject_pupil_vecs(subject_et_with_lingmeasures_processed_df, sentence_types, subject_id, mental_scores  )

            if is_valid_gazeroute(gazeroute_vecs):
                sentencesprocessing_rows.append(subject_overall_unified_row)
                gazeroute_rows.append([subject_id] + mental_scores + gazeroute_vecs)
            else:
                print(" *** Please notice - subject {} has invalid gazeroute ***".format(subject_id))

            pupil_rows += pupil_subj_rows
            all_sentences_all_subjects_df = pd.concat([all_sentences_all_subjects_df, subject_et_with_lingmeasures_processed_df])

        sentencesprocessing_unified_group_df = pd.DataFrame(data=sentencesprocessing_rows, columns=sentencesprocessing_columns)
        gaze_route_group_df = pd.DataFrame(data=gazeroute_rows, columns = gazeroutes_columns)
        pupil_vec_df = pd.DataFrame(data = pupil_rows , columns = pupil_vec_columns )

        return sentencesprocessing_unified_group_df, gaze_route_group_df, pupil_vec_df, all_sentences_all_subjects_df


def is_valid_gazeroute(gazeroute_vecs):
    x_vec = gazeroute_vecs[0]
    y_vec = gazeroute_vecs[1]
    num_x_zeros = len([x for x in x_vec if x == 0])
    num_y_zeros = len([x for x in y_vec if x == 0])

    if num_x_zeros < 0.5 * len(x_vec) and num_y_zeros < 0.5 * len(y_vec):
        return True
    return False

