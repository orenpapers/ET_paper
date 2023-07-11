import pandas as pd

from OutputsAnalyser.SentencesAnalyser.GraphGenerators.SentencesProcessing_GraphGenerator import SentencesProcessing_GraphGenerator
from OutputsAnalyser.SentencesAnalyser.GraphGenerators.LingMeasuresDist_GraphGenerator import LingMeasuresDist_GraphGenerator
from OutputsAnalyser.SentencesAnalyser.GraphGenerators.Features_GraphGenerator import Features_GraphGenerator



measures = ["total_gaze_duration", "first_fixation_duration", "second_fixation_duration",
            "regression_path_duration", "skipping_probability", "word_has_first_pass_regression",
            "num_of_fixations", "word_pupil_diameter_mean", "sentence_pupil_diameter_mean"]
word_types = ["target_word", "source_word"]
conds = ["A", "B", "C", "D"]


def get_list_of_sentence_feats():
    feats = []
    for w in word_types:
        for c in conds:
            for m in measures:
                f = c + "_" + w + "_" + m
                feats.append(f)

    return feats

def create_measures_df(group_df):

    new_df = pd.melt(group_df, id_vars=['phq_level','phq_score', 'subject_id'], var_name='abc').sort_values(by=['subject_id', 'phq_level'])
    new_df['sentence_cond'] = new_df['abc'].apply(lambda x: (x.split('_'))[0])
    new_df['gaze_measure'] = new_df['abc'].apply(lambda x: "_".join(x.split('_')[3:]))
    df = new_df[(new_df["sentence_cond"].str.startswith("A")) | (new_df["sentence_cond"].str.startswith("B")) | (new_df["sentence_cond"].str.startswith("C")) | (new_df["sentence_cond"].str.startswith("D"))]
    return df

class SentencesAnalyser:

    def __init__(self, group_df, analysis_output_dir_path, remove_outliers = True):
        self.group_df = group_df
        list_of_feats = get_list_of_sentence_feats()
        self.analysis_output_dir_path = analysis_output_dir_path
        self.measures_df = create_measures_df(group_df)
        self.ling_measures_dist_graph_generator = LingMeasuresDist_GraphGenerator(self.group_df, self.measures_df, analysis_output_dir_path=self.analysis_output_dir_path)
        self.features_graph_generator = Features_GraphGenerator(self.group_df, self.measures_df, list_of_feats,  analysis_output_dir_path=self.analysis_output_dir_path)

        sentype_dict = {"A": "pos_pos", "B": "pos_neg", "C": "neg_neg", "D": "neg_pos"}
        self.measures_df = self.measures_df.dropna(subset=["phq_level"]).reset_index()
        self.measures_df["sentence_cond"] = self.measures_df["sentence_cond"].apply(lambda x: sentype_dict[x])
        self.measures_df["word_type"] = self.measures_df["abc"].apply(lambda x : x.split("_")[1] + "_word")
        if remove_outliers:
            outliers_cond = self.measures_df[((self.measures_df["gaze_measure"] == "first_fixation_duration") & (self.measures_df["value"] > 150)) |
                               ((self.measures_df["gaze_measure"] == "second_fixation_duration") & (self.measures_df["value"] > 150)) |
                               ((self.measures_df["gaze_measure"] == "total_gaze_duration") & (self.measures_df["value"] > 1000)) |
                               ((self.measures_df["gaze_measure"] == "trial_total_distance_covered") & (self.measures_df["value"] > 10000)) |
                               ((self.measures_df["gaze_measure"] == "regression_path_duration") & (self.measures_df["value"] > 130)) |
                               ((self.measures_df["gaze_measure"] == "sentence_pupil_diameter_mean") & (self.measures_df["value"] > 7000))|
                               ((self.measures_df["gaze_measure"] == "word_pupil_diameter_mean") & (self.measures_df["value"] > 7000))]
            self.measures_df = self.measures_df.drop(outliers_cond.index, axis = 0)

        print("PHQ distribution is: \n" , self.group_df.phq_level.value_counts())

    def draw_sentence_processing_graphs(self, graph_types ):
        phq_dict1 = {"Mild" : "Level1", "Minimal" : "Level1", "Moderate": "Level2", "Moderately Severe" : "Level3", "Severe" : "Level3" }
        phq_dict2 = {"Mild": "Level1", "Minimal": "Level2", "Moderate": "Level3", "Moderately Severe": "Level4",
                     "Severe": "Level5"}
        suffixes = ["phq_3_level", "phq_5_level"]

        for i, phq_dict in enumerate([ phq_dict2]):
            sentences_processing_graph_generator = SentencesProcessing_GraphGenerator(self.group_df,
                                                                                           self.measures_df,
                                                                                           dir_suffix = suffixes[i],
                                                                                            phq_dict1 = phq_dict,
                                                                                           analysis_output_dir_path=self.analysis_output_dir_path)

            sentences_processing_graph_generator.generate_regplot(remove_outlier = True , robust = False)
            sentences_processing_graph_generator.generate_sentencecond_phq_gazemeasures(graph_types = graph_types)
            sentences_processing_graph_generator.generate_sentype2gaze_cat_graphs(graph_types = graph_types + ["reg"])
            sentences_processing_graph_generator.generate_phq_sentencecond_gazemeasures(graph_types = graph_types)
            sentences_processing_graph_generator.generate_phq2gaze_cat_graphs_with_sourceword_subplot(graph_types = graph_types, word_type="target_word")
            sentences_processing_graph_generator.generate_phq2gaze_cat_graphs_with_sourceword_subplot(graph_types=graph_types, word_type="source_word")
            sentences_processing_graph_generator.generate_sentence_cond_main_effect(cond_to_check="sentence_cond", graph_types = graph_types)
            sentences_processing_graph_generator.generate_sentence_cond_main_effect(cond_to_check="phq_level", graph_types = graph_types)
            sentences_processing_graph_generator.generate_phq2gaze_cat_graphs(graph_types = graph_types)


    def draw_feature_corr_graphs(self):
        self.features_graph_generator.draw_feats_pearson_corr()
        self.features_graph_generator.draw_features_rank()
        self.features_graph_generator.draw_features_parallel_coords()
        try:
            self.features_graph_generator.draw_features_radviz_rank()
        except:
            print("Cant create grapth draw_feature_corr_graphs")

    def draw_ling_measures_dist_graphs(self, draw_slow_graphs):
        self.ling_measures_dist_graph_generator.draw_ling_measures_graphs_jointplot()
        self.ling_measures_dist_graph_generator.draw_ling_measures_graphs_pairplot(draw = draw_slow_graphs)







