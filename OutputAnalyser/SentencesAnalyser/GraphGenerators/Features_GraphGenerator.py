import seaborn as sns
import pandas as pd
import numpy as np
from yellowbrick.features import Rank2D, Rank1D, RadViz, ParallelCoordinates
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates


class Features_GraphGenerator:

    def __init__(self, group_df, measures_df, list_of_feats , analysis_output_dir_path):
        self.group_df = group_df
        self.analysis_output_dir_path = analysis_output_dir_path
        self.measures_df = measures_df
        self.list_of_feats = list_of_feats


    def draw_feats_pearson_corr(self):
        visualizer = Rank2D(algorithm="pearson", size = (2000, 2000), title = "Features correlation")
        X = self.group_df[self.list_of_feats]
        y = self.group_df["phq_level"]
        visualizer.fit(X, y)  # Fit the data to the visualizer
        visualizer.transform(X)
        custom_viz = visualizer.ax
        custom_viz.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45)
        custom_viz.tick_params(labelsize=4)
        custom_viz.figure.savefig( self.analysis_output_dir_path + "/features_pearson_corr.png")
        plt.close()
        plt.clf()

    def draw_features_rank(self):
        visualizer = Rank1D(algorithm="shapiro", size = (2000, 2000), title = "Features Rank")
        X = self.group_df[self.list_of_feats]
        X.columns = sorted(X.columns)
        y = self.group_df["phq_level"]
        visualizer.fit(X, y)  # Fit the data to the visualizer
        visualizer.transform(X)
        custom_viz = visualizer.ax
        custom_viz.set_xticklabels(visualizer.ax.get_yticklabels(), rotation=45)
        custom_viz.tick_params(labelsize=9)
        custom_viz.figure.savefig(self.analysis_output_dir_path + "/features_rank.png")
        plt.close()
        plt.clf()

    def draw_features_radviz_rank(self):
        lev2int_dict = {"Minimal" : 0,  "Mild" : 1, "Moderate" : 2 , "Moderately Severe" : 3, "Severe" : 4}
        df = self.group_df.dropna(subset=["phq_level"])
        X = df[self.list_of_feats]
        measures = ["total_gaze_duration", "first_fixation_duration", "second_fixation_duration",
                                 "regression_path_duration", "skipping_probability", "word_has_first_pass_regression",
                                 "num_of_fixations", "word_pupil_diameter_mean", "sentence_pupil_diameter_mean"]
        y_orig = df["phq_level"]
        y_int = pd.Series([lev2int_dict[x] for x in y_orig])
        for feat_to_test_str in ["A_target", "B_target","C_target","D_target"] + measures:
            print("Creating RadViz to ", feat_to_test_str)
            feats = [x for x in X.columns if feat_to_test_str in x]

            new_X = X[feats].reset_index(drop=True)
            nan_indexes = list(np.where(new_X[feats].isnull())[0])
            new_X = new_X.drop(nan_indexes).reset_index(drop=True)
            y = y_orig.drop(nan_indexes).reset_index(drop=True)
            y_int = pd.Series([lev2int_dict[x] for x in y])
            new_X.columns = [x.replace(feat_to_test_str+"_", "") for x in new_X.columns]

            visualizer = RadViz(classes=list(lev2int_dict.keys()))
            visualizer.fit(new_X, y_int)
            visualizer.transform(X)  # Transform the data
            custom_viz = visualizer.ax
            custom_viz.set_title("New title")
            custom_viz.figure.legend(
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                title="level",
                loc=0,
            )
            fn = "{}/{}features_radviz_rank.png".format(self.analysis_output_dir_path, feat_to_test_str)
            custom_viz.figure.savefig(fn)
            print("Saved to ", fn)
            plt.close()
            plt.clf()


    def draw_features_parallel_coords(self, tool = 'pandas'):
        # for feat_type in ["A_target", "A_source", "B_target", "B_source", "C_target", "C_source", "D_target", "D_source"]:
        feats = ["A_target_word_first_fixation_duration", "A_source_word_first_fixation_duration","A_target_word_second_fixation_duration"]
        df = self.group_df.dropna(subset = feats + ["phq_level"])
        if tool == 'pandas':
            plt.figure(figsize=(10, 10))
            f = parallel_coordinates(df, class_column= "phq_level", cols=feats, colormap=plt.get_cmap("Set2"))

            f.set_xticklabels(f.get_xticklabels(), rotation=45)
            f.figure.savefig(self.analysis_output_dir_path + "/pandas_parallel_coords.png")

        if tool == 'yb':

            X = df[feats]
            y = df["phq_level"]
            visualizer = ParallelCoordinates(classes=y.unique(), features=feats, sample=0.2, shuffle=True)
            visualizer.fit_transform(X, y)
            custom_viz = visualizer.ax
            custom_viz.set_xticklabels(visualizer.ax.get_xticklabels(), rotation=45)
            custom_viz.tick_params(labelsize=10)
            custom_viz.figure.savefig(self.analysis_output_dir_path + "/parallel_coords.png")
        plt.close()
        plt.clf()
