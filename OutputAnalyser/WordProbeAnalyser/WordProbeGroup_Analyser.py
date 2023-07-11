import seaborn as sns
import matplotlib.pyplot as plt
from OutputsAnalyser import Utils
import pandas as pd

class WordProbeGroup_Analyser:

    def __init__(self, analysis_output_dir_path, dot_probe_group_behav_df, debug_mode=False):
        self.analysis_output_dir_path = analysis_output_dir_path
        self.dot_probe_group_behav_df = dot_probe_group_behav_df
        self.debug_mode = debug_mode
        print("Saving wordprobe-group BEHAV analysis in ", self.analysis_output_dir_path)

    def draw_per_subject_facet_barplot(self, graph_type):
        df = self.dot_probe_group_behav_df

        if graph_type =="RT":
            rt_df = df.set_index(['subject_id', 'word_type','phq_level'])['RT_vec'].apply(pd.Series).stack().reset_index()
            rt_df.columns = ['subject_id', 'word_type','phq_level', 'trial_num','RT']


            g = sns.FacetGrid(rt_df, col="subject_id",hue="phq_level", height=4, aspect=.5,  palette = 'Set1')
            g.map(sns.barplot, "word_type", graph_type, edgecolor="w")

            fn = "{}/WordProbe_facet_{}_boxplot.png".format(self.analysis_output_dir_path, graph_type)
            print("Saved draw_per_subject_GD_facet_barplot to {}".format(graph_type, fn))
            g.savefig(fn)
            plt.close()
            plt.clf()
            plt.cla()

        if graph_type == "GD":
            gd_df = df.set_index(['subject_id', 'word_type','phq_level'])['GD_vec'].apply(pd.Series).stack().reset_index()
            gd_df.columns = ['subject_id', 'word_type','phq_level', 'trial_num', 'GD']

            g = sns.FacetGrid(gd_df, col="subject_id", hue="phq_level",height=4, aspect=.5, palette='Set1')
            g.map(sns.barplot, "word_type", graph_type, edgecolor="w")

            fn = "{}/WordProbe_facet_{}_boxplot.png".format(self.analysis_output_dir_path, graph_type)
            print("Saved draw_per_subject_{}_facet_barplot to {}".format(graph_type, fn))
            g.savefig(fn)
            plt.close()
            plt.clf()
            plt.cla()


    def draw_grouplevel_effect(self):
        for y in ["mean_RT","mean_GD"]:
            pl = sns.boxplot(x="word_type", y=y, hue="phq_level", data=self.dot_probe_group_behav_df)
            fn = "{}/WordProbe_group_{}_box_plot.png".format(self.analysis_output_dir_path, y)
            pl.figure.savefig(fn)
            plt.close()
            plt.clf()
            plt.cla()
            pl = sns.boxenplot(x="word_type", y=y, hue="phq_level", data=self.dot_probe_group_behav_df)
            fn = "{}/WordProbe_group_{}_boxen_plot.png".format(self.analysis_output_dir_path, y)
            pl.figure.savefig(fn)
            plt.close()
            plt.clf()
            plt.cla()
            pl = sns.violinplot(x="word_type", y=y, hue="phq_level", data=self.dot_probe_group_behav_df)
            fn = "{}/WordProbe_group_{}_violin_plot.png".format(self.analysis_output_dir_path, y)
            pl.figure.savefig(fn)
            plt.close()
            plt.clf()
            plt.cla()
            print("Saved draw_grouplevel_RT_effect to {}".format(fn))
            plt.close()
            plt.clf()
            plt.cla()

