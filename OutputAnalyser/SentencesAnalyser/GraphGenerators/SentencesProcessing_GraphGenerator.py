import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class SentencesProcessing_GraphGenerator:

    def __init__(self, group_df, measures_df,  analysis_output_dir_path, dir_suffix, phq_dict1):
        self.group_df = group_df
        self.analysis_output_dir_path = analysis_output_dir_path + dir_suffix + "/"
        os.mkdir(self.analysis_output_dir_path)
        print("analysis_output_dir_path is ", analysis_output_dir_path)
        self.hue_order = ['Moderate', 'Minimal', 'Moderately Severe', 'Mild', 'Severe']
        if phq_dict1:
            measures_df["phq_level"] = measures_df["phq_level"].apply(lambda x: phq_dict1[x])

        self.measures_df = measures_df
    def create_df_with_word_type_col(self, df , cols_to_melt, value_col_name, id_cols = ["subject_id"]):

        cols = id_cols + cols_to_melt
        df = df[cols]
        df1 = pd.melt(df, id_vars=id_cols, var_name=value_col_name)
        return df1


    def generate_regplot(self, remove_outlier = True , robust = False):

        df = self.measures_df
        if remove_outlier:
            outliers_cond = df[((df["gaze_measure"] == "first_fixation_duration") & (df["value"] > 150)) |
                               ((df["gaze_measure"] == "second_fixation_duration") & (df["value"] > 150)) |
                               ((df["gaze_measure"] == "total_gaze_duration") & (df["value"] > 1000)) |
                               ((df["gaze_measure"] == "trial_total_distance_covered") & (df["value"] > 10000)) |
                               ((df["gaze_measure"] == "regression_path_duration") & (df["value"] > 130)) |
                               ((df["gaze_measure"] == "sentence_pupil_diameter_mean") & (df["value"] > 7000))|
                               ((df["gaze_measure"] == "word_pupil_diameter_mean") & (df["value"] > 7000))]
            df = df.drop(outliers_cond.index, axis = 0)





        g = sns.lmplot(x="value", y="phq_score", hue="sentence_cond", col="gaze_measure", col_wrap=4, robust = robust,
                       data=df, height=3, sharex=False)
        fn = self.analysis_output_dir_path + "/AA_all_reg.png"
        g.savefig(fn)
        plt.close()
        print("Draw reg plot to ", fn)
        a = 2

    def generate_sentencecond_phq_gazemeasures(self, graph_types ):

        df = self.measures_df[self.measures_df.gaze_measure.isin(["first_fixation_duration","second_fixation_duration"])]

        for kind in graph_types:
            for i, hue in enumerate([None, "word_type"]):

                g = sns.catplot(data = df, x="sentence_cond", y="value", col = "phq_level", row = "gaze_measure", hue= hue,
                            kind = kind, sharex=False, sharey= False, legend=True, legend_out=False,
                                # col_order = col_order
                                )

                # for ax in g.axes.flat:
                    # labels = ax.get_xticklabels()  # get x labels
                    # for i, l in enumerate(labels):
                    #     if (i % 2 == 0): labels[i] = ''  # skip even labels
                    # ax.set_xticklabels(col_order, rotation=30)  # set new labels

                fn = self.analysis_output_dir_path + "/AAmeasures_cat_plot_{}_{}.png".format( kind, i)
                plt.legend(loc='upper right')

                plt.savefig(fn)
                plt.close()
                print("Saved to {}".format(fn))

    def generate_phq_sentencecond_gazemeasures(self, graph_types):

        df = self.measures_df[
            self.measures_df.gaze_measure.isin(["first_fixation_duration", "second_fixation_duration"])]
        df = df.rename({"sentence_cond" : "condition", "gaze_measure" : "measure"}, axis='columns')
        for kind in graph_types:
            for i, hue in enumerate([None, "word_type"]):
                col_order = ["Level1","Level2","Level3"] if "Level1" in df["phq_level"] else ["Level1","Level2","Level3","Level4","Level5"]
                g = sns.catplot( data = df, x="phq_level", y="value", col = "condition", row = "measure", hue = hue,
                            kind = kind , sharex=True, sharey= True, legend=True, legend_out=False,
                                 row_order=["first_fixation_duration", "second_fixation_duration"])
                fn = self.analysis_output_dir_path + "/BBcond_cat_plot_{}.png".format( kind)
                # plt.show()
                # g.legend()
                g.fig.subplots_adjust(wspace=0.1, hspace=0.08)
                for ax in g.axes.flat:
                    ax.set_xticklabels(col_order, rotation=30)  # set new labels
                plt.legend(loc='upper right')
                plt.savefig(fn)
                plt.close()
                print("Saved to {}".format(fn))

    def generate_sentype2gaze_cat_graphs(self, graph_types, print_n = False ):

        word_type = "target_word"
        conds = ["A", "B", "C", "D"]
        for graph_type in graph_types:
            for measure_type in ["trial_total_distance_covered", "total_gaze_duration", "first_fixation_duration", "second_fixation_duration",
                                 "regression_path_duration", "skipping_probability", "word_has_first_pass_regression",
                                 "num_of_fixations", "word_pupil_diameter_mean", "sentence_pupil_diameter_mean"]:

                if graph_type == "reg":
                    mdf = self.measures_df[self.measures_df.gaze_measure == measure_type]
                    g = sns.lmplot(x="value", y="phq_score", hue="sentence_cond",
                                   data= mdf , height=3)

                    fn = self.analysis_output_dir_path + "/CCsentences_{}_{}plot.png".format(measure_type, graph_type)

                    g.savefig(fn)
                    # plt.show()
                    plt.close()

                else:
                    cols_to_melt = [x + "_" + word_type + "_" + measure_type for x in conds]
                    df =  self.create_df_with_word_type_col(self.group_df, id_cols=["phq_level"],
                                                            cols_to_melt = cols_to_melt,
                                                           value_col_name = "sentence_cond").dropna(subset=["phq_level"])

                    df['cond'] = df['sentence_cond'].str.split('_',expand=True)[0]
                    df['measure'] = measure_type

                    hue_col = "phq_level"
                    order = ["A","B","C","D"]
                    x_col = "cond"
                    y_col = "value"
                    width = 1.5

                    g = sns.catplot(x=x_col, y=y_col, hue=hue_col, kind=graph_type, data=df , legend=False,
                                    height=8, aspect=1, hue_order=self.hue_order)

                    # get the offsets used by boxplot when hue-nesting is used
                    # https://github.com/mwaskom/seaborn/blob/c73055b2a9d9830c6fbbace07127c370389d04dd/seaborn/categorical.py#L367
                    ax = g.axes[0, 0]
                    n_levels = len(df[hue_col].unique())
                    each_width = width / n_levels
                    offsets = np.linspace(0, width - each_width, n_levels)
                    offsets -= offsets.mean()

                    pos = [x + o for x in np.arange(len(order)) for o in offsets]

                    counts = df.groupby([x_col, hue_col])[y_col].size()
                    counts = counts.reindex(pd.MultiIndex.from_product([order, self.hue_order]))
                    medians = df.groupby([x_col, hue_col])[y_col].median()
                    medians = medians.reindex(pd.MultiIndex.from_product([order, self.hue_order]))

                    for p, n, m in zip(pos, counts, medians):
                        if not np.isnan(m):
                            if print_n:
                                ax.annotate('N={:.0f}'.format(n), xy=(p, m), xycoords='data', ha='center', va='bottom')



                    fn = self.analysis_output_dir_path + "/CCsentences_{}_{}plot.png".format(measure_type, graph_type)
                    plt.ylabel(measure_type)
                    plt.xlabel("Sentence condition")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title = "PHQ level")

                    g.savefig(fn)
                    # plt.show()
                    plt.close()


    def generate_sentence_cond_main_effect(self, cond_to_check, graph_types ):

        for kind in graph_types:
            for sharey in [True, False]:
                suffix = str(sharey)
                df = self.measures_df
               # df = df[(df.gaze_measure == 'word_pupil_diameter_mean') | (df.gaze_measure == 'trial_total_distance_covered')]
                sns.catplot(x=cond_to_check, y="value", data=df,  palette="vlag", kind=kind,
                             sharey= False , col="gaze_measure" )

                fn = self.analysis_output_dir_path + "/DD{}_sentence_cond_main_effect_{}_sharey{}.png".format(kind , cond_to_check, suffix)

                plt.savefig(fn)
                print("Saved to {}".format(fn))
                plt.clf()
                plt.close()






        a = 2

    def generate_phq2gaze_cat_graphs_with_sourceword_subplot(self, graph_types, word_type ):
        conds = ["A", "B", "C", "D"]
        word_types = ["target_word", "source_word"] #todo fix with loop to put on the same grapth
        for graph_type in graph_types:
            for measure_type in ["total_gaze_duration", "first_fixation_duration", "second_fixation_duration",
                                 "regression_path_duration", "skipping_probability", "word_has_first_pass_regression",
                                 "num_of_fixations", "word_pupil_diameter_mean", "sentence_pupil_diameter_mean"]:
                # word_type = "source_word"
                # f, axes = plt.subplots(2, 1, figsize=(15,15))
                #
                # for subplot_idx, word_type in enumerate(word_types):

                print("Working on measure {} with {}".format(measure_type, graph_type))
                cols_to_melt = [x + "_" + word_type + "_" + measure_type for x in conds]
                df = self.create_df_with_word_type_col(self.group_df, id_cols=["phq_level"],
                                                       cols_to_melt=cols_to_melt,
                                                       value_col_name="sentence_cond")

                df.sentence_cond = df.sentence_cond.str.extract(r'(\w)_')
                # df = df.rename({"value" : measure_type}, axis=1)
                g = sns.catplot(x="phq_level", y="value", hue="sentence_cond", kind=graph_type, data=df,
                                hue_order=self.hue_order,
                                # ax=axes[subplot_idx] ,
                            legend=False)
                # try:
                #     axes[subplot_idx].get_legend().remove()
                # except:
                #     pass
                plt.subplots_adjust()
                # axes[subplot_idx].set_title("{} of {}".format(measure_type, word_type))

                # handles, labels = axes[0].get_legend_handles_labels()
                # f.legend(handles, labels, loc='upper center')

                # if subplot_idx == 0:
                #     axes[0].legend(bbox_to_anchor=(1.02, 1), loc=0, borderaxespad=0., title = "Sentence type")


                fn = self.analysis_output_dir_path + "/EE{}_{}plot_{}.png".format(measure_type, graph_type, word_type)
                print("Saved phq2{} to {}".format(measure_type, fn))
                g.savefig(fn)
                plt.title("{} - {}".format(measure_type, word_type))
                plt.close()

    def generate_phq2gaze_cat_graphs(self, graph_types =["box","boxen","point","strip"]):

        conds = ["A","B","C","D"]
        word_type = "target_word"
        for graph_type in graph_types:
            for measure_type in ["total_gaze_duration", "first_fixation_duration", "second_fixation_duration",
                                 "regression_path_duration", "skipping_probability","word_has_first_pass_regression",
                                 "num_of_fixations","word_pupil_diameter_mean","sentence_pupil_diameter_mean"]:

                print("Working on measure {} with {}".format(measure_type, graph_type))
                cols_to_melt = [x + "_" + word_type + "_" + measure_type for x in conds]
                df = self.create_df_with_word_type_col(self.group_df, id_cols=["phq_level"],
                                                       cols_to_melt = cols_to_melt ,
                                                       value_col_name = "sentence_cond")

                df.sentence_cond = df.sentence_cond.str.extract(r'(\w)_')
                s = sns.catplot(x="phq_level", y="value", hue="sentence_cond", kind=graph_type, data=df,legend=False,
                                height=8, aspect=1)

                cnts = dict(self.group_df['phq_level'].value_counts())
                key = list(cnts.keys())
                vals = list(cnts.values())
                s.set_axis_labels("", "pulse")
                s.set_xticklabels([(key[i] + '\n(n = ' + str(vals[i]) + ')') for i in range(len(key))])

                plt.ylabel(measure_type)
                plt.xlabel("PHQ")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , title="Sentence type")

                fn = self.analysis_output_dir_path + "/FF{}_{}plot.png".format(measure_type, graph_type)
                s.savefig(fn)
                print("Saved phq2{} to {}".format(measure_type, fn))
                plt.close()
