
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime


class PupilProcessing_GraphGenerator:

    def __init__(self, group_df, analysis_output_dir_path):
        self.group_df = group_df
        self.analysis_output_dir_path = analysis_output_dir_path

    def generate_pupil_diameter_graph_with_ci(self, per_subject_pupil_vec_per_sentence_type_df, hue ='phq_level', draw=True, max_ts = 6500):

        if not draw:
            return
        a = datetime.datetime.now()
        print("{} : draw generate_pupil_diameter_graph_with_hue with hue {}".format(a , hue))
        df = per_subject_pupil_vec_per_sentence_type_df
        df = df.set_index('phq_level').sentence_pupil_diameter_vec.apply(pd.Series).stack().reset_index().rename(
            columns={0: 'sentence_pupil_diameter_vec'})
        df = df.rename(columns={"level_1": "timestamp"})
        df_ts = df[df["sentence_pupil_diameter_vec"] < max_ts]
        s = sns.lineplot(x="timestamp", y="sentence_pupil_diameter_vec",
                         hue=hue,
                         # ci=None,
                         # style="phq_level",
                         data=df_ts)
        b = datetime.datetime.now()
        print("Total time is ", b - a)
        plt.show()
        f = s.get_figure()
        f.savefig(self.analysis_output_dir_path + "/pupil_lineplot_per_group.png")
        plt.close()


    def generate_pupil_diameter_graph(self, per_subject_pupil_vec_per_sentence_type_df,add_ci = False,  draw=False, min_diameter = 500, max_ts = 6500):
        if not draw:
            return

        a = datetime.datetime.now()
        print("{} : draw generate_pupil_diameter_graph_with_hue with phq {}".format(a ,  "and ci" if add_ci else ""))
        df = per_subject_pupil_vec_per_sentence_type_df
        df = df.set_index(['subject_id', 'phq_level']).sentence_pupil_diameter_vec.apply(
            pd.Series).stack().reset_index().rename(columns={0: 'sentence_pupil_diameter_vec'})
        df["subject_id"] = df["subject_id"].apply(lambda x: str(x))
        df = df.rename(columns={"level_2": "timestamp"})
        df_ts = df[(df["sentence_pupil_diameter_vec"] > min_diameter) & (df["timestamp"] < max_ts)]

        if add_ci:
            s = sns.lineplot(x="timestamp", y="sentence_pupil_diameter_vec", hue='phq_level', data=df_ts)
        else:
            s = sns.lineplot(x="timestamp", y="sentence_pupil_diameter_vec", ci=None, hue='phq_level', data=df_ts)


        s.set_title("pupil diameter per group (up to ts = {} & filter blanks)".format(max_ts))
        f = s.get_figure()
        f.savefig(self.analysis_output_dir_path + "/pupil_lineplot_per_PHQ{}.png".format( "_ci" if add_ci else ""))
        plt.close()
        plt.clf()
        b = datetime.datetime.now()
        print("Total time is ", b - a)