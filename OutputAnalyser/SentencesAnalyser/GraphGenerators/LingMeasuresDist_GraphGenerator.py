import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

class LingMeasuresDist_GraphGenerator:

    def __init__(self, group_df, measures_df, analysis_output_dir_path):
        self.group_df = group_df
        self.analysis_output_dir_path = analysis_output_dir_path
        self.measures_df = measures_df

    def draw_ling_measures_graphs_jointplot(self):

        feat_duos = [('wordrprobe_positive_sentiment_RT', 'wordrprobe_negative_sentiment_RT')]
        for feat1, feat2 in feat_duos:
            g = sns.jointplot(feat1, feat2, data=self.group_df, kind="hex")
            # plt.show()

            fn = self.analysis_output_dir_path + "/{}_{}_jointplotplot.png".format(feat1, feat2)
            g.savefig(fn)

    def draw_ling_measures_graphs_pairplot(self, draw = False): #notice - this is VERY slow

        if draw:
            b = datetime.datetime.now()
            print("{} Draw pairplot".format(b) )
            s = sns.pairplot(self.group_df, hue="phq_level")#, markers=["o", "s", "D"])
            e = datetime.datetime.now()
            print("{} Done drawing pairplot ({} secs)".format(e , e-b))
            plt.show()

            fn = self.analysis_output_dir_path + "/feats_pairplot.png"
            s.savefig(fn)
        else:
            return




