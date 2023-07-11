from datetime import datetime
import pandas as pd
from kats.tsfeatures.tsfeatures import TsFeatures as KATS_TsFeatures
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, TrendChangeParameters
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector
from yellowbrick.features import ParallelCoordinates
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from pyts.classification import BOSSVS, SAXVSM, LearningShapelets, KNeighborsClassifier
from pyts.transformation import ShapeletTransform, BagOfPatterns,BOSS, WEASEL, ROCKET
from sklearn.linear_model import LogisticRegression
import numpy as np

def extract_kats_feats(df, gaze_col):
    print("{} Extract 65 kats feats".format(datetime.now().time()))
    kats_model = KATS_TsFeatures()
    kats_feats_series = df.apply(lambda x : kats_model.transform(TimeSeriesData(pd.DataFrame(x[gaze_col]).
                                                                                reset_index(), time_col_name="index")) , axis=1)
    kats_feats_df = pd.DataFrame(kats_feats_series.tolist())
    return kats_feats_df

def extract_bagofpattern(df, gaze_col):
    bop_st = BagOfPatterns(window_size=48, word_size=8)
    bop_spm_x = bop_st.fit_transform(df[gaze_col])
    bop_df = bop_spm_x.toarray()
    return bop_df

def extract_BOSS(df, gaze_col, y):
    boss_st = BOSS(window_size=16, word_size=4)
    boss_spm = boss_st.fit_transform(df[gaze_col],y)
    boss_df = boss_spm.toarray()

def extract_WEASEL(df, gaze_col, y):
    wsl_st = WEASEL(sparse=False, word_size=8)
    wsl_spm = wsl_st.fit_transform(df[gaze_col],y)
    wsl_df = wsl_spm

def extract_BOSSVS(df, gaze_col, y):
    clf = BOSSVS(window_size = 28, word_size=10,n_bins=5,strategy='uniform')
    clf.fit(df,y)
    clf.score(df,y)