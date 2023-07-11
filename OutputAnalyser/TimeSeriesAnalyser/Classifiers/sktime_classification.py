
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
def tsfeats_classification(X,y):
    t = TSFreshFeatureExtractor(default_fc_parameters="minimal", show_warnings=False)

    Xt = t.fit_transform(pd.DataFrame(dts))

    t_all = TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False)
    Xt_all = t.fit_transform(pd.DataFrame(dts))
