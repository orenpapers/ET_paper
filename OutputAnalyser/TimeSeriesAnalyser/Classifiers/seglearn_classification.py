import pandas as pd
from seglearn.transform import FeatureRep, Segment, PadTrunc
from seglearn.pipe import Pype
from seglearn.transform import Segment
import numpy as np
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from seglearn.feature_functions import all_features
import pandas as pd
from sklearn.metrics import accuracy_score

def multimodel_classifier(df, timeseries_col = 'x_gaze_location'):
    feats = all_features().copy()
    feats.pop('hmean')
    pipe = Pype([('ftr', FeatureRep(features=feats))])
    timeseries_feats = pipe.fit_transform(df[timeseries_col])
    nlp_feats = ""

def seglearn_feats(X_ts, y_ts):
    feats = all_features().copy()
    feats.pop('hmean')
    pipe = Pype([
        # ('trunc', PadTrunc(width=width)),
        # ('seg', Segment(width=width // seg_div_factor, overlap=0.0)), #For pad 5000, width 500 there are 87120 segments
        ('ftr', FeatureRep(features=feats)),
        # ('scaler', StandardScaler()),
        # ('rf',RandomForestClassifier(random_state=42))
    ])
    X , y = pipe.fit_transform(X_ts, np.array(y_ts))
    Xdf = pd.DataFrame(X, columns=["seglearn_featrep#{}".format(i) for i in range(X.shape[1])])
    return Xdf,y

def featurerep_classifier(df_subject_res,X_ts,y_ts, train_idx, test_idx, binary_threshold=0.8, seg_div_factor = 20, width = 5000):

    feats = all_features().copy()
    feats.pop('hmean')
    pipe = Pype([
        ('trunc', PadTrunc(width=width)),
        ('seg', Segment(width=width // seg_div_factor, overlap=0.0)), #For pad 5000, width 500 there are 87120 segments
        ('ftr', FeatureRep(features=feats)),
        # ('scaler', StandardScaler()),
        # ('rf',RandomForestClassifier(random_state=42))
    ])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)
    est = RandomForestClassifier(random_state=42)

    X_train, X_test, y_train, y_test = X_ts.ts_data[train_idx], X_ts.ts_data[test_idx], \
                                       np.array(y_ts)[train_idx], np.array(y_ts)[test_idx]
    X_train_seg, y_train_seg = pipe.fit_transform(X_train, y_train)
    X_test_seg, y_test_seg = pipe.transform(X_test, y_test)
    X_train_seg = X_train_seg[:, ~np.isnan(X_train_seg).any(axis=0)]
    X_test_seg = X_test_seg[:, ~np.isnan(X_test_seg).any(axis=0)]

    est.fit(X_train_seg, y_train_seg)
    score = est.score(X_test_seg, y_test_seg)
    print("featurerep_classifier score is " , score)

    pred1 = np.array([x[0] for x in est.predict_proba(X_test_seg)])
    pred = pred1.reshape(len(pred1)//seg_div_factor, seg_div_factor).mean(axis=1)

    df_subject_res = df_subject_res.iloc[test_idx]
    df_subject_res["trial_pred"] = pred
    res_dict = {}
    pred2label_vec = []
    for subject, subject_res_df in df_subject_res.groupby(['Subject']):
        subj_pred = subject_res_df["trial_pred"].max()
        try:
            subj_label = subject_res_df["phq_label"].mean()
        except Exception as e:
            a = 2
        res_dict[subject] = {}
        res_dict[subject]["pred"] = subj_pred
        res_dict[subject]["label"] = subj_label
        res_dict[subject]["is_correct"] = (subj_pred == subj_label)
        pred2label_vec.append((subj_label, subj_pred))
    per_subj_score = accuracy_score([x[0] for x in pred2label_vec], [1 if x > binary_threshold else 0 for x in [y[1] for y in pred2label_vec]])
    print("featurerep_classifier subj score is {} ({} - {}) ".format(per_subj_score, [round(x[1],2) for x in pred2label_vec], [x[0] for x in pred2label_vec]))
    # precision, recall, thresholds = precision_recall_curve(test_subjects_labels, per_subj_pred_vec)
    #disp = plot_precision_recall_curve(est, X_test, y_test)
    #plt.show()
    return est, pipe, score, per_subj_score, pred, X_test, y_test

def segment_condrnn_classifier(X,y):
    #todo implement: https://github.com/philipperemy/cond_rnn
    pass

def segment_crnn_classifier(X,y, train_idx, test_idx):

    timesteps = 10
    n_features = 500

    def convrnn_model(ts = timesteps, nf = n_features , n_classes=2, conv_kernel_size=5,
                   conv_filters=3, lstm_units=3):
        input_shape = (ts, nf)
        model = Sequential()
        model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                         padding='valid', activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                         padding='valid', activation='relu'))

        model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(n_classes, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        return model

    # create a segment learning pipeline
    pipe = Pype([
        ('trunc', PadTrunc(width=5000)),
        ('seg', Segment(width=n_features, overlap=0.0, order='C')),
        ('scaler', StandardScaler())
    ])

    nn = KerasClassifier(build_fn=convrnn_model, epochs=1, batch_size=256, verbose=0)
    X = X.ts_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, y_train = pipe.fit_transform(X_train, y_train)
    X_test, y_test = pipe.fit_transform(X_test, y_test)
    X_train_3d = X_train.reshape((X_train.shape[0] // timesteps, timesteps, n_features))
    X_test_3d = X_test.reshape((X_test.shape[0] // timesteps, timesteps, n_features))
    y_train_n = y_train.reshape(X_train.shape[0]//10, timesteps).mean(axis=1)
    y_test_n = y_test.reshape(X_test.shape[0]//10, timesteps).mean(axis=1)
    nn.fit(X_train_3d, [bool(x) for x in y_train_n])
    score = nn.score(X_test_3d, y_test_n)
    pred = nn.predict_proba(X_test)
    print("DL Accuracy score: ", score)

    return nn, score, pred
