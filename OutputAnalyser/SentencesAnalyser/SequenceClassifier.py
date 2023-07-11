import numpy as np
import pandas as pd
import os, random
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV , LogisticRegression , LinearRegression
# import geopy.distance
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import matplotlib.pyplot as plt
# from keras.utils import plot_model
# from keras.optimizers import Adam
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score , mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier
from yellowbrick.contrib.classifier import DecisionViz
from yellowbrick.classifier import ROCAUC, ConfusionMatrix, ClassificationReport
from yellowbrick.regressor import ResidualsPlot
from OutputsAnalyser.SentencesAnalyser.SentencesAnalyser import create_measures_df

word_type = "target_word"
conds = ["A", "B", "C", "D"]
cols_to_features_prediction = []
for measure_type in ["total_gaze_duration", "first_fixation_duration", "second_fixation_duration", "num_of_fixations",
                     "regression_path_duration", "skipping_probability", "word_has_first_pass_regression"]:

    cols_to_features_prediction += [x + "_" + word_type + "_" + measure_type for x in conds]


def draw_reg_residual_plot(alg, x_train, y_train, x_test, y_test, analysis_output_dir_path):
    visualizer = ResidualsPlot(alg)

    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show(analysis_output_dir_path + "/reg_residual_plot.png")

def draw_clas_rep(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path):
    visualizer = ClassificationReport(alg, classes=lbls, support=True)
    visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show(analysis_output_dir_path + "/classification_report.png")

def draw_conf_mat(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path):
    cm = ConfusionMatrix(alg, classes=lbls)
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(analysis_output_dir_path + "/confusion_matrix.png")


def draw_roc_auc(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path):
    visualizer = ROCAUC(alg, classes=lbls)
    visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.show(analysis_output_dir_path + "/roc_auc.png")


def draw_dec_bounderies(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path):
    viz = DecisionViz(alg, title="Dec boundaries", classes=lbls,
                      features=['A_target_word_total_gaze_duration', 'A_target_word_first_fixation_duration'])
    viz.fit(x_train, y_train)
    viz.draw(x_test, y_test)
    viz.show(analysis_output_dir_path + "/classifier_decision_boundaries.png")

def draw_evaluations(alg_type, alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path):

    if alg_type == "cls":
        draw_conf_mat(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path)
        draw_clas_rep(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path)
        draw_roc_auc(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path)
        draw_dec_bounderies(alg, x_train, y_train, x_test, y_test, lbls, analysis_output_dir_path)

    if alg_type == "reg":
        draw_reg_residual_plot(alg, x_train, y_train, x_test, y_test, analysis_output_dir_path)

class SequenceClassifier:

    def __init__(self, all_subjects_df, group_gazeroute_df, per_subject_gaze_route_group_df, all_sentences_all_subjects_df, analysis_output_dir_path):
        self.all_subjects_df = all_subjects_df
        self.per_subject_gaze_route_group_df = per_subject_gaze_route_group_df
        self.group_gazeroute_df = group_gazeroute_df
        self.all_sentences_all_subjects_df = all_sentences_all_subjects_df
        self.analysis_output_dir_path = analysis_output_dir_path

    def classify_PHQ_group_by_measures(self, should_draw_evaluations , alg_type ="cls"):
        for alg_type in ["cls", "reg"]:
            if alg_type == "cls":
                y_cn = "phq_level"
                alg = LogisticRegressionCV(cv=2)
                acc_fn = accuracy_score
            elif alg_type == "reg":
                y_cn = "phq_score"
                alg = LinearRegression()
                acc_fn = mean_squared_error
                return

            cols = cols_to_features_prediction + [y_cn]
            df = self.all_subjects_df[cols]
            df = df.dropna(subset=[y_cn])
            y_true = df.pop(y_cn)

            X = df.values
            x_train, x_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
            alg.fit(x_train, y_train)

            y_train_pred = alg.predict(x_train)
            train_acc = acc_fn(y_train_pred, y_train)

            y_test_pred = alg.predict(x_test)
            test_acc = acc_fn(y_test_pred, y_test)

            print("For {} train_acc={}, test_acc={} (train for {} rows and test for {} rows".format(alg_type , train_acc , test_acc, len(x_train) , len(x_test)))
            lbls = ['Mild', 'Minimal', 'Moderate', 'Moderately Severe']
            if should_draw_evaluations:
                draw_evaluations(alg_type, alg, x_train, y_train, x_test, y_test, lbls, self.analysis_output_dir_path)


    def classify_sentence_type_by_measures(self, all_sentences_all_subjects_df, should_draw_evaluations):

        X_cols = ['second_fixation_duration',  'total_gaze_duration', 'num_of_fixations', 'word_has_first_pass_regression', 'regression_path_duration',  'is_skipping_trial', 'trial_total_distance_covered', 'sentence_pupil_diameter_mean']
        df = all_sentences_all_subjects_df.dropna(subset=X_cols)
        y = df["Sentence_type"]
        word_type_one_hot_encoded_columns_df = pd.get_dummies(df["word_type"])
        measures_X = df[X_cols]
        X = pd.concat([measures_X , word_type_one_hot_encoded_columns_df] , axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = ExtraTreesClassifier()
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train_pred, y_train)

        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test_pred, y_test)

        print("For {} train_acc={}, test_acc={} (train for {} rows and test for {} rows".format("ExtraTreesClassifier", train_acc,
                                                                                                test_acc, len(X_train), len(X_test)))
        lbls = list(y.unique())
        if should_draw_evaluations:
            draw_evaluations("cls", clf, X_train, y_train, X_test, y_test, lbls, self.analysis_output_dir_path)

    def classify_sentence_type_by_raw_sequence(self, should_draw_evaluations, batch_size = 32):

        locations_tup_vec_as_str = self.all_sentences_all_subjects_df["trial_actual_tuples_of_orig_gaze_locations"].apply(lambda x : x.replace("[","").replace("]","")).dropna()
        locations_tup_vec_as_list = list(locations_tup_vec_as_str.apply(lambda x : eval(x)))

        locations_tup_vec_as_df = pd.DataFrame(locations_tup_vec_as_list)
        labels = self.all_sentences_all_subjects_df["Sentence_type"]
        #X is raw sequence of locations - locations_tup_vec_as_list
        #y is Sentence_type, here called labels to prevent from confusing with y as a location
        x_train, x_test, y_train, y_test = train_test_split(locations_tup_vec_as_df, labels, test_size=0.33, random_state=42)

        model = _get_model()
        chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
        adam = Adam(lr=0.001)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=4,
                  validation_data=[x_test, y_test])

        # loading the model and checking accuracy on the test data
        model = load_model('best_model.pkl')


        y_pred = model.predict_classes(y_test)
        accuracy_score(y_pred, y_test)

    def classify_PHQ_level_by_raw_sequence(self, should_draw_evaluations , batch_size = 32):

        X , y = 1 , 2
        maxlen = []
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        print('Train...')
        model = _get_model()
        chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
        adam = Adam(lr=0.001)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=4,
                  validation_data=[x_test, y_test])

        # loading the model and checking accuracy on the test data
        model = load_model('best_model.pkl')


        y_pred = model.predict_classes(y_test)
        accuracy_score(y_pred, y_test)

    def draw_conf_mat(self):
        pass

def _get_model(seq_len = 10):

    adam = Adam(lr=0.001)
    chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)

    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_len, 4)))
    #
    #
    # bidirectional lstm  ? model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
    # add maxpooling (let's say, one from every x time-window), to reduce dimensionality
    # replace with  model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='/Users/orenkobo/Downloads/reconstruct_lstm_autoencoder.png')
    print(model.summary())
    return model

