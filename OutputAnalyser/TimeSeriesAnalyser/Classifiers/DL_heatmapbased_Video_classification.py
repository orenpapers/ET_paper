_MD_COLS = ['video_segment_num', 'sentence_idx']
_ET_COLS = ['sentence_type_isA', 'sentence_type_isB', 'sentence_type_isC', 'sentence_type_isD',
            'sentence_pupil_diameter_mean', 'word_pupil_diameter_mean', 'word_has_first_pass_regression', 'trial_total_distance_covered',
            'is_skipping_trial', 'regression_path_duration', 'num_of_fixations', 'total_gaze_duration', 'second_fixation_duration','first_fixation_duration']
_NLP_COLS = ['sentence_encoding']
_CV_COLS = ['frame_vgg_3d_data']
_LABEL_COL = ['phq_group']
_ID_COLS = ['subject_id']

import keras
from keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Flatten
from keras.utils import plot_model
from OutputAnalyser.TimeSeriesAnalyser import analysis_output_dir
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def _build_video_net(vgg_fe = "1d"):

    frames, channels, rows, columns = 5,3,224,224

    video = Input(shape=(frames,
                         rows,
                         columns,
                         channels))

    cnn_base = VGG16(input_shape=(rows,
                                  columns,
                                  channels),
                     weights="imagenet",
                     include_top=True)
    cnn_base.trainable = False

    cnn = Model(cnn_base.input, cnn_base.layers[-3].output, name="VGG_fm")
    encoded_frames = TimeDistributed(cnn , name = "encoded_frames")(video)
    encoded_sequence = LSTM(256, name = "encoded_seqeunce")(encoded_frames)
    hidden_layer = Dense(1024, activation="relu" , name = "hidden_layer")(encoded_sequence)

    encoding_input = Input(shape=(784,), name="Encoded_sentence", dtype='float')
    sentence_features = Dense(units = 60, name = 'sentence_features')(encoding_input)

    fixations_input = Input(shape=(16,), name="Fixation_features", dtype='float')

    x = layers.concatenate([sentence_features, hidden_layer, fixations_input])
    outputs = Dense(10, activation="softmax")(x)

    model = Model([video,encoding_input, fixations_input], outputs) #<=== double input
    model.summary()

    return model

def _generate_video_dataset(df):
    train_idx = df[df["df_type"] == "train"].index
    val_idx = df[df["df_type"] == "val"].index
    test_idx = df[df["df_type"] == "test"].index

    X_video = df[_ID_COLS+_CV_COLS + _NLP_COLS + _ET_COLS]
    y_video = df[_LABEL_COL]
    X_video_train = X_video.iloc[train_idx]
    X_video_test = X_video.iloc[test_idx]
    X_video_val = X_video.iloc[val_idx]
    y_video_train = y_video.iloc[train_idx]
    y_video_test = y_video.iloc[test_idx]
    y_video_val = y_video.iloc[val_idx]

    return X_video, X_video_train, X_video_test, X_video_val, y_video, y_video_train, y_video_test, y_video_val

def _video_classification(df):
    video_lstm_cnn_net = _build_video_net()
    X_video, X_video_train, X_video_test, X_video_val, y_video, y_video_train, y_video_test, y_video_val = _generate_video_dataset(df)

    early_stop = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=75, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001)
    model_cp = ModelCheckpoint(filepath = ".model_prod.hdf5", monitor= 'loss',
                               verbose=0, save_best_only=False, save_weights_only=False,
                               mode='auto', save_freq=5)
    callbacks=[reduce_lr, early_stop]

    history = video_lstm_cnn_net.fit(x=X_video_train.groupby(), y=y_video_train, verbose=1, callbacks=callbacks,
                                          epochs=800, batch_size = 128, validation_data=(X_video_val, y_video_val))

    test_scores = video_lstm_cnn_net.evaluate(X_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    output_plot = analysis_output_dir + "/video_net.png"
    print("Saved video_lstm_cnn_net plot to ", output_plot)
    plot_model(video_lstm_cnn_net, output_plot, show_shapes=True)

    return test_scores, video_lstm_cnn_net

def video_classification_manager(df):
    sc_dict = {}
    models_dict = {}

    df = df[_ID_COLS+_CV_COLS + _NLP_COLS + _ET_COLS + _MD_COLS]

    sc_video , model_video = _video_classification()

    sc_dict["DL_heatmap_video"] = sc_video
    models_dict["DL_heatmap_video"] = model_video
    return sc_dict , models_dict