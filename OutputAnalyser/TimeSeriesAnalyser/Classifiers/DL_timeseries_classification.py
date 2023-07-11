from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from OutputAnalyser.TimeSeriesAnalyser import phq2int_dict
from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import Segment
from OutputAnalyser.TimeSeriesAnalyser import split_df
import numpy as np
from OutputAnalyser.TimeSeriesAnalyser import segement_width
import time

def _conditionalrnn_model(cond , X_train,NUM_CELLS = 5):
    import cond_rnn
    #https://github.com/philipperemy/cond_rnn

    outputs = cond_rnn.ConditionalRNN(units=NUM_CELLS, cell='GRU')([X_train, cond])
    return outputs

def _crnn_model(width=segement_width, n_vars=1, n_classes=5, conv_kernel_size=5,
               conv_filters=3, lstm_units=3):
    input_shape = (width, n_vars)
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


def to_conditional_rnn_data(series_of_timeseries, df_cond):
    #return 3-D Tensor with shape [batch_size, timesteps, input_dim] from series series_of_timeseries
    #return 2-D Tensor or list of tensors with shape [batch_size, cond_dim] from cond
    #per https://github.com/philipperemy/cond_rnn , https://github.com/philipperemy/cond_rnn/issues
    a = 2
    x_3d = 1
    cond_2d = 2
    return x_3d , cond_2d

def _conditional_rnn_classification(df, series_of_timeseries, contextual_cols, y_col ='binary_label'):
    #https://github.com/philipperemy/cond_rnn/blob/master/examples/temp.py
    #https://github.com/philipperemy/cond_rnn/issues/15
    X = to_conditional_rnn_data(series_of_timeseries, df[contextual_cols])
    X_train , X_test , y_train , y_test , _ , _= split_df(df, series_of_timeseries, y_col, dictify_y = False)
    a = _conditionalrnn_model(df[contextual_cols] , X_train)
    f = 3

def _direct_1d_crnn_classification(df, series_of_timeseries, contextual_cols, y_col ='binary_label'):

    pipe = Pype([
                 ('crnn', KerasClassifier(build_fn=_crnn_model, n_classes = len(list(df[y_col].unique())), epochs=5, batch_size=4, verbose=5))
    ])
    #todo add contextual (CRNN(

    X_train , X_test , y_train , y_test , _ , _= split_df(df, series_of_timeseries, y_col, dictify_y = False)

    X_train_nonan = [np.nan_to_num(x, nan=0) for x in X_train]
    X_test_nonan = [np.nan_to_num(x, nan=0) for x in X_test]
    train_segmented_X, train_segmented_y, _= Segment(width = segement_width , overlap = 0.2).fit_transform(X_train_nonan, np.array(y_train))
    train_segmented_X_3d = np.reshape(train_segmented_X , (train_segmented_X.shape[0], train_segmented_X.shape[1], -1))

    test_segmented_X, test_segmented_y, _= Segment(width = segement_width , overlap = 0.2).fit_transform(X_test_nonan, np.array(y_test))
    test_segmented_X_3d = np.reshape(test_segmented_X , (test_segmented_X.shape[0], test_segmented_X.shape[1], -1))

    print("DL - N series in train: ", len(X_train))
    print("DL - N series in test: ", len(X_test))
    print("DL - N segments in train: ", len(train_segmented_X))
    print("DL - N segments in test: ", len(test_segmented_X))

    start = time.time()
    pipe.fit(train_segmented_X_3d , train_segmented_y)

    end = time.time()
    score = pipe.score(test_segmented_X_3d , test_segmented_y)

    print("DL - Accuracy score: ", score)
    print("DL - Classification time : " , end - start)

    return score, pipe

def dl_classification_manager(df, series_of_timeseries, contextual_cols):
    sc_dict = {}
    models_dict = {}

    sc_conditional , model_conditional = _conditional_rnn_classification(df = df, series_of_timeseries = series_of_timeseries,
                                                          contextual_cols = contextual_cols)

    sc , m1d = _direct_1d_crnn_classification(df = df, series_of_timeseries = series_of_timeseries,
                                              contextual_cols = None)
    sc_contextual, m1d_contextual = _direct_1d_crnn_classification(df = df, series_of_timeseries = series_of_timeseries,
                                                                   contextual_cols = contextual_cols)

    sc_dict["DL_raw"] = sc
    sc_dict["DL_raw_contextual"] = sc_contextual
    sc_dict["conditional_rnn"] = sc_conditional
    models_dict["CRNN_1d"] = m1d
    models_dict["CRNN_1d_contextual"] = m1d_contextual
    models_dict["conditional_rnn"] = model_conditional
    return sc_dict , models_dict
