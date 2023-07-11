import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import Activation, Bidirectional , LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression , LogisticRegression
from scipy import signal
import datetime, os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import OutputAnalyser.TimeSeriesAnalyser.ts_params
from OutputAnalyser.TimeSeriesAnalyser import tensorboard_logs_path
from sklearn.preprocessing import LabelEncoder
# os.cmd(f"rm -rf ./{tensorboard_logs_path}/")
log_dir = tensorboard_logs_path + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
KATS_DSP_FEATS = ['length', 'mean', 'var', 'entropy', 'lumpiness', 'stability', 'flat_spots', 'hurst', 'std1st_der', 'crossing_points', 'binarize_mean', 'unitroot_kpss', 'heterogeneity', 'histogram_mode', 'linearity', 'trend_strength', 'seasonality_strength', 'spikiness', 'peak', 'trough', 'level_shift_idx', 'level_shift_size', 'y_acf1', 'y_acf5', 'diff1y_acf1', 'diff1y_acf5', 'diff2y_acf1', 'diff2y_acf5', 'y_pacf5', 'diff1y_pacf5', 'diff2y_pacf5', 'seas_acf1', 'seas_pacf1', 'firstmin_ac', 'firstzero_ac', 'holt_alpha', 'holt_beta']
#https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

early_stop = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=20, restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.0001)


def create_multimodal_rawdata_unsegmented_projection(dim, layers):
    input_heatmap = Input(shape=(4096,))
    input_sentence = Input(shape=(768,))
    input_kats = Input(shape=(3500,))

    # the first branch operates on the first input
    heatmap_projection  = Dense(25, activation="linear")(input_heatmap)
    sentence_projection = Dense(25, activation="linear")(input_sentence)

    concat_layer = concatenate([input_kats.output, sentence_projection.output, heatmap_projection.output])
    x = Dense(64, activation="relu")(concat_layer)
    x = Dense(16, activation="relu")(x)
    y = Dense(1, activation="linear")(x)

    x = Model(inputs=[input_kats, input_sentence, input_heatmap], outputs=x)
    model = Model(inputs=[x.input, y.input], outputs=y)

def create_multimodal_LSTMs(hparams_d, h_s, l_s, k_s):

    input_heatmap = tf.keras.Input(shape=(h_s), name="heatmap_input")
    input_sentence = tf.keras.Input(shape=(l_s), name="sentence_input")
    input_kats = tf.keras.Input(shape=(k_s), name="features_input")
    # input_cat = tf.keras.Input(shape=(s4,), name="categorical_input")

    heatmap_projection  = tf.keras.layers.Dense(300, activation=hparams_d["activation"], name="heatmap_projection")(input_heatmap)
    sentence_projection = tf.keras.layers.Dense(300, activation=hparams_d["activation"], name="sentence_projection")(input_sentence)
    kats_projection = tf.keras.layers.Dense(30, activation=hparams_d["activation"], name="kats_projection")(input_kats)

    combined = concatenate([
        heatmap_projection,
        sentence_projection,
        kats_projection
    ])

    LSTM_A = Bidirectional(LSTM(64))(combined)
    # LSTM_B = Bidirectional(LSTM(32))(sentence_projection)


    dense1 = Dense(16, activation='relu')(LSTM_A)
    output = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[
        input_heatmap,
        input_sentence,
        input_kats
    ], outputs=output)

    opt = Adam(lr=hparams_d["activation"]["lr"])#, decay=0.01 / 200)
    loss = "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt)

    print("Plotting model to ", OutputsAnalyser.TimeSeriesAnalyser.ts_params.artifacts_dir)
    tf.keras.utils.plot_model(
        model,
        to_file= OutputsAnalyser.TimeSeriesAnalyser.ts_params.artifacts_dir + "multimodal_kats_unsegmented_projection_model.png",
        show_shapes=True,
        # show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )
    return model

def create_multimodal_kats_unsegmented_projection(s1,s2,s3, s4):
    # define two sets of inputs

    input_heatmap = tf.keras.Input(shape=(s3,), name="heatmap_input")
    input_sentence = tf.keras.Input(shape=(s2,), name="sentence_input")
    input_kats = tf.keras.Input(shape=(s1,), name="features_input")
    input_cat = tf.keras.Input(shape=(s4,), name="categorical_input")
    # the first branch operates on the first input
    heatmap_projection  = tf.keras.layers.Dense(25, activation="linear", name="heatmap_projection")(input_heatmap)
    sentence_projection = tf.keras.layers.Dense(25, activation="linear", name="sentence_projection")(input_sentence)

    concat_layer = tf.keras.layers.concatenate([input_kats, sentence_projection, heatmap_projection], name="modalities_concat")
    x = Dense(64, activation="relu")(concat_layer)
    x = Dense(16, activation="relu")(x)
    x = Reshape(input_shape=(16,), target_shape=(16, 1))(x)
    x = Conv1D(filters=4,  kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    x = tf.keras.layers.concatenate([x , input_cat])
    y = Dense(2, activation="softmax")(x)

    model = Model(inputs=[input_kats, input_sentence, input_heatmap, input_cat], outputs=y)
    opt = Adam(lr=0.1, decay=0.01 / 200)
    loss = "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt)
    print("Plotting model to ", OutputsAnalyser.TimeSeriesAnalyser.ts_params.artifacts_dir)
    tf.keras.utils.plot_model(
        model,
        to_file= OutputsAnalyser.TimeSeriesAnalyser.ts_params.artifacts_dir + "multimodal_kats_unsegmented_projection_model.png",
        show_shapes=True,
        # show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )
    return model

def create_multimodal_mlp_segmented_pca(dim, layers = [8, 4], regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(layers[0], input_dim=dim, activation="relu"))
    model.add(Dense(layers[1], activation="relu"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    else:
        model.add(Dense(1, activation="softmax"))
    # return our model
    opt = Adam(lr=0.01, decay=0.01 / 200)
    loss = "mean_absolute_percentage_error" if regress else "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt)
    return model

def create_cnn(inputShape, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    # inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)         # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)
        # check to see if the regression node should be added
        if regress:
            x = Dense(1, activation="linear")(x)

        # construct the CNN
        model = Model(inputs, x)
        # return the CNN
        return model



def get_res(preds, y_test, reg):
    if reg:
        diff = preds - y_test
        percentDiff = (diff / y_test) * 100
        res = np.abs(percentDiff)
    else:
        preds = [round(x) for x in preds]
        res = sum(preds == y_test) / len(y_test)
    return res

def apply_phq_cutoff(df , neg_phq_cutoff, pos_phq_cutoff):
    df["phq_label"] = [0.0 if x <= neg_phq_cutoff else 1.0 if x >= pos_phq_cutoff else "other" for x in df.phq_score]
    df = df[df.phq_label!= 'other']
    return df


def get_linear_df_from_nn_df(df, label_cat_feat):
    linear_reg_rows = []

    for subject_id, subject_df in df.groupby(["Subject"]):
        assert (subject_df[label_cat_feat].nunique() == 1)

        row = [subject_id] + \
              [np.mean(subject_df[subject_df.Sentence_type == st].nn_pred) for st in ["A","B","C","D","F"]] + \
              [subject_df[label_cat_feat].iloc[0]]
        linear_reg_rows.append(row)
    linear_reg_df = pd.DataFrame(data = linear_reg_rows , columns=["subject_id","A_pred","B_pred","C_pred","D_pred",
                                                                   "F_pred", "subject_label"])
    return linear_reg_df

def create_time_window_for_lstm(df, group_key, ling_feat, heatmap_feat, KATS_DSP_FEATS, y_col):

    def split_into_inputs(subject_df):
        x_data_inp1.append(list(subject_df[ling_feat]))
        x_data_inp2.append(list(subject_df[heatmap_feat]))
        x_data_inp3.append(list(subject_df["kats"]))
        assert (subject_df[y_col].nunique() == 1)
        y_data.append(subject_df[y_col].unique()[0])

    df["kats"] = df[KATS_DSP_FEATS].apply(lambda row: list(eval(' , '.join(row.values.astype(str)))), axis=1)
    x_data_inp1 = []
    x_data_inp2 = []
    x_data_inp3 = []
    y_data = []

    df = df.sort_values(by=['Subject','Sentence_type'])
    df.groupby([group_key]).apply(lambda group: split_into_inputs(group))

    x_data_inp1 = np.array(x_data_inp1, dtype=float)
    x_data_inp2 = np.array(x_data_inp2, dtype=float)
    x_data_inp3 = np.array(x_data_inp3, dtype=float)

    # Convert labels from chars into digits - creating instance of labelencoder - Assigning numerical values. Convert 'A','B' into 0, 1
    y_data = LabelEncoder().fit_transform(y_data)

    return x_data_inp1, x_data_inp2, x_data_inp3, y_data


def model2talos(nn_model, hparams_d, x_data_h_inp_train, x_data_l_inp_train, x_data_k_inp_train, y_train ):
    #https://github.com/autonomio/talos
    out = nn_model.fit(
        x = [x_data_h_inp_train, x_data_l_inp_train, x_data_k_inp_train],
        y = y_train,
        # validation_data=([x_data_h_inp_val, x_data_l_inp_val, x_data_k_inp_val], y_val),
        epochs=hparams_d["epoch"],
        # batch_size=,
        callbacks=[tensorboard_callback]#, reduce_lr, early_stop]
    )

    return out, nn_model

def train_unsegmented_projection_multimodal_network(df, override_cutoff):
    clf_res_dict = {}
    clf_res_dict["iters"] = {}
    label_cat_feat = "phq_label"
    label_reg_feat = "phq_score"
    df = apply_phq_cutoff(df, neg_phq_cutoff = override_cutoff[0], pos_phq_cutoff = override_cutoff[1] )
    y_vec = df[label_cat_feat]
    sentence_type_onehot_feats = ["Sentence_type_A", "Sentence_type_B","Sentence_type_C","Sentence_type_D","Sentence_type_F"]
    ling_feat = "alephbert_enc"
    heatmap_feat = "1d_vgg_featuremap"
    KATS_DSP_FEATS.append("trial_total_distance_covered")

    hparams_d = {'activation':['relu', 'linear'],
             'lr'         : [0.01, 0.05],
             'batch_size' : [4,16,32],
             'epochs'     : [20,40]
              }


    nn_model = create_multimodal_LSTMs(hparams_d, l_s=(96, 768), h_s=(96, 4096), k_s=(96,38))
    gss = GroupShuffleSplit(n_splits=4, train_size=.8, random_state=42)

    for iter_num, idx in enumerate(gss.split(X = df, y = y_vec, groups=df.Subject)):

        print("[Iter #{}]".format(iter_num))
        train_idx_all, test_idx = idx[0], idx[1]
        df_train_all = df.iloc[train_idx_all]
        y_train_all = y_vec.iloc[train_idx_all]
        val_gss = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=42)
        train_idx, val_idx = list(val_gss.split(X = df_train_all , y=y_train_all , groups=df_train_all.Subject))[0]

        df_train = df_train_all.iloc[train_idx]
        df_val = df_train_all.iloc[val_idx]
        df_test = df.iloc[test_idx]

        x_data_l_inp_train, x_data_h_inp_train, x_data_k_inp_train, y_train = create_time_window_for_lstm(df_train, "Subject",  ling_feat, heatmap_feat, KATS_DSP_FEATS,label_cat_feat)
        x_data_l_inp_test, x_data_h_inp_test, x_data_k_inp_test, y_test = create_time_window_for_lstm(df_test, "Subject",  ling_feat, heatmap_feat, KATS_DSP_FEATS,label_cat_feat)
        x_data_l_inp_val, x_data_h_inp_val, x_data_k_inp_val, y_val = create_time_window_for_lstm(df_val, "Subject",  ling_feat, heatmap_feat, KATS_DSP_FEATS,label_cat_feat)


        train_subjs , test_subjs, val_subjs = list(df_train.Subject.unique()),  list(df_val.Subject.unique()), list(df_test.Subject.unique())
        print("Subject on:\ntrain ({}) : {}\ntest ({}): {}\nval ({}): {}".format(len(train_subjs), train_subjs, len(test_subjs), test_subjs, len(val_subjs), val_subjs))
        print("Fitting model... {} train , {} val , {} test".format(len(df_train), len(df_val), len(df_test)))


        iter_history = nn_model.fit(
            x = [x_data_h_inp_train, x_data_l_inp_train, x_data_k_inp_train],
            y = y_train,
            # validation_data=([x_data_h_inp_val, x_data_l_inp_val, x_data_k_inp_val], y_val),
            epochs=200,
            # batch_size=,
            callbacks=[tensorboard_callback]#, reduce_lr, early_stop]
        )

        # hp_model = model2talos(nn_model, hparams_d, x_data_h_inp_train, x_data_l_inp_train, x_data_k_inp_train, y_train)
        # import talos
        # scan_object = talos.Scan(x = [x_data_h_inp_train, x_data_l_inp_train, x_data_k_inp_train], y = y_train,
        #                          model=hp_model, params=hparams_d, experiment_name='iris', fraction_limit=0.1)


        print("TensorBoard logs are at : " , tensorboard_logs_path)

        # make predictions on the testing data
        print("[INFO] predicting PHQ...")

        per_sentence_prediction = [x[0] for x in nn_model.predict([x_data_h_inp_test, x_data_l_inp_test, x_data_k_inp_test])]
        sentences_res = get_res(per_sentence_prediction, y_test, reg=False)
        print("[RESULT] sentences_res = {} (Chance level is {})".format(round(sentences_res, 2) , sum(y_test) / len(y_test)))
        clf_res_dict["iters"][iter_num] = {}
        clf_res_dict["iters"][iter_num]['model_history'] = iter_history
        clf_res_dict["iters"][iter_num]['sentences_res'] = sentences_res
    a = 2
        # df_train["nn_pred"] = [x[0] for x in nn_model.predict([kats_train, ling_train, heatmap_train, cat_train])]
        # df_val["nn_pred"] = [x[0] for x in nn_model.predict([kats_val, ling_val, heatmap_val, cat_val])]
        # df_test["nn_pred"] = [x[0] for x in nn_model.predict([kats_test, ling_test, heatmap_test, cat_test])]

        # linear_reg_df_train = get_linear_df_from_nn_df(pd.concat([df_train, df_val], axis=0), label_cat_feat)
        # linear_reg_df_test = get_linear_df_from_nn_df(df_test, label_cat_feat)
        #
        # linear_X_train = linear_reg_df_train[["A_pred","B_pred","C_pred","D_pred","F_pred"]]
        # linear_X_test = linear_reg_df_test[["A_pred","B_pred","C_pred","D_pred","F_pred"]]
        #
        # linear_y_train = linear_reg_df_train["subject_label"]
        # linear_y_test = linear_reg_df_test["subject_label"]
        #
        # sentences_to_subject_linear_model = LogisticRegression( random_state = 7)
        # sentences_to_subject_linear_model.fit(linear_X_train, linear_y_train)
        #
        # subject_pred = sentences_to_subject_linear_model.predict_proba(linear_X_test)
        # subject_results = get_res(subject_pred , linear_y_test, reg=False)
        # print("[RESULT] subject_results = {} (Chance level is {})".format(round(subject_results,2) , sum(linear_y_test) / len(linear_y_test)))
        # a = 2

def train_multimodal_model_segmented_and_pca(df, reg, override_cutoff):

    clf_res_dict = {}
    clf_res_dict["iters"] = {}
    label_cat_feat = "phq_label"
    label_reg_feat = "phq_score"
    label_col = label_reg_feat if reg else label_cat_feat

    print("[INFO] processing data...")
    if not reg and override_cutoff:
        df = apply_phq_cutoff(df, neg_phq_cutoff = override_cutoff[0], pos_phq_cutoff = override_cutoff[1] )
        df = df.reset_index(drop=True)
        # y_vec = df[label_col]
        # df["y"] = y_vec

    ling_enc_pca_feats = "alephbert_enc_20_PCA"
    vgg_featuremap_pca_feats = "VGG_1d_featuremap_20_PCA"
    dsp_feats = KATS_DSP_FEATS#[x for x in df.columns if "seglearn_featrep" in x and x!="seglearn_featrep#2"]

    df_alephbert_enc_20_PCA  = df[ling_enc_pca_feats].apply(pd.Series)
    df_alephbert_enc_20_PCA.columns = ["enc_pca#{}".format(i) for i in range(df_alephbert_enc_20_PCA.shape[1])]
    df_VGG_1d_featuremap_20_PCA  = df[vgg_featuremap_pca_feats].apply(pd.Series)
    df_VGG_1d_featuremap_20_PCA.columns = ["vgg_pca#{}".format(i) for i in range(df_VGG_1d_featuremap_20_PCA.shape[1])]

    x_feats = list(df_alephbert_enc_20_PCA.columns) + list(df_VGG_1d_featuremap_20_PCA.columns) + dsp_feats
    x_feats = [x for x in x_feats if x not in ["hw_alpha","hw_beta","hw_gamma"]]
    print("has df with shape {}".format(df.shape))
    df = pd.concat([df, df_alephbert_enc_20_PCA, df_VGG_1d_featuremap_20_PCA] , axis=1)
    print("concat with pca dfs to get shape of {} - using {} feats".format(df.shape , len(x_feats)))
    gss = GroupShuffleSplit(n_splits=4, train_size=.8, random_state=42)
    print("before dropna based on feats - {}".format(df.shape))
    df = df.dropna(subset=x_feats).reset_index(drop=True)
    y_vec = df[label_col]
    print("After dropna based on feats - {}".format(df.shape))
    X_all, y_all = df[x_feats] , y_vec

    for iter, idx in enumerate(gss.split(X = df, y = y_vec, groups=df.Subject)):

        print("[Iter #{}]".format(iter))
        train_idx_all, test_idx = idx[0], idx[1]
        df_train_all = df.iloc[train_idx_all]
        y_train_all = y_vec.iloc[train_idx_all]
        val_gss = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=42)
        train_idx, val_idx = list(val_gss.split(X = df_train_all , y=y_train_all , groups=df_train_all.Subject))[0]
        df_train = df_train_all.iloc[train_idx]
        df_val = df_train_all.iloc[val_idx]
        df_test = df.iloc[test_idx]

        X_train = df_train[x_feats]
        X_test = df_test[x_feats]
        X_val = df_val[x_feats]

        y_train = y_train_all.iloc[train_idx]
        y_test = y_vec.iloc[test_idx]
        y_val = y_train_all.iloc[val_idx]

        train_subjs , test_subjs, val_subjs = list(df_train.Subject.unique()),  list(df_val.Subject.unique()), list(df_test.Subject.unique())
        print("Subject on:\ntrain ({}) : {}\ntest ({}): {}\nval ({}): {}".format(len(train_subjs), train_subjs, len(test_subjs), test_subjs, len(val_subjs), val_subjs))

        mlp_model = create_multimodal_mlp_segmented_pca(X_train.shape[1], regress=reg)

        print("Fitting model...")
        mlp_model.fit(
            x=X_train.values.astype('float32'), y=y_train.astype('float32'),
            validation_data=(X_val.astype('float32'), y_val.astype('float32')),
            epochs=5, batch_size=8,
            callbacks=[reduce_lr, early_stop])
        # make predictions on the testing data
        print("[INFO] predicting PHQ...")
        preds = [x[0] for x in mlp_model.predict_proba(X_test)]
        seg_res = get_res(preds, y_test, reg)
        print("[INFO] res is {}".format(seg_res))
        df["segment_score"] = mlp_model.predict(X_all)

        print("Create sentnece score from segments scores")
        for k , sentence_segments_rows in df.groupby(["Subject","sentence_run_num","sentence_trial_num"]):
            sentence_labels = sentence_segments_rows[label_col]
            assert (len(sentence_labels.unique()) == 1)
            segment_standalone_probes = sentence_segments_rows.segment_score
            segment_weighting_window = signal.windows.gaussian(len(sentence_segments_rows), std=len(sentence_segments_rows)/4) #https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.signal.windows.gaussian.html?highlight=scipy%20signal%20gaussian#scipy.signal.windows.gaussian
            sentence_score = sum([x[0] * x[1] for x in zip(segment_standalone_probes , segment_weighting_window)]) / sum(segment_weighting_window)
            df.loc[sentence_segments_rows.index, "sentence_score"] = sentence_score

        print("Going over {} subjects to create sentence-to-subject model".format(len(df.Subject.unique())))
        rows = []
        for k , subject_rows in df.groupby(["Subject"]):
            subject_row_per_sentence = subject_rows.drop_duplicates(["Sentence"])\
                                                   .sort_values(["Sentence_type", "Sentence"])

            sentence_labels = subject_row_per_sentence[label_col]
            assert (len(sentence_labels.unique()) == 1)
            subject_label = sentence_labels.iloc[0]
            sentence_scores = subject_row_per_sentence.sentence_score.values.tolist()
            try:
                assert (len(subject_row_per_sentence) == 96)
            except Exception as e:
                print("Only {} rows for subject #{}".format(len(subject_row_per_sentence), k))
                continue
            row = [k] + sentence_scores + [subject_label]
            rows.append(row)

        x_cols = ['Subject']
        for stype in ["A","B","C","D", "F"]:
            sc = 64 if stype == "F" else 8
            for snum in range(sc):
                c = "sentype_{}#{}".format(stype, snum)
                x_cols.append(c)

        df_lreg = pd.DataFrame(data=rows , columns=x_cols + ["subject_label"])

        df_lreg_train = df_lreg[df_lreg.Subject.isin(train_subjs + val_subjs)]
        df_lreg_test  = df_lreg[df_lreg.Subject.isin(test_subjs)]

        subject_lreg_model = LinearRegression() if reg else LogisticRegression()
        subject_lreg_model.fit(df_lreg_train[x_cols], df_lreg_train['subject_label'])
        if reg:
            subject_lreg_model_probs_vec = [x for x in subject_lreg_model.predict(df_lreg_test[x_cols])]
        else:
            subject_lreg_model_probs_vec = [x[1] for x in subject_lreg_model.predict_proba(df_lreg_test[x_cols])]
            subject_lreg_model_confmat =  confusion_matrix([x for x in subject_lreg_model.predict(df_lreg_test[x_cols])], df_lreg_test['subject_label']).ravel()

        subject_lreg_model_res = subject_lreg_model.score(df_lreg_test[x_cols], df_lreg_test['subject_label'])

        clf_res_dict["iters"]["iter{}".format(iter)] = {}
        clf_res_dict["iters"]["iter{}".format(iter)]["train_subj"] = train_subjs
        clf_res_dict["iters"]["iter{}".format(iter)]["test_subj"] = test_subjs
        clf_res_dict["iters"]["iter{}".format(iter)]["val_subj"] = val_subjs
        clf_res_dict["iters"]["iter{}".format(iter)]["segment_model_fitted"] = mlp_model
        clf_res_dict["iters"]["iter{}".format(iter)]["segment_model_score"] = seg_res
        clf_res_dict["iters"]["iter{}".format(iter)]["subject_lreg_model_score"] = subject_lreg_model_res
        clf_res_dict["iters"]["iter{}".format(iter)]["subject_lreg_model_probs_vec"] = subject_lreg_model_probs_vec
        clf_res_dict["iters"]["iter{}".format(iter)]["subject_lreg_model_fitted"] = subject_lreg_model
        if not reg:
            clf_res_dict["iters"]["iter{}".format(iter)]["subject_lreg_model_confmat"] = subject_lreg_model_confmat

    clf_res_dict["mean_subj_acc"] = np.mean([x[1]['subject_lreg_model_score'] for x in clf_res_dict["iters"].items()])
    clf_res_dict["mean_segment_acc"] = np.mean([x[1]['segment_model_score'] for x in clf_res_dict["iters"].items()])
    return clf_res_dict
    # configs_result_dict[config] = res
        #
        # print("Create dsp_feats_mlp")
        # dsp_feats_mlp = create_mlp(df_train[dsp_feats].shape[1], regress=reg)
        # print("Create sent_enc_mlp")
        # sent_enc_mlp = create_mlp(len(df_train.iloc[0][ling_enc_feat]), regress=reg)
        # print("Create vgg_heatmap_mlp")
        # vgg_heatmap_mlp = create_mlp(len(df_train.iloc[0][vgg_featuremap_feat]), regress=reg)
        # print("Create heatmap_cnn")
        # print("Create input_configs_dict")
        # heatmap_cnn = create_cnn(df_train[cnn_3d_feat].shape, regress=reg)
        # # create the MLP and CNN models
        # input_configs_dict = {
        #      "dsp" : [dsp_feats_mlp , df_train[dsp_feats].astype('float32'),
        #                              df_test[dsp_feats].astype('float32')],
        #      "enc" : [sent_enc_mlp , df_train[ling_enc_feat].apply(pd.Series),
        #                              df_test[ling_enc_feat].apply(pd.Series)], #df_train.iloc[:5][ling_enc_feats].apply(lambda x: [[float(t) for t in y] for y in x])
        #      "vgg_heatmap" : [vgg_heatmap_mlp , np.vstack(df_train[vgg_featuremap_feat]).astype('float32'),
        #                                         np.vstack(df_test[vgg_featuremap_feat]).astype('float32')],
        #
        #      # "cnn_heatmap" : [heatmap_cnn , np.vstack(df_train[cnn_3d_feat]).astype('float32'),
        #      #                                np.vstack(df_test[cnn_3d_feat]).astype('float32')],
        #                       }
        #
        # sub_configurations = ["dsp", "enc" ,"vgg_heatmap"]#,"cnn_heatmap"]
        # configs = [list(x) for x in(set(compress(sub_configurations,mask))
        #                                 for mask in product(*[[0,1]]*len(sub_configurations))) if len(x) > 1]# if 'dsp' in x]
        # configs = [x for x in configs if vgg_heatmap_mlp not in x or heatmap_cnn not in x]
        # configs_result_dict = {}
        # for ci , config in enumerate(configs):
        #     print("[Config #{}/{}: {}]".format(ci, len(configs), " & " .join(config )))
        #     input_config_nets = [input_configs_dict[x][0] for x in config]
        #     # input_feats_models_train = pd.concat([df_train[input_configs_dict[x][1]] for x in config] , axis=1)
        #     # input_feats_models_train = [df_train[input_configs_dict[x][1]] for x in config]
        #     input_feats_models_train = [input_configs_dict[x][1] for x in config]
        #     input_feats_models_test = [input_configs_dict[x][2] for x in config]
        #     # input_feats_models_train = np.concatenate([input_configs_dict[x][1] for x in config], axis=1)
        #     # input_feats_models_test = np.concatenate([input_configs_dict[x][2] for x in config], axis=1)
        #     final_act = "linear" if reg else "softmax"
        #     combinedInput = concatenate([n.output for n in input_config_nets]) # create the input to our final set of layers as the *output* of both the MLP and CNN
        #     x = Dense(4, activation="relu")(combinedInput) # our final FC layer head will have two dense layers, the final one being our regression head
        #     x = Dense(1, activation=final_act)(x)
        #     model = Model(inputs=[n.input for n in input_config_nets], outputs=x) # our final model will accept categorical/numerical data on the MLP  input and images on the CNN input, outputting a single value
        #     opt = Adam(lr=0.01, decay=0.01 / 200)
        #     loss = "mean_absolute_percentage_error" if reg else "binary_crossentropy"
        #     model.compile(loss=loss, optimizer=opt)
        #
        #     print("[INFO] training model...")
        #     early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, restore_best_weights = True)
        #     reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=0.0001)
        #     model.fit(
        #         x=input_feats_models_train, y=y_train.astype('float32'),
        #         # validation_data=([input_feats_models_val], y_val),
        #         epochs=100, batch_size=8,
        #         callbacks=[reduce_lr, early_stop])
        #     # make predictions on the testing data
        #     print("[INFO] predicting PHQ...")
        #     preds = model.predict(input_feats_models_test)[:,0]
        #     res = get_res(preds, y_test, reg)
        #     configs_result_dict[config] = res


    # print(f"{datetime.datetime.now()} Creating time windows for lstm")
    # # Create windows of data:
    # df["kats"] = df[KATS_DSP_FEATS].apply(lambda row: list(eval(' , '.join(row.values.astype(str)))), axis=1)
    # list_of_indexes=[]
    # df.index.to_series().rolling(50).apply((lambda x: list_of_indexes.append(x.tolist()) or 0), raw=False)
    #
    # s_ling = df[ling_feat].apply(list)
    # s_heatmap = df[heatmap_feat].apply(list)
    # s_kats = df["kats"].apply(list)
    # s_y = df[y_col]
    # # s_y = df[y_col].apply(list)
    #
    # l = [[s_ling[ix] for ix in x] for x in list_of_indexes]
    # h = [[s_heatmap[ix] for ix in x] for x in list_of_indexes]
    # k = [[s_kats[ix] for ix in x] for x in list_of_indexes]
    # y = [[s_y[ix] for ix in x] for x in list_of_indexes]
    #
    # l_v = np.array(l)
    # h_v = np.array(h)
    # k_v = np.array(k)
    # y_v = np.array(y)
    #
    # print(f'l_v shape: {l_v.shape}, h_v shape: {h_v.shape}, k_v shape: {k_v.shape}, y_v shape {y_v.shape}')
    # return l_v, h_v, k_v, y_v
    # def create_data_for_multimodal_kats_unsegmented(df, ling_feat, heatmap_feat, sentence_type_onehot_feats):
    #     ling_data = np.vstack(df[ling_feat])
    #     heatmap_data = np.vstack(df[heatmap_feat])
    #     kats_data = df[KATS_DSP_FEATS].values
    #     cat_data = df[sentence_type_onehot_feats].values
    #     return ling_data, heatmap_data, kats_data, cat_data
    # ling_data, heatmap_data, kats_data, cat_data = create_data_for_multimodal_kats_unsegmented(df, ling_feat, heatmap_feat, sentence_type_onehot_feats)
    # nn_model = create_multimodal_kats_unsegmented_projection(s1=kats_data.shape[1],s2=ling_data.shape[1],
    #                                                          s3=heatmap_data.shape[1], s4= cat_data.shape[1])

    # kats_train = df_train[KATS_DSP_FEATS].values
    # ling_train = np.vstack(df_train[ling_feat])
    # heatmap_train = np.vstack(df_train[heatmap_feat])
    # cat_train = df_train[sentence_type_onehot_feats]
    #
    # kats_test = df_test[KATS_DSP_FEATS].values
    # ling_test = np.vstack(df_test[ling_feat])
    # heatmap_test = np.vstack(df_test[heatmap_feat])
    # cat_test = df_test[sentence_type_onehot_feats]
    #
    # kats_val = df_val[KATS_DSP_FEATS].values
    # ling_val = np.vstack(df_val[ling_feat])
    # heatmap_val = np.vstack(df_val[heatmap_feat])
    # cat_val = df_val[sentence_type_onehot_feats]
    #
    # y_train = y_train_all.iloc[train_idx]
    # y_test = y_vec.iloc[test_idx]
    # y_val = y_train_all.iloc[val_idx]
