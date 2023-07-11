from keras.wrappers.scikit_learn import KerasClassifier
from keras import models as keras_models
from keras import layers
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
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
from OutputsAnalyser.run_timepoints_utils import get_image_df, apply_phq_cutoff
import matplotlib.pyplot as plt
#https://chrisalbon.com/code/deep_learning/keras/k-fold_cross-validating_neural_networks/
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import LeavePGroupsOut
import pandas as pd
import numpy as np
import  datetime
def train_conv2d_no_sentype(X_train, y_train, X_val, y_val, epochs = 5):

    #todo add image data generator
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(550,800,3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
                  metrics = ['accuracy'])

    history = model.fit(X_train,tf.one_hot(y_train , depth = y_train.nunique()),epochs = epochs ,
                        validation_data = (X_val, tf.one_hot(y_val , depth = y_val.nunique())))

    return model, history

def train_simple_MLP_with_sen_type(df_full, y_full, train_idx, test_idx):
    epochs = 20
    print(f"{datetime.datetime.now()} : train_simple_MLP_with_sen_type")

    X_hm_data_col = np.array(df_full[hm_data_col].values.tolist())
    X_cond = pd.get_dummies(df_full["Sentence_type"], prefix='Cond')

    X_train, X_val = X_hm_data_col[train_idx.tolist(), :].astype('float32'), \
                     X_hm_data_col[test_idx, :].astype('float32')

    cond_train, cond_val = X_cond.iloc[train_idx].astype('float32'), \
                            X_cond.iloc[test_idx].astype('float32')

    y_train, y_val = y_full[train_idx].astype('float32'), y_full[test_idx].astype('float32')

    input_image = Input(shape=(550,800,3))
    input_sentype = Input(shape=(5))

    x_image = (Conv2D(32,3,padding="same", activation="relu", input_shape=(550,800,3)))(input_image)
    x_image = (MaxPool2D())(x_image)
    x_image = (Dropout(0.4))(x_image)
    x_image = (Flatten())(x_image)
    x_image = (Dense(128,activation="relu"))(x_image)

    concat_layer = concatenate([x_image, input_sentype])

    y = Dense(1, activation="linear")(concat_layer)

    model = Model(inputs=[input_image, input_sentype], outputs=y)
    opt = Adam(lr=0.1, decay=0.01 / 200)
    loss = "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt)
    history = model.fit([X_train, cond_train.values.astype('float')],tf.one_hot(y_train , depth = y_train.nunique()),epochs = epochs,
                        validation_data = ([X_val, cond_val], tf.one_hot(y_val , depth = y_val.nunique()))
    )
    return model, history

def train_simple_MLP_with_sen_type_and_sentence(df_full, y_full, train_idx, test_idx):
    epochs = 20
    # X_ling_col = np.array(df_full["alephbert_enc"].values.tolist())

    # np.array(list(df[ling_feat]), dtype=float)
    X_ling_col = np.array(df_full["alephbert_enc"].values.tolist())

    X_hm_data_col = np.array(df_full[hm_data_col].values.tolist())
    # X_cond = pd.get_dummies(df_full["Sentence_type"], prefix='Cond')

    X_train, X_val = X_hm_data_col[train_idx.tolist(), :].astype('float32'), \
                     X_hm_data_col[test_idx, :].astype('float32')

    ling_train, ling_val = X_ling_col[train_idx.tolist(), :].astype('float32'), \
                           X_ling_col[test_idx, :].astype('float32')

    y_train, y_val = y_full[train_idx].astype('float32'), y_full[test_idx].astype('float32')

    print(f"{datetime.datetime.now()} : train_simple_MLP_with_sen_type")
    input_image = Input(shape=(550,800,3))
    input_sentype = Input(shape=(768))

    x_image = (Conv2D(32,3,padding="same", activation="relu", input_shape=(550,800,3)))(input_image)
    x_image = (MaxPool2D())(x_image)
    x_image = (Dropout(0.4))(x_image)
    x_image = (Flatten())(x_image)
    x_image = (Dense(128,activation="relu"))(x_image)

    concat_layer = concatenate([x_image, input_sentype])

    y = Dense(1, activation="linear")(concat_layer)

    model = Model(inputs=[input_image, input_sentype], outputs=y)
    opt = Adam(lr=0.1, decay=0.01 / 200)
    loss = "binary_crossentropy"
    model.compile(loss=loss, optimizer=opt)
    history = model.fit([X_train, ling_train],tf.one_hot(y_train , depth = y_train.nunique()),epochs = epochs ,
                        validation_data = ([X_val, ling_val], tf.one_hot(y_val , depth = y_val.nunique())))
    return model, history


def viz_training(model, history, epochs, output_dir):

    print("Plotting model to ", output_dir)
    tf.keras.utils.plot_model(
        model,
        to_file= output_dir + "multimodal_kats_unsegmented_projection_model.png",
        show_shapes=True,
        # show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    predictions = model.predict_classes(X_val)
    predictions = predictions.reshape(1,-1)[0]
    print(classification_report(y_val, predictions, target_names = ['Rugby (Class 0)','Soccer (Class 1)']))
    model.summary()

def create_image_classifier():
    # https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418
    # https://www.thepythoncode.com/article/image-classification-keras-python
    # https://towardsdatascience.com/image-classification-using-tensorflow-in-python-f8c978824edc
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
    pass

# Wrap Keras model so it can be used by scikit-learn

fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new.csv"
train_type = "full"
cv_histories = {}

if train_type == "full":
    df_full,  hm_data_col = get_image_df(fn, apply_mean = False)
    df_full = apply_phq_cutoff(df_full, neg_phq_cutoff = 4,
                                    pos_phq_cutoff = 7 ).reset_index()

    y_full = df_full["phq_binary_label"]
    groups = df_full['Subject']

    lpgo = LeavePGroupsOut(n_groups=2)
    cv_i = 0
    for train_idx, test_idx in lpgo.split(df_full, y_full, groups):
        print(f"{datetime.datetime.now()} : cv = {cv_i}")
        cv_histories['conv2d_with_sentype'] = []

        # cv_model, cv_history = train_simple_MLP_with_sen_type(X_train, y_train, X_val, y_val, cond_train, cond_test)
        cv_model1, cv_history1 = train_simple_MLP_with_sen_type_and_sentence(df_full, y_full, train_idx, test_idx)
        cv_model2, cv_history2 = train_simple_MLP_with_sen_type(df_full, y_full, train_idx, test_idx)
        cv_histories['conv2d_with_sentype'].append(cv_history)
    print("Done!!!")

if train_type == "mean":
    df_subj_mean, mean_hm_data_col = get_image_df(fn, apply_mean = True)
    df_subj_mean = apply_phq_cutoff(df_subj_mean, neg_phq_cutoff = 4,
                                pos_phq_cutoff = 7 )

    X_mean_hm_data_col = df_subj_mean[mean_hm_data_col]
    X_cond = df_subj_mean['Sentence_type']
    y_mean = df_subj_mean["phq_binary_label"]
    Xd_mean_hm_data_col = np.array(X_mean_hm_data_col.values.tolist()).astype('float32')
    yd_mean = y_mean.astype('float32')

    #Train cond-averaged models - no sen type
    for i in range(10):
        cv_histories['conv2d_no_sentype'] = []
        X_train, X_val, y_train, y_val = train_test_split(Xd_mean_hm_data_col,yd_mean, test_size=0.2, random_state=42)
        cv_model, cv_history = train_conv2d_no_sentype(X_train, y_train, X_val, y_val)
        cv_histories['conv2d_no_sentype'].append(cv_history)


# simple_heatmap_cnn = KerasClassifier(build_fn= create_image_classifier(),
#                                      epochs=10,
#                                      batch_size=100,
#                                      verbose=0)
#
# simple_MLP = KerasClassifier(build_fn=create_simple_MLP(),
#                              epochs=10,
#                              batch_size=100,
#                              verbose=0)

