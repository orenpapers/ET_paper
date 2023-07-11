import datetime
from tsai.all import *
from scipy import stats
import numpy as np
import keras
from OutputsAnalyser.run_timepoints_utils import get_timecols_df_for_DL, apply_phq_cutoff
from sklearn.model_selection import train_test_split, LeavePGroupsOut
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import concatenate, Activation, Bidirectional , LSTM, Dense,Flatten,Dropout, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os
n_timesteps = 675
n_features = 1
n_outputs = 1
num_iters = 25

def create_lstm_dataset(df, X, y):
    Xs, ys = [], []
    for i in range(len(df)):
        v = X.iloc[i].values
        label = y.iloc[i]
        Xs.append(v)
        ys.append(label)
    return Xs, ys

#https://curiousily.com/posts/time-series-classification-for-human-activity-recognition-with-lstms-in-keras/

def unify_list(l,chunk_size):
    return pd.DataFrame([l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]).mean(axis=1)

def run_lstm_cond(df, feats):
    cond_scores = []
    subj_scores = []
    Xlstm, ylstm = create_lstm_dataset(df, df[feats], df['phq_binary_label'])
    lpgo = LeavePGroupsOut(n_groups=10)

    for train_idx, test_idx in itertools.islice(lpgo.split(df[feats], df['phq_binary_label'], df['Subject']), num_iters):
        test_subj = df['Subject'].iloc[test_idx].unique()
        train_subj = df['Subject'].iloc[train_idx].unique()
        assert len(list(set(test_subj) & set(train_subj))) == 0
        # X_train, X_test, y_train, y_test = train_test_split(np.array(Xlstm), np.array(ylstm))
        X_train, X_test = np.array(Xlstm)[train_idx], np.array(Xlstm)[test_idx]
        y_train, y_test = np.array(ylstm)[train_idx], np.array(ylstm)[test_idx]
        Xlstm_train = X_train[:,:-4].reshape(-1,1,675)
        Xlstm_test = X_test[:,:-4].reshape(-1,1,675)
        Xcond_train = X_train[:,-4:]
        Xcond_test = X_test[:,-4:]

        input_lstm = Input(shape=(n_features, n_timesteps))
        input_condition = Input(shape=(4,))

        lstm1  = Bidirectional(LSTM(8))(input_lstm)
        x = Dropout(0.5)(lstm1)
        lstm_out = Dense(20, activation='relu')(x)
        x = concatenate([lstm_out, input_condition])
        # x = lstm_out
        x = Dense(1, activation='sigmoid')(x)
        # model = Model(inputs=input_lstm , input_condition], outputs=[x])
        model = Model(inputs=[input_lstm, input_condition] , outputs=[x])
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )
        history = model.fit(
            [Xlstm_train, Xcond_train], y_train,
            epochs=80,batch_size=64,validation_split=0.33
        )
        d = history.history
        results = model.evaluate([Xlstm_test, Xcond_test], y_test)
        cond_scores.append(results[1])

        subject_pred_per_cond = list(model.predict([Xlstm_test, Xcond_test]))
        chunk_size = df.Sentence_type.nunique()
        subject_pred_unified = [round(x) for x in unify_list(l=subject_pred_per_cond, chunk_size=chunk_size)]# pd.DataFrame([subject_pred_per_cond[i:i + chunk_size] for i in range(0, len(subject_pred_per_cond), chunk_size)]).mean()
        subject_label = unify_list(l=df.iloc[test_idx]['phq_binary_label'].tolist(), chunk_size=chunk_size)#  [x for x in df["phq_binary_label"].iloc[list(range(0, len(subject_pred_per_cond), chunk_size))]]
        subj_acc = accuracy_score(subject_pred_unified, subject_label)
        subj_scores.append(subj_acc)

    cond_sc = np.mean(cond_scores)
    subj_sc = np.mean(subj_scores)
    return cond_sc, subj_sc

def run_lstm(df, feats):
    scores = []
    c = feats + ['phq_binary_label']
    Xlstm, ylstm = create_lstm_dataset(df, df[c], df['phq_binary_label'])
    lpgo = LeavePGroupsOut(n_groups=10)

    for train_idx, test_idx in itertools.islice(lpgo.split(df[feats], df['phq_binary_label'], df['Subject']), 1):
        test_subj = df['Subject'].iloc[test_idx].unique()
        train_subj = df['Subject'].iloc[train_idx].unique()
        assert len(list(set(test_subj) & set(train_subj))) == 0
        # X_train, X_test, y_train, y_test = train_test_split(np.array(Xlstm), np.array(ylstm))
        X_train, X_test = np.array(Xlstm)[train_idx], np.array(Xlstm)[test_idx]
        y_train, y_test = np.array(ylstm)[train_idx], np.array(ylstm)[test_idx]

        # Xlstm_train, Xlstm_test, ylstm_train, ylstm_test = train_test_split(np.array(Xlstm), np.array(ylstm))
        Xlstm_train = X_train.reshape(-1,1,676)
        Xlstm_test = X_test.reshape(-1,1,676)
        print("Xlstm_train shape is " , Xlstm_train.shape)

        model = keras.Sequential()
        model.add(Bidirectional(LSTM(units=8,input_shape=[n_timesteps, n_features])))
        model.add(Dropout(rate=0.4))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
        history = model.fit(
            Xlstm_train, y_train,
            epochs=1500,batch_size=64,validation_split=0.2)

        d = history.history
        results = model.evaluate(Xlstm_test, y_test)
        print("====  test loss, test acc:", results)
        scores.append(results[1])
    sc = np.mean(scores)
    print("Total score is " ,sc )
    return sc
# print(Xlstm.shape, ylstm.shape)

def run_sentence_classification(df, feats, cond):
    from transformers import pipeline
    from simpletransformers.language_modeling import LanguageModelingModel
    from simpletransformers.classification import ClassificationModel, ClassificationArgs
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # alephbert_cls_pipeline = pipeline('text-classification', model='onlplab/alephbert-base', tokenizer='onlplab/alephbert-base')
    print("A : ", df.shape)
    df = df[df['words_order'].apply(lambda x: len(x)) > 2].reset_index(drop=True)
    print("B : ", df.shape)

    X = df["words_order"].apply(lambda x : " ".join(x))
    y = df["phq_binary_label"]
    print("Chance level is " , sum(y)/len(y))
    data_df = pd.DataFrame(data = [[X.iloc[i], y.iloc[i]] for i in range(len(df))], columns = ["text","labels"])
    model_args = ClassificationArgs(num_train_epochs=10, overwrite_output_dir=True,
                                    early_stopping_patience = 5 )

    # Create a ClassificationModel

    results = []
    accs = []
    for i in range(5):
        print(f"{datetime.datetime.now()} : Cond {cond} , iter {i}")
        model = ClassificationModel(
            model_type = 'bert', model_name = 'onlplab/alephbert-base', tokenizer_name='onlplab/alephbert-base',
            num_labels=2,  args=model_args, use_cuda=False )
        train, test = train_test_split(data_df, test_size=0.2)
        model.train_model(train)
        result, model_outputs, wrong_predictions = model.eval_model(test)
        results.append(result)
        accs.append((result['tp'] + result['tn']) / len(test))
    results_d = {}
    for k in result.keys():
        results_d[k] = np.mean([d[k] for d in results])
    results_d["acc"] = np.mean(accs)
    print(f"{datetime.datetime.now()}: For cond {cond} , Got : {results_d}")
    return results_d

def run_LSTM_with_cond_upper_embedding(df, feats):
    data_3d = [] #create a list of N samples, each is ndarray of 3500X5 , 5 being 1 x and 4 cond
    cond_cols = ['A','B','C','D']

    import tensorflow as tf
    import numpy as np

    from tensorflow.keras import layers
    from tensorflow.keras.models import Model

    num_timesteps = 3300
    max_features_values = [100, 3]
    num_observations = 40

    input_list = [[[np.random.randint(0, v) for _ in range(num_timesteps)]
                   for v in max_features_values]
                  for _ in range(num_observations)]
    from sklearn.preprocessing import LabelEncoder
    df['encoded_cond'] = LabelEncoder().fit_transform(df['Sentence_type'])
    labels = []
    for idx , row in df.iloc[:40].iterrows():
        l = []
        for col in feats:
            l.append([row[col]] + [row['encoded_cond']])
        data_3d.append(l)
        labels.append(row["phq_binary_label"])
    print("A2 : ", datetime.datetime.now())
    X_input = np.asarray(data_3d)
    X_input_reshaped = np.swapaxes(X_input,1,2)
    input_arr = np.array(input_list)  # shape (2, 3, 100)
    num_timesteps = 3300
    voc_size = 4  # 4
    inp1 = layers.Input(shape=(1, num_timesteps))  # TensorShape([None, 2, 100])
    inp2 = layers.Input(shape=(1, num_timesteps))  # TensorShape([None, 1, 100])
    x2 = layers.Embedding(input_dim=voc_size, output_dim=8)(inp2)  # TensorShape([None, 1, 100, 8])
    x2_reshaped = tf.transpose(tf.squeeze(x2, axis=1), [0, 2, 1])  # TensorShape([None, 8, 100])
    x = layers.concatenate([inp1, x2_reshaped], axis=1)
    x = layers.LSTM(32)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inp1, inp2], outputs=[x])
    inp1_np = input_arr[:, :1, :]
    inp2_np = input_arr[:, 1:, :]
    model.predict([inp1_np, inp2_np])

    inp1_np1 = X_input_reshaped[:, :1, :]
    inp2_np1 = X_input_reshaped[:, 1:, :]
    model.predict([inp1_np1, inp2_np1])
    print("A1 : ", datetime.datetime.now())

    # import tensorflow as tf
    # #https://stackoverflow.com/questions/70274978/deep-learning-how-to-split-5-dimensions-timeseries-and-pass-some-dimensions-thro?noredirect=1#comment124227501_70274978
    # input = tf.keras.layers.Input(shape=(3300, 5))
    # x1 = tf.expand_dims(input[:, :, 4], axis=-1)
    # x2 = tf.expand_dims(input[:, :, 3], axis=-1)
    # x3 = tf.expand_dims(input[:, :, 2], axis=-1)
    # embed_input = tf.concat([x1, x2, x3], axis=-1)
    # y = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(8, activation='relu'))(embed_input)
    # output = tf.keras.layers.Concatenate(axis=-1)([input[:, :, :2], y])
    # model = tf.keras.Model(input, output)
    # print(model(tf.random.uniform((samples, 3300, 5), maxval=3, dtype=tf.int32)).shape)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    his = model.fit(
        x=[inp1_np1, inp2_np1], y=np.array(labels))

    a = 2
    return his

def run_ts_transformer(df , timepoint_cols):
    tsdf = df[timepoint_cols]
    x = tsdf.values.reshape(tsdf.shape[0],1,tsdf.shape[1])
    y = df['phq_binary_label']
    batch_tfms = TSStandardize()
    clf = TSClassifier(x, y, arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
    clf.fit_one_cycle(100, 3e-4)
    a = 2
def run_multimodel_with_transformers(df, cat_feats = ['A','B','C','D'], num_feats = ['sentence_pupil_diameter_mean'],
                                     nlp_feat = 'words_order'):
    from transformers import BertConfig
    import torch
    from multimodal_transformers.model import BertWithTabular
    from multimodal_transformers.model import TabularConfig
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("/Users/orenkobo/Desktop/PhD/Aim1/LM/alephbert-base")
    bert_config = BertConfig.from_pretrained('/Users/orenkobo/Desktop/PhD/Aim1/LM/alephbert-base')

    tabular_config = TabularConfig(
        combine_feat_method='attention_on_cat_and_numerical_feats',  # change this to specify the method of combining tabular data
        cat_feat_dim=len(cat_feats),  # need to specify this
        numerical_feat_dim=len(num_feats),  # need to specify this
        num_labels=2,   # need to specify this, assuming our task is binary classification
        use_num_bn=False,
    )
    # 'onlplab/alephbert-base'

    texts = [" ".join(x) for x in df[nlp_feat]]
    model_inputs = tokenizer(texts, padding=True, padding_side='left')

    # 5 numerical features
    numerical_feat = torch.tensor(df[num_feats].values).float()
    # 9 categorical features
    categorical_feat = torch.tensor(df[cat_feats].values).float()
    labels = torch.tensor([1, 0])

    model_inputs['cat_feats'] = categorical_feat
    model_inputs['num_feats'] = numerical_feat
    model_inputs['labels'] = labels
    bert_config.tabular_config = tabular_config
    model = BertWithTabular.from_pretrained('/Users/orenkobo/Desktop/PhD/Aim1/LM/alephbert-base', config=bert_config)
    loss, logits, layer_outs = model(
        torch.tensor(model_inputs['input_ids']),
        token_type_ids=torch.tensor(model_inputs['token_type_ids']),
        labels=labels,
        cat_feats=categorical_feat,
        numerical_feats=numerical_feat
    )
    a = 2

    with torch.no_grad():
        _, logits, classifier_outputs = model(
            torch.tensor(model_inputs['input_ids']),
            token_type_ids=torch.tensor(model_inputs['token_type_ids']),
            cat_feats=categorical_feat,
            numerical_feats=numerical_feat
        )
    a = 2

def main():
    d = {}
    df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL.csv"
    et_scale_col = "x_gaze_location"
    override_cutoff = [7,8]
    df, timepoint_cols = get_timecols_df_for_DL(fn =df_fn, scale_col = et_scale_col)
    df = apply_phq_cutoff(df,
                          neg_phq_cutoff = override_cutoff[0],
                          pos_phq_cutoff = override_cutoff[1])

    cond_df = pd.get_dummies(df['Sentence_type'])
    cond_cols = cond_df.columns.tolist()
    df = pd.concat([df, cond_df],axis=1)
    d["LSTM_with_cond_NN"] = {}
    # sc_lstm_cond_cond_acc, sc_lstm_cond_subj_acc = run_lstm_cond(df, feats = timepoint_cols + cond_cols)
    # d["LSTM_with_cond_NN"]["cond_acc"] = sc_lstm_cond_cond_acc
    # d["LSTM_with_cond_NN"]["subj_acc"] = sc_lstm_cond_subj_acc
    # sc_lstm = run_lstm(df, feats = timepoint_cols )
    res_dict = {}
    for cond in ["A","B","C","D"]:
        cond_df = df[df.Sentence_type == cond]
        res_dict["sc_transformers"] = run_ts_transformer(cond_df, timepoint_cols = timepoint_cols)

    res_dict["sc_ts_with_cond"] = run_LSTM_with_cond_upper_embedding(df, feats = timepoint_cols)
    res_dict["multimodel_with_transformers"] = run_multimodel_with_transformers(df)
    # res_dict[f"sc_sentence_lstm_{cond}"] = run_sentence_classification(df[df.Sentence_type == cond], feats = timepoint_cols, cond=cond )

    import datetime
    print(f"{datetime.datetime.now} : {res_dict}")
    exit()
    # d["LSTM"] = sc_lstm
    # d["LSTM_cond_cond"] = sc_lstm_cond_cond
    # d["LSTM_cond_subj"] = sc_lstm_cond_subj
    print(d)
    #todo implement: https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/07_Time_Series_Classification_with_Transformers.ipynb
    #todo implement : https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4
    #todo implement https://colab.research.google.com/github/georgianpartners/Multimodal-Toolkit/blob/master/notebooks/text_w_tabular_classification.ipynb#scrollTo=MGjGmetXRsuS
    #todo implement https://multimodal-toolkit.readthedocs.io/en/latest/notes/introduction.html
if __name__ == "__main__":
    main()


