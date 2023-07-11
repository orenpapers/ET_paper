import datetime
import joblib
import pandas as pd

scale_col = "x_gaze_location_standard_scaled"

def apply_phq_cutoff(df , neg_phq_cutoff, pos_phq_cutoff):
    df["phq_binary_label"] = [0.0 if x <= neg_phq_cutoff else 1.0 if x >= pos_phq_cutoff else "other" for x in df.phq_score]
    df = df[df.phq_binary_label!= 'other']
    return df


def get_timecols_df_for_DL(fn ="/Users/orenkobo/Desktop/PhD_new/repos/HebLingStudy/notebooks/df.csv",
                           scale_col = scale_col,
                           word_embedding_dict_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/word2alephbert_encodingdict.jbl"):

    import string, re
    print(f"{datetime.datetime.now()} Reading csv from {fn}")
    df = pd.read_csv(fn,
                     index_col=None,
                     converters={#'alephbert_enc': eval,
                         scale_col : eval,
                         # 'x_gaze_location_minmax_scaled' : eval,
                         # 'x_gaze_location_standard_scaled' : eval,
                         # 'target_word_x_range' : eval
                         # 'phq_label': bool
                     })
    print(df.shape)

    def exclude_char(s):
        exclude = set(string.punctuation)
        exclude.remove("-")
        return ''.join(ch for ch in s if ch not in exclude)
    # wed_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/Output_2021_Aug/word2alephbert_encodingdict.jbl"
    wed_fn = "word2alephbert_encodingdict.jbl"
    words_embedding_dict = joblib.load(wed_fn)
    df = df[df.Sentence_type != 'F'].reset_index(drop=True)
    df["words_order"] = df["words_order"].apply(lambda x : [exclude_char(w) for w in eval(x)])
    df['word_embeddings_order'] = df["words_order"].apply(lambda x : [words_embedding_dict[w] for w in x])

    id_cols = ["phq_score","phq_group","Subject", "Sentence_type",
               "sentence_pupil_diameter_mean","set_num", "words_order","word_embeddings_order"]
    # vec_size = 3500
    # new_colname = f"x_gaze_location_{vec_size}"
    cols = [f"timepoint#{i}" for i in range(875)]
    # df[new_colname] = df["x_gaze_location_standard_scaled"].apply(lambda x : x[:vec_size])
    timeseries_df = pd.DataFrame(data = df[scale_col].to_list() , columns = cols)
    timeseries_df[id_cols] = df[id_cols]
    timeseries_df = timeseries_df.iloc[:,200:]
    cols = [x for x in timeseries_df.columns if "timepoint" in x]
    return timeseries_df, cols

import numpy as np
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments
from transformers import BertTokenizer

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import AutoModelWithTabular

from multimodal_transformers.model import TabularConfig

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(metadata={
        'help': 'the path to the csv file containing the dataset'
    })
    column_info_path: str = field(
        default=None,
        metadata={
            'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
        })

    column_info: dict = field(
        default=None,
        metadata={
            'help': 'a dict referencing the text, categorical, numerical, and label columns'
                    'its keys are text_cols, num_cols, cat_cols, and label_col'
        })

    categorical_encode_type: str = field(default='ohe',
                                         metadata={
                                             'help': 'sklearn encoder to use for categorical data',
                                             'choices': ['ohe', 'binary', 'label', 'none']
                                         })
    numerical_transformer_method: str = field(default='yeo_johnson',
                                              metadata={
                                                  'help': 'sklearn numerical transformer to preprocess numerical data',
                                                  'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                              })
    task: str = field(default="classification",
                      metadata={
                          "help": "The downstream training task",
                          "choices": ["classification", "regression"]
                      })

    mlp_division: int = field(default=4,
                              metadata={
                                  'help': 'the ratio of the number of '
                                          'hidden dims in a current layer to the next MLP layer'
                              })
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                     metadata={
                                         'help': 'method to combine categorical and numerical features, '
                                                 'see README for all the method'
                                     })
    mlp_dropout: float = field(default=0.1,
                               metadata={
                                   'help': 'dropout ratio used for MLP layers'
                               })
    numerical_bn: bool = field(default=True,
                               metadata={
                                   'help': 'whether to use batchnorm on numerical features'
                               })
    use_simple_classifier: str = field(default=True,
                                       metadata={
                                           'help': 'whether to use single layer or MLP as final classifier'
                                       })
    mlp_act: str = field(default='relu',
                         metadata={
                             'help': 'the activation function to use for finetuning layers',
                             'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                         })
    gating_beta: float = field(default=0.2,
                               metadata={
                                   'help': "the beta hyperparameters used for gating tabular data "
                                           "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                               })

    def __post_init__(self):
        assert self.column_info != self.column_info_path
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)


logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'


d = {}
# df_fn = "/export/home/orenkobo/Aim1/paper_analysis/Artifacts/df_new_full__unsegmented_alldata_new_FINAL.csv"
df_fn = "df_new_full__unsegmented_alldata_new_FINAL.csv"
# df_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/ts_data/Artifacts2/df_new_full__unsegmented_alldata_new_FINAL.csv"

et_scale_col = "x_gaze_location"
override_cutoff = [7,8]
df, timepoint_cols = get_timecols_df_for_DL(fn =df_fn, scale_col = et_scale_col)
df = apply_phq_cutoff(df,
                      neg_phq_cutoff = override_cutoff[0],
                      pos_phq_cutoff = override_cutoff[1])


cond_df = pd.get_dummies(df['Sentence_type'])
cond_cols = cond_df.columns.tolist()
df = pd.concat([df, cond_df],axis=1)
cat_feats = ['A','B','C','D']
num_feats = ['sentence_pupil_diameter_mean']
nlp_feats = ['words_order']

text_cols = nlp_feats# ['Title', 'Review Text']
cat_cols = cat_feats# ['Clothing ID', 'Division Name', 'Department Name', 'Class Name']
numerical_cols = num_feats#['Rating', 'Age', 'Positive Feedback Count']

column_info_dict = {
    'text_cols': text_cols,
    'num_cols': numerical_cols,
    'cat_cols': cat_cols,
    'label_col': 'phq_binary_label',
    'label_list': [0,1]
}

model_args = ModelArguments(
    # model_name_or_path= '/Users/orenkobo/Desktop/PhD/Aim1/LM/alephbert-base' #'bert-base-uncased'
    model_name_or_path= "/export/home/orenkobo/Aim1/paper_analysis/LM/alephbert-base"
)

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)

tokenizer = BertTokenizer.from_pretrained("/export/home/orenkobo/Aim1/paper_analysis/LM/alephbert-base")
num_labels = 2

import numpy as np
from scipy.special import softmax
from sklearn.metrics import auc,  precision_recall_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix, matthews_corrcoef

def calc_classification_metrics(p: EvalPrediction):
    pred_labels = np.argmax(p.predictions, axis=1)
    pred_scores = softmax(p.predictions, axis=1)[:, 1]
    labels = p.label_ids

    roc_auc_pred_score = roc_auc_score(labels, pred_scores)
    acc_score = accuracy_score(labels, pred_labels)
    precisions, recalls, thresholds = precision_recall_curve(labels,
                                                             pred_scores)
    fscore = (2 * precisions * recalls) / (precisions + recalls)
    fscore[np.isnan(fscore)] = 0
    ix = np.argmax(fscore)
    threshold = thresholds[ix].item()
    pr_auc = auc(recalls, precisions)
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
    result = {'roc_auc': roc_auc_pred_score,
              'threshold': threshold,
              'pr_auc': pr_auc,
              'recall': recalls[ix].item(),
              'accuracy' : acc_score,
              'precision': precisions[ix].item(), 'f1': fscore[ix].item(),
              'tn': tn.item(), 'tp': tp.item(), 'fp': fp.item(), 'fn': fn.item()
              }


    return result

import numpy as np
from sklearn.model_selection import LeavePGroupsOut
groups1 = df['Subject']
lpgo1 = LeavePGroupsOut(n_groups=20)
combine_method = 'gating_on_cat_and_num_feats_then_sum'
res_dict = {}
i = 0
num_iters = 3
print("QQQQQQQQQQQQQQQQ")
for tmp_index, test_index in lpgo1.split(X = df[timepoint_cols] , y = df['phq_binary_label'], groups = groups1):
    if i==num_iters:
        break
    print(f"{datetime.datetime.now()} : start iter {i}")
    res_dict[i] = {}
    lpgo2 = LeavePGroupsOut(n_groups=20)
    test_subjects = list(np.unique(groups1.iloc[test_index]))
    for train_index, val_index in lpgo2.split(X = df.iloc[tmp_index][timepoint_cols] , y = df.iloc[tmp_index]['phq_binary_label'],
                                              groups = df.iloc[tmp_index]['Subject']):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        val_df = df.iloc[val_index]
        break
    train_df.to_csv(f'train_iter{i}.csv')
    val_df.to_csv(f'val_iter{i}.csv')
    test_df.to_csv(f'test_iter{i}.csv')
    print('Num examples train-val-test')
    print(len(train_df), len(val_df), len(test_df))

    data_args = MultimodalDataTrainingArguments(
        data_path='',
        combine_feat_method=combine_method,
        column_info=column_info_dict,
        task='classification'
    )

    training_args = TrainingArguments(
        output_dir=f"./multimodal_logs/model_name_iter{i}",
        logging_dir=f"./multimodal_logs/runs_iter{i}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        evaluate_during_training=True,
        logging_steps=25,
        eval_steps=250
    )

    set_seed(training_args.seed)
    # Get Datasets
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=data_args.column_info['label_list'],
        categorical_cols=data_args.column_info['cat_cols'],
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
    )
    tabular_config = TabularConfig(num_labels=num_labels,
                                   cat_feat_dim=train_dataset.cat_feats.shape[1],
                                   numerical_feat_dim=train_dataset.numerical_feats.shape[1],
                                   **vars(data_args))
    config.tabular_config = tabular_config
    model = AutoModelWithTabular.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calc_classification_metrics,
    )
    print("BBBBBBBBBBBBB")
    trainer.train()
    eval_result = trainer.evaluate(eval_dataset=val_dataset)
    print(f"{datetime.datetime.now()} : eval of iter {i} is {eval_result}")
    res_dict[i]["eval_result"] = eval_result