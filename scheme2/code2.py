#! -*- coding:utf-8 -*-
import os
import re
import gc
import sys
import json
import codecs
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choice
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


tqdm.pandas()
seed = 2019
random.seed(seed)
tf.set_random_seed(seed)
np.random.seed(seed)
warnings.filterwarnings('ignore')



################################################################
data_path = './data/'

train = pd.read_csv(data_path + 'Round2_train.csv', encoding='utf-8')
train2= pd.read_csv(data_path + './Train_Data.csv', encoding='utf-8')
train=pd.concat([train, train2], axis=0, sort=True)
test = pd.read_csv(data_path + 'round2_test.csv', encoding='utf-8')

train = train[train['entity'].notnull()]
test = test[test['entity'].notnull()]

train=train.drop_duplicates(['title','text','entity','negative','key_entity'])   # 去掉重复的data

print(train.shape) ###(10526, 6)
print(test.shape)  ####((9997, 4)


def get_or_content(y,z):
    s=''
    if str(y)!='nan':
        s+=y
    if str(z)!='nan':
        s+=z
    return s

#获取title+text
train['content']=list(map(lambda y,z: get_or_content(y,z),train['title'],train['text']))
test['content']=list(map(lambda y,z: get_or_content(y,z),test['title'],test['text']))


def entity_clear_row(entity,content):
    entities = entity.split(';')
    entities.sort(key=lambda x: len(x))
    n = len(entities)
    tmp = entities.copy()
    for i in range(n):
        entity_tmp = entities[i]
        #长度小于等于1
        if len(entity_tmp)<=1:
            tmp.remove(entity_tmp)
            continue
        if i + 1 >= n:
            break
        for entity_tmp2 in entities[i + 1:]:
            if entity_tmp2.find(entity_tmp) != -1 and (
                    entity_tmp2.find('?') != -1 or content.replace(entity_tmp2, '').find(entity_tmp) == -1):
                tmp.remove(entity_tmp)
                break
    return ';'.join(tmp)

train['entity']=list(map(lambda entity,content:entity_clear_row(entity,content),train['entity'],train['content']))
test['entity']=list(map(lambda entity,content:entity_clear_row(entity,content),test['entity'],test['content']))

test['text'] = test.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)
train = train[(train['entity'].notnull()) & (train['negative'] == 1)]   ### 

emotion = pd.read_csv('./submit/sub_qinggan_vote20191109_score0392098.csv', encoding='utf-8')
emotion = emotion[emotion['negative'] == 1]
test = emotion.merge(test, on='id', how='left')


################################################################
train_id_entity = train[['id', 'entity']]
train_id_entity['entity'] = train_id_entity['entity'].apply(lambda index: index.split(';'))
id, entity = [], [] 
for index in range(len(train_id_entity['entity'])):
    entity.extend(list(train_id_entity['entity'])[index])
    id.extend([list(train_id_entity['id'])[index]] * len(list(train_id_entity['entity'])[index]))

train_id_entity = pd.DataFrame({'id': id, 'entity_label': entity})

test_id_entity = test[['id', 'entity']]
test_id_entity['entity'] = test_id_entity['entity'].apply(lambda index: index.split(';'))
id, entity = [], [] 
for index in range(len(test_id_entity['entity'])):
    entity.extend(list(test_id_entity['entity'])[index])
    id.extend([list(test_id_entity['id'])[index]] * len(list(test_id_entity['entity'])[index]))

test_id_entity = pd.DataFrame({'id': id, 'entity_label': entity})

train = train.merge(train_id_entity, on='id', how='left')
train['flag'] = train.apply(lambda index: 1 if index.key_entity.split(';').count(index.entity_label) >= 1 else 0, axis=1)
test = test.merge(test_id_entity, on='id', how='left')

################################################################
print(train.shape)
print(test.shape)

def extract_feature(data):
    data['sub_word_num'] = data.apply(lambda index: index.entity.count(index.entity_label) - 1, axis=1)
    data['question_mark_num'] = data['entity_label'].apply(lambda index: index.count('?'))
    data['occur_in_title_num'] = data.apply(lambda index: 0 if index.title is np.nan else index.title.count(index.entity_label), axis=1)
    data['occur_in_text_num'] = data.apply(lambda index: 0 if index.text is np.nan else index.text.count(index.entity_label), axis=1)
    data['occur_in_partial_text_num'] = data.apply(lambda index: 0 if index.text is np.nan else index.text[:507].count(index.entity_label), axis=1)
    data['occur_in_entity'] = data.apply(lambda index: 0 if index.text is np.nan else index.entity.count(index.entity_label) - 1, axis=1)
    data['is_occur_in_article'] = data.apply(lambda index: 1 if (index.occur_in_title_num >= 1) | (index.occur_in_text_num >= 1) else 0, axis=1)
    return data

train = extract_feature(train)
test = extract_feature(test)
print(train.columns)


train['entity_len'] = train['entity_label'].progress_apply(lambda index: len(index))
test['entity_len'] = test['entity_label'].progress_apply(lambda index: len(index))

train[train['entity_len'] == 1].shape
train = train[train['entity_len'] > 1]

test[test['entity_len'] == 1].shape
test = test[test['entity_len'] > 1]

train_feature = train[['sub_word_num', 'question_mark_num', 'occur_in_title_num', 'occur_in_text_num', 'is_occur_in_article', 'occur_in_entity', 'occur_in_partial_text_num']]
test_feature = test[['sub_word_num', 'question_mark_num', 'occur_in_title_num', 'occur_in_text_num', 'is_occur_in_article', 'occur_in_entity', 'occur_in_partial_text_num']]

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_feature = scaler.fit_transform(train_feature)
test_feature = scaler.fit_transform(test_feature)

def get_other_content(x,y):
    entitys=x.split(";")
    if len(entitys)<=1:
        return  np.nan
    l=[]
    for e in entitys:
        if e!=y:
            l.append(e)
    return ';'.join(l)

train['other_entity']=list(map(lambda x,y :get_other_content(x,y),train['entity'],train['entity_label']))
test['other_entity']=list(map(lambda x,y :get_other_content(x,y),test['entity'],test['entity_label']))

def get_content(x,y):
    if str(y)=='nan':
        return x
    y=y.split(";")
    y = sorted(y, key=lambda i:len(i),reverse=True)
    for i in y:
        x = '其他实体'.join(x.split(i))
    return x

train['text']=list(map(lambda x,y: get_content(x,y), train['text'], train['other_entity']))
test['text']=list(map(lambda x,y: get_content(x,y), test['text'], test['other_entity']))

maxlen = 509
bert_path = 'E:/chinese_wwm_ext_L-12_H-768_A-12/'     # chinese_L-12_H-768_A-12    chinese_wwm_ext_L-12_H-768_A-12
config_path = bert_path + 'bert_config.json'
checkpoint_path = bert_path + 'bert_model.ckpt'
dict_path = bert_path + 'vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)  # 给每个token 按序编号

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

class data_generator:
    def __init__(self, data, feature, batch_size=8, shuffle=True):    # 8
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature = feature
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            
            if self.shuffle:
                np.random.shuffle(idxs)
            
            X1, X2, Y, Fea = [], [], [], []
            for i in idxs:
                d = self.data[i]
                fea = self.feature[i]   # add feature
                first_text = d[0]
                second_text = d[2][:maxlen - d[1]]
                x1, x2 = tokenizer.encode(first=first_text, second=second_text)   # , max_len=512
                y = d[3]
                Fea.append(fea)
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2, padding=1)
                    Fea = seq_padding(Fea)
                    Y = seq_padding(Y)
                    yield [X1, X2, Fea], Y[:, 0, :]
                    [X1, X2, Y, Fea] = [], [], [], []

from keras.metrics import top_k_categorical_accuracy
from keras.metrics import categorical_accuracy

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
                    

def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
#         print(l)
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(train_feature.shape[1],))
    
    feature = Dense(64, activation='relu')(x3_in)
    
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    x = concatenate([x, feature])
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in, x3_in], p)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(1e-5),                # lr: 5e-5   3e-5   2e-5    epoch: 3, 4    batch_size: 16, 32    
                  metrics=['accuracy', f1_metric])      # categorical_accuracy
    print(model.summary())
    return model


################################################################
from keras.utils import to_categorical

DATA_LIST = []
for data_row in train.iloc[:].itertuples():
    DATA_LIST.append((data_row.entity_label, data_row.entity_len, data_row.text, to_categorical(data_row.flag, 2)))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.entity_label, data_row.entity_len, data_row.text, to_categorical(0, 2)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)
################################################################

f1, acc = [], []
def run_cv(nfold, data, feature_train, data_label, data_test, feature_test):
    kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(data, train['flag'])
    train_model_pred = np.zeros((len(data), 2))           # 2
    test_model_pred = np.zeros((len(data_test), 2))       # 2

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        X_train_fea, X_valid_fea = feature_train[train_fold, :], feature_train[test_fold, :]
        
        model = build_bert(2)                             # 2
        early_stopping = EarlyStopping(monitor='val_acc', patience=2)   # val_acc
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=1)
        checkpoint = ModelCheckpoint('./model/' + str(i) + '.hdf5', monitor='val_acc', 
                                         verbose=2, save_best_only=True, mode='max',save_weights_only=True)

        train_D = data_generator(X_train, X_train_fea, shuffle=True)
        valid_D = data_generator(X_valid, X_valid_fea, shuffle=False)
        test_D = data_generator(data_test, feature_test, shuffle=False)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),   ## ?? ##
            epochs=10,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
            verbose=2
        )

        model.load_weights('./model/' + str(i) + '.hdf5')
        
        # return model
        val = model.predict_generator(valid_D.__iter__(), steps=len(valid_D),verbose=0)
        
        print(val)
        score = f1_score(train['flag'].values[test_fold], np.argmax(val, axis=1))
        acc_score = accuracy_score(train['flag'].values[test_fold], np.argmax(val, axis=1))
        global f1, acc
        f1.append(score)
        acc.append(acc_score)
        print('validate f1 score:', score)
        print('validate accuracy score:', acc_score)

        train_model_pred[test_fold, :] = val
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D),verbose=0)

        del model; gc.collect()
        K.clear_session()
    return train_model_pred, test_model_pred


################################################################
train_model_pred, test_model_pred = run_cv(10, DATA_LIST, train_feature, None, DATA_LIST_TEST, test_feature)
print('validate aver f1 score:', np.average(f1))
print('validate aver accuracy score:', np.average(acc))
np.save('weights/bert_prob_train_binary_label_add_feature_extend_trainSet-PreProcess-roberta-large.npy', train_model_pred)
np.save('weights/bert_prob_test_binary_label_add_feature_extend_trainSet-PreProcess-roberta-large.npy', test_model_pred)
################################################################

# 结果一 #
def return_list(group):
    return ';'.join(list(group))

sub = test.copy()
sub['label'] = [np.argmax(index) for index in test_model_pred]

test_2 = pd.read_csv(data_path + 'round2_test.csv', encoding='utf-8')
submit = test_2[['id']]

sub = sub[sub['label'] == 1]
key_entity = sub.groupby(['id'], as_index=False)['entity_label'].agg({'key_entity': return_list})

submit = submit.merge(key_entity, on='id', how='left')
submit['negative'] = submit['key_entity'].apply(lambda index: 0 if index is np.nan else 1)
submit = submit[['id', 'negative', 'key_entity']]
submit.to_csv('submit/sub_binary_label_roberta-large.csv', encoding='utf-8', index=None)
print(submit[submit['key_entity'].notnull()].shape)


################################################################
# 结果二 #
def return_list(group):
    return ';'.join(list(group))

sub = test.copy()
sub['label'] = [np.argmax(index) for index in test_model_pred]
test['prob'] = [index[1] for index in test_model_pred]

sub = sub[sub['label'] == 1]
key_entity = sub.groupby(['id'], as_index=False)['entity_label'].agg({'key_entity': return_list})

sub_id = set(test['id']) - set(key_entity['id'])
sub_test = test[test['id'].isin(sub_id)]
sub_test = sub_test.sort_values(by=['id', 'prob'], ascending=False).drop_duplicates(['id'], keep='first')
sub_test['key_entity'] = sub_test['entity_label']
key_entity = pd.concat([key_entity, sub_test[['id', 'key_entity']]], axis=0, ignore_index=True)

test_2 = pd.read_csv(data_path + 'round2_test.csv', encoding='utf-8')
submit = test_2[['id']]

submit = submit.merge(key_entity, on='id', how='left')
submit['negative'] = submit['key_entity'].apply(lambda index: 0 if index is np.nan else 1)
submit = submit[['id', 'negative', 'key_entity']]
submit.to_csv('submit/sub_binary_label_roberta-large_all_neg_samples.csv', encoding='utf-8', index=None)
print(submit[submit['key_entity'].notnull()].shape)

