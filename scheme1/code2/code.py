import os
import re
import gc
import sys
import json
import codecs
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from random import choice
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

tqdm.pandas()
np.random.seed(214683)
warnings.filterwarnings('ignore')


data_path = 'datasets/'
train = pd.read_csv(data_path + 'Round2_train.csv', encoding='utf-8')
train2= pd.read_csv(data_path + 'Train_data.csv', encoding='utf-8')
train=pd.concat([train,train2],axis=0,sort=True)
test = pd.read_csv(data_path + 'round2_test.csv', encoding='utf-8')

train = train[train['entity'].notnull()]
test = test[test['entity'].notnull()]

train=train.drop_duplicates(['title','text','entity','negative','key_entity'])   #去掉重复的data


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

# 去掉空实体
def duplicate_entity(entity):
    def is_empty(x):
        return (x != '') & (x != ' ')

    if entity is np.nan:
        return entity
    else:
        entity = filter(is_empty, entity.split(';'))
        return ';'.join(list(set(entity)))

train['entity'] = train['entity'].apply(lambda index: duplicate_entity(index))
test['entity'] = test['entity'].apply(lambda index: duplicate_entity(index))

# 正则表达式清洗文本
def delete_tag(s):
    
    s = re.sub('\{IMG:.?.?.?\}', '', s)                    #图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)   #网址
    s = re.sub(re.compile('<.*?>'), '', s)                 #网页标签
#     s = re.sub(re.compile('&[a-zA-Z]+;?'), ' ', s)         #网页标签
#     s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), ' ', s)
#     r4=re.compile('\d{4}[-/]\d{2}[-/]\d{2}')             #日期
#     s=re.sub(r4,'某时',s)    
    return s

train['title'] = train['title'].apply(lambda x: delete_tag(x) if str(x)!='nan' else x)
train['text'] = train['text'].apply(lambda x: delete_tag(x) if str(x)!='nan' else x)
test['title'] = test['title'].apply(lambda x: delete_tag(x) if str(x)!='nan' else x)
test['text'] = test['text'].apply(lambda x: delete_tag(x) if str(x)!='nan' else x)

# 使用title来填充测试集中text的缺失值，text null:1
train['title'] = train.apply(lambda index: index.text if index.title is np.nan else index.title, axis=1)
test['title'] = test.apply(lambda index: index.text if index.title is np.nan else index.title, axis=1)
train['text'] = train.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)
test['text'] = test.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)

# 选取非空样本
train = train[train['entity'].notnull()] # train entity null:18
test = test[test['entity'].notnull()]    #  test entity null:16

# train
train_id_entity = train[['id', 'entity']]
train_id_entity['entity'] = train_id_entity['entity'].apply(lambda index: index.split(';'))
ids, entity = [], [] 
for index in range(len(train_id_entity['entity'])):
    entity.extend(list(train_id_entity['entity'])[index])
    ids.extend([list(train_id_entity['id'])[index]] * len(list(train_id_entity['entity'])[index]))
train_id_entity = pd.DataFrame({'id': ids, 'entity_label': entity}) # train len:11448

# test
test_id_entity = test[['id', 'entity']]
test_id_entity['entity'] = test_id_entity['entity'].apply(lambda index: index.split(';'))
ids, entity = [], [] 
for index in range(len(test_id_entity['entity'])):
    entity.extend(list(test_id_entity['entity'])[index])
    ids.extend([list(test_id_entity['id'])[index]] * len(list(test_id_entity['entity'])[index]))
test_id_entity = pd.DataFrame({'id': ids, 'entity_label': entity})  # test len:11580

# 
train.pop('negative') # 去掉negative列
train = train.merge(train_id_entity, on='id', how='left')
train['label'] = train.apply(lambda index: 0 if index.key_entity is np.nan else 1, axis=1)
train['key_entity'] = train['key_entity'].fillna('')
# train['label'] = train.apply(lambda index: 1 if index.key_entity.find(index.entity_label) != -1 else 0, axis=1)
train['label'] = train.apply(lambda index: 1 if index.key_entity.split(';').count(index.entity_label) >= 1 else 0, axis=1)

test = test.merge(test_id_entity, on='id', how='left')

# 去除长度小于的1的entity
train['entity_label_len'] = train['entity_label'].apply(lambda x: len(x))
test['entity_label_len'] = test['entity_label'].apply(lambda x: len(x))
train = train[train['entity_label_len']>1]
test = test[test['entity_label_len']>1]

def get_first_index(row, flag):
    if flag=='title':
        return row['title'].find(row['entity_label'])
    else:
        return row['text'].find(row['entity_label'])

train['title_first_index'] = train.apply(lambda row: get_first_index(row, 'title'), axis=1)
train['text_first_index'] = train.apply(lambda row: get_first_index(row, 'text'), axis=1)

test['title_first_index'] = test.apply(lambda row: get_first_index(row, 'title'), axis=1)
test['text_first_index'] = test.apply(lambda row: get_first_index(row, 'text'), axis=1)

def text_truncate(row):
    title_first_index = row['title_first_index']
    text_first_index = row['text_first_index']
    if title_first_index==-1 and text_first_index>480:
        return row['text'][text_first_index-200:text_first_index+300]
    else:
        return row['text']

train['text'] = train.apply(lambda row: text_truncate(row), axis=1)
test['text'] = test.apply(lambda row: text_truncate(row), axis=1)

def get_content(x, y, z):
    s='[E]' # E:Entity
    if str(x)!='nan':
        s+=x
    if str(y)!='nan' and str(z)!='nan' and  y==z:
        s+='[S]' # S:Same
        s+=y
    else:
        s+='[T]' # T:Title
        if str(y)!='nan':
            s+=y
        s+='[C]' # C:Content
        if str(z)!='nan':
            s+=z  
    #添加
#     if str(x)!='nan':
#         x_len=len(x)
#         end=len(s)-x_len
#         i=0
#         out=''
#         while i<=end:
#             if s[i:i+x_len]==x:
#                 out+='$'
#                 out+=x
#                 out+="$"
#                 i+=x_len
#             else:
#                 out+=s[i]
#                 i+=1
#         if i!=len(s):
#             out+=s[i:]
#         s=out
    return s 

# def get_content(x, y, z):
#     s=''      # E:Entity
# #     if str(x)!='nan':
# #         s+=x
#     if str(y)!='nan' and str(z)!='nan' and  y==z:
#         s+='[S]' # S:Same
#         s+=y
#     else:
#         s+='[T]' # T:Title
#         if str(y)!='nan':
#             s+=y
#         s+='[C]' # C:Content
#         if str(z)!='nan':
#             s+=z  
#     #添加
# #     if str(x)!='nan':
# #         x_len=len(x)
# #         end=len(s)-x_len
# #         i=0
# #         out=''
# #         while i<=end:
# #             if s[i:i+x_len]==x:
# #                 out+='$'
# #                 out+=x
# #                 out+="$"
# #                 i+=x_len
# #             else:
# #                 out+=s[i]
# #                 i+=1
# #         if i!=len(s):
# #             out+=s[i:]
# #         s=out
#     return s 

train['corpus']=list(map(lambda x,y,z: get_content(x,y,z),tqdm(train['entity_label'].values),train['title'],train['text']))
test['corpus']=list(map(lambda x,y,z: get_content(x,y,z),tqdm(test['entity_label'].values),test['title'],test['text']))

def get_other_content(x,y):
    entitys=x.split(';')
    if len(entitys)<=1:
        return np.nan
    l=[]
    for e in entitys:
        if e!=y and e!='':
            l.append(e)
    return ';'.join(l)
train['other_entity'] = list(map(lambda x, y: get_other_content(x, y), train['entity'], train['entity_label']))
test['other_entity'] = list(map(lambda x, y: get_other_content(x, y), test['entity'], test['entity_label']))

def get_content(x, y):
    if str(y) == 'nan':
        return x
    y = y.split(';')
    for i in y:
#         x=x.replace(i,'其他实体')
        x = 'O_E'.join(x.split(i)) # O_E:Other_Entity
    return x
train['corpus'] = list(map(lambda x, y: get_content(x, y), train['corpus'], train['other_entity']))
test['corpus'] = list(map(lambda x, y: get_content(x, y), test['corpus'], test['other_entity']))

MAXLEN = 510 # 510

bert_path = 'E:/NLP_corpus/BERT/hgd/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = bert_path + 'bert_config.json'
checkpoint_path = bert_path + 'bert_model.ckpt'
dict_path = bert_path + 'vocab.txt'

# 给每个token按序编号，构建词表
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 分词器
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')     # 剩余的字符是[UNK]
        return R
tokenizer = OurTokenizer(token_dict)

# Padding，默认添 0
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

# 数据生成
class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):    # 8
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size # 迭代完一个epoch需要的步数
        if len(self.data) % self.batch_size != 0:      # 保证步数为整数
            self.steps += 1
            
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:MAXLEN]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
#                 Y.append([y])
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y#[:, 0, :]
                    X1, X2, Y = [], [], []

# 计算：最高的k分类准确率
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# 计算：F1值
def f1_metric(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
                    
# BERT模型建立
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for layer in bert_model.layers:
#         print(l)
        layer.trainable = True

    # inputs
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
#     print('Bert output shape', x.shape)
    x = Lambda(lambda x: x[:, 0])(x)
    
    # outputs
    p = Dense(nclass, activation='softmax')(x)

    # 模型建立与编译
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(1e-5),
                  metrics=['accuracy', f1_metric, categorical_accuracy])
    print(model.summary())
    return model

from keras.callbacks import Callback
from sklearn.metrics import f1_score,accuracy_score

learning_rate = 5e-5
min_learning_rate = 1e-5

class Evaluate(Callback):
    def __init__(self):
        self.best = 0.
        self.passed = 0
        
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
            
# 训练集去重
train.drop_duplicates('corpus', inplace=True)

DATA_LIST = []
for data_row in train.iloc[:].itertuples():
    DATA_LIST.append((data_row.corpus, to_categorical(data_row.label, 2)))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.corpus, to_categorical(0, 2)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)

f1 = []
def run_cv(nfolds, data, data_label, data_test, epochs=10, date_str='1107'):
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=214683).split(data, train['label'])
    train_model_pred = np.zeros((len(data), 2))
    test_model_pred = np.zeros((len(data_test), 2))

    for i, (train_fold, test_fold) in enumerate(skf):
        print('Fold: ', i+1)
        
        '''数据部分'''
        # 数据划分
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=False)
        test_D = data_generator(data_test, shuffle=False)
        
        time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        '''模型部分'''
        # 生成模型
        model = build_bert(2)
        # callbacks
        early_stopping = EarlyStopping(monitor='val_f1_metric', patience=3)   # val_acc
        plateau = ReduceLROnPlateau(monitor="val_f1_metric", verbose=1, mode='max', factor=0.5, patience=1) # max：未上升则降速
        checkpoint = ModelCheckpoint('./models/keras_model/fusai' + date_str + str(i) + '.hdf5', monitor='val_f1_metric', 
                                     verbose=2, save_best_only=True, mode='max',save_weights_only=True) # period=1: 每1轮保存
        
        evaluator = Evaluate()
        
        # 模型训练，使用生成器方式训练
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),   ## ?? ##
            epochs=epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint, evaluator], # evaluator, 
            verbose=2
        )
        
        model.load_weights('./models/keras_model/fusai' + date_str + str(i) + '.hdf5')
        
        # return model
        val = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=0)
        print(val)
        
        score = f1_score(train['label'].values[test_fold], np.argmax(val, axis=1))
        global f1
        f1.append(score)
        print('validate {} f1_score:{}'.format(i+1, score))
        
        train_model_pred[test_fold, :] = val
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D),verbose=0)
        
        del model
        gc.collect()
        
        K.clear_session()
        # break
        
    return train_model_pred, test_model_pred


start_time = time.time()

n_folds = 10
folds_num = str(n_folds) + 'folds_' 
date_str = '1114'
strategy = '_withprocess_chusai&fusaidata_'
model = 'robeta_large'

train_model_pred, test_model_pred = run_cv(n_folds, DATA_LIST, None, DATA_LIST_TEST, date_str=date_str)
print('Validate 5folds average f1 score:', np.average(f1))
np.save('weights/keras_weight/fusai/train' + model + strategy + folds_num + date_str + '.npy', train_model_pred)
np.save('weights/keras_weight/fusai/test' + model + strategy + folds_num + date_str + '.npy', test_model_pred)

end_time = time.time()
print('Time cost(min): ', (end_time-start_time)/60)


def return_list(group):
    return ';'.join(list(group))

sub = test.copy()
sub['label'] = [np.argmax(index) for index in test_model_pred]
sub_label = sub[sub['label'] == 1].groupby(['id'], as_index=False)['entity_label'].agg({'key_entity': return_list})

test_2 = pd.read_csv('datasets/round2_test.csv', encoding='utf-8') # 导入测试集
submit = test_2[['id']]
submit = submit.merge(sub_label, on='id', how='left')
submit['negative'] = submit['key_entity'].apply(lambda index: 0 if index is np.nan else 1)
submit = submit[['id', 'negative', 'key_entity']]

time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(time_now)
submit.to_csv('submission/' + model + strategy + folds_num + '{}.csv'.format(time_now), encoding='utf-8', index=None)

