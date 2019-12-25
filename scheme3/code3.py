#! -*- coding:utf-8 -*-
import os
import re
import gc
import sys
import json
import codecs
import warnings
import numpy as np
import pandas as pd
from keras import initializers
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
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_tqdm import TQDMNotebookCallback
tqdm.pandas()
np.random.seed(123)
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #指定gpu


data_path = '../data/'   #数据地址

train_path=data_path + 'Round2_train.csv'
train2_path=data_path + 'Train_data.csv'
test_path=data_path + 'round2_test.csv'

maxlen =  1   # 510
learning_rate = 5e-5
min_learning_rate = 1e-5
batch_size=20



save_model_path='zy'
save_mdoel_name_pre='large'

bert_path ='E:/code/pre_model/bert/chinese_L-12_H-768_A-12/'    #模型地址

config_path = bert_path + 'bert_config.json'
checkpoint_path = bert_path + 'bert_model.ckpt'
dict_path = bert_path + 'vocab.txt'


#################################################load data#########################################
train = pd.read_csv(train_path, encoding='utf-8')
train2= pd.read_csv(train2_path, encoding='utf-8')


train=pd.concat([train,train2],axis=0,sort=True)
test = pd.read_csv(test_path, encoding='utf-8')
test['text'] = test.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)

train = train[train['entity'].notnull()]
test = test[test['entity'].notnull()]

train=train.drop_duplicates(['title','text','entity','negative','key_entity'])   #去掉重复的data

print(train.shape)
print(test.shape)

#########################################删选实体####################################################################
'大家一样的函数处理'
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


#####################################################句子预处理####################################################
def delete_tag(s):
    r1 = re.compile(r'\{IMG:.?.?.?\}')  # 图片
    s = re.sub(r1, '', s)
    r2 = re.compile(r'[a-zA-Z]+://[^\u4e00-\u9fa5|\?]+')  # 网址
    s = re.sub(r2, '', s)
    r3 = re.compile(r'<.*?>')  # 网页标签
    s = re.sub(r3, '', s)
    r4 = re.compile(r'&[a-zA-Z0-9]{1,4}')  # &nbsp  &gt  &type &rdqu   ....
    s = re.sub(r4, '', s)
    r5 = re.compile(r'[0-9a-zA-Z]+@[0-9a-zA-Z]+')  # 邮箱
    s = re.sub(r5, '', s)
    r6 = re.compile(r'[#]')  # #号
    s = re.sub(r6, '', s)
    return s


train['title'] = train['title'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)
train['text'] = train['text'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)
test['title'] = test['title'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)
test['text'] = test['text'].apply(lambda x: delete_tag(x) if str(x) != 'nan' else x)


###############################################获取content#################################################
def get_or_content(y, z):
    s = ''
    if str(y) != 'nan' and str(z) != 'nan' and y == z:
        s += '标题和内容相同，是'
        s += y
    else:
        s += '标题是'
        if str(y) != 'nan':
            if len(y) > 172:
                s += y[:172]
            else:
                s += y
        else:
            s += '无'
        s += '内容是'
        if str(z) != 'nan':
            s += z
    return s


train['content'] = list(map(lambda y, z: get_or_content(y, z), train['title'], train['text']))
test['content'] = list(map(lambda y, z: get_or_content(y, z), test['title'], test['text']))


####################################################mark label##################################################
def get_id_entity(data):
    right_id, right_entity = [], []
    for row in data.itertuples():
        entities=row.entity.strip(';').split(';')
        entities.sort(key=lambda x:len(x))  #排序
        entities_num=len(entities)
        for index,entity in enumerate(entities):
            right_entity.append(entity)
            right_id.append(row.id)
    return pd.DataFrame({'id': right_id, 'entity_label': right_entity})

train_id_entity =get_id_entity(train[['id', 'entity','content']])
test_id_entity = get_id_entity(test[['id', 'entity','content']])

print('success')
# train.pop('negative')
train = train.merge(train_id_entity, on='id', how='left')
train['label'] = train.apply(lambda index: 0 if index.key_entity is np.nan else 1, axis=1)
train['key_entity'] = train['key_entity'].fillna('')
train['label'] = train.apply(lambda index: 1 if index.key_entity.split(';').count(index.entity_label) >= 1 else 0, axis=1)

test = test.merge(test_id_entity, on='id', how='left')

print(test.shape)
print(train.shape)

####################################获取预料############################################################

'tttttt 获取实体出现的位置 前510'


def get_new_content(entity, content):
    len_append = len('实体是' + entity)
    if len_append + len(content) > 510:
        return content[:510 - len_append]
    return content


def get_position(entity, corpus):
    tag = int(corpus.count(entity) > 0)
    # 为了消除空格的影响
    if tag > 0:
        return 1
    return int(corpus.count(entity.strip()) > 0)


train['content2'] = list(map(lambda x, y: get_new_content(x, y), train['entity'], train['content']))
test['content2'] = list(map(lambda x, y: get_new_content(x, y), test['entity'], test['content']))
train['entity_label_position2'] = list(map(lambda x, y: get_position(x, y), train['entity_label'], train['content2']))
test['entity_label_position2'] = list(map(lambda x, y: get_position(x, y), test['entity_label'], test['content2']))

train['entity_label_position'] = list(map(lambda x, y: get_position(x, y), train['entity_label'], train['content']))
test['entity_label_position'] = list(map(lambda x, y: get_position(x, y), test['entity_label'], test['content']))

train.pop('content')
train.pop('content2')

test.pop('content')
test.pop('content2')
print('success')

print(train['entity_label_position2'].value_counts())
print(train['entity_label_position'].value_counts())
print(test['entity_label_position2'].value_counts())
print(test['entity_label_position'].value_counts())

import re


# 前510没有内容的样本 corpus=i实体+标题+部分内容（无）
def get_context_content(x, y, z, p2):
    s = '实体是'
    if str(x) != 'nan':
        s += x
    s += '。标题和部分内容没有实体'
    if str(y) != 'nan' and str(z) != 'nan' and y == z:
        s += '，标题和内容相同，是'
        s += y
    else:
        s += '标题是'
        if str(y) != 'nan':
            if len(y) > 172:
                s += y[:172]
            else:
                s += y
        else:
            s += '无'
        # 有部分内容
        if p2 == 1:
            s += '。部分内容是'
            if str(z) != 'nan':
                z_list = re.split(r'，|。|？|；', z)
                for i in z_list:
                    if x in i:
                        s += i
                    if len(s) > 700:
                        break
            else:
                s += '无'
        else:
            s += '全文没有匹配的内容。内容是'
            if str(z) != 'nan':
                s += z
            else:
                s += '无'
    if len(s) > 700:
        return s[:700]
    return s


def get_content(x, y, z, position, p2):
    # 前510有内容
    if position == 1:
        s = '实体是'
        if str(x) != 'nan':
            s += x
        if str(y) != 'nan' and str(z) != 'nan' and y == z:
            s += '，标题和内容相同，是'
            s += y
        else:
            s += '标题是'
            if str(y) != 'nan':
                if len(y) > 172:
                    s += y[:172]
                else:
                    s += y
            else:
                s += '无'
            s += '内容是'
            if str(z) != 'nan':
                s += z
            else:
                s += '无'
    else:
        # 前510没有内容
        s = get_context_content(x, y, z, p2)
    if len(s) > 700:
        return s[:700]
    return s


train['corpus'] = list(
    map(lambda x, y, z, position, p2: get_content(x, y, z, position, p2), tqdm(train['entity_label'].values),
        train['title'], train[
            'text'], train['entity_label_position2'], train['entity_label_position']))
test['corpus'] = list(
    map(lambda x, y, z, position, p2: get_content(x, y, z, position, p2), tqdm(test['entity_label'].values),
        test['title'], test[
            'text'], test['entity_label_position2'], test['entity_label_position']))

def get_position(entity,corpus):
    tag=int(corpus.count(entity)>1)
    #为了消除空格的影响
    if tag>1:
        return 1
    return int(corpus.count(entity.strip())>1)

train['entity_label_position3']=list(map(lambda x,y:get_position(x,y),train['entity_label'],train['corpus']))
test['entity_label_position3']=list(map(lambda x,y:get_position(x,y),test['entity_label'],test['corpus']))


print(train['entity_label_position3'].value_counts())
print(test['entity_label_position3'].value_counts())



def get_other_content(x,y):
    entitys=x.strip(';').split(";")
    if len(entitys)<=1:
        return  np.nan
    l=[]
    for e in entitys:
        if e!=y:
            l.append(e)
    return ';'.join(l)
train['other_entity']=list(map(lambda x,y :get_other_content(x,y),train['entity'],train['entity_label']))
test['other_entity']=list(map(lambda x,y :get_other_content(x,y),test['entity'],test['entity_label']))
def get_content(x,y,z):
    if str(y)=='nan':
        return x
    y=y.split(";")
    y=sorted(y,key=lambda x: len(x),reverse=True)
    #如果该实体没有出现直接返回原始语句
    if x.count(z)<=1:
        return x
    #出现了就直接替换其他的实体
    for i in y:
        if i not in z:               #不是该候选实体的子串
            x='其他实体'.join(x.split(i))
    return x
train['corpus']=list(map(lambda x,y,z :get_content(x,y,z),train['corpus'],train['other_entity'],train['entity_label']))
test['corpus']=list(map(lambda x,y ,z:get_content(x,y,z),test['corpus'],test['other_entity'],test['entity_label']))



from keras.utils import to_categorical

DATA_LIST = []
for data_row in train.iloc[:].itertuples():
    DATA_LIST.append((data_row.corpus, data_row.negative,data_row.label))
DATA_LIST = np.array(DATA_LIST)
print(DATA_LIST.shape)

DATA_LIST_TEST = []
for data_row in test.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.corpus, 0,0))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)
print(DATA_LIST_TEST.shape)


########################################模型部分#############################################

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
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

'填充序列长度'


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    def __init__(self, data, batch_size=batch_size, shuffle=True):  # 8
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
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

            X1, X2, Y1, Y2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y1 = d[1]
                y2 = d[2]
                X1.append(x1)
                X2.append(x2)
                Y1.append([y1])
                Y2.append([y2])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y1 = seq_padding(Y1)
                    Y2 = seq_padding(Y2)
                    yield [X1, X2, Y1, Y2], None
                    X1, X2, Y1, Y2 = [], [], [], []


from keras.metrics import top_k_categorical_accuracy
from keras.metrics import categorical_accuracy


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# from keras.utils import multi_gpu_model


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1],),
                                 initializer=initializers.glorot_uniform(seed=2019), trainable=True,
                                 name='kernel')
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        soft_w = K.softmax(self.w)

        return x * soft_w

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


# def get_means_bert_n_layers(x):
#     shape = list(x.shape)
#     layer_size = 768
#     number = int(int(shape[1]) / layer_size)
#     res = x[:, :layer_size]
#     for i in range(2, number + 1):
#         res += x[:, layer_size * (i - 1):layer_size * i]
#     print('return ')
#     return tf.divide(res, number)

#
# def get_means_bert_n_layers_shape(input_shape):
#     shape = list(input_shape)
#     layer_size = 768
#     shape[-1] = layer_size
#     return tuple(shape)


def build_bert(nclass):
    seed = 2019
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, output_layer_num=1, seq_len=None)

    layer_ = 1
    for l in bert_model.layers:
        if layer_ >= 105 - 105:
            l.trainable = True
        #         print(l)
        layer_ += 1
    print(' layer:', layer_)
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    y1_in = Input(shape=(None,))
    y2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])

    x0 = Lambda(lambda x: x[:, 0])(x)

    #     x0=Lambda(get_means_bert_n_layers,get_means_bert_n_layers_shape)(x0)

    x1 = Dense(300, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=seed))(x0)
    p1 = Dense(nclass, activation='sigmoid')(x1)

    x1_st = Lambda(lambda x: K.stop_gradient(x))(x1)
    #     x1_st=MyLayer()(x1_st)
    p1_st = Lambda(lambda x: K.stop_gradient(x))(p1)

    x2 = concatenate([x0, x1_st, p1_st])

    x2 = Dense(300, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=seed))(x2)
    p2 = Dense(nclass, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=seed))(x2)

    model = Model([x1_in, x2_in], [p1, p2])

    train_model = Model([x1_in, x2_in, y1_in, y2_in], [p1, p2])

    loss1 = K.mean(K.binary_crossentropy(y1_in, p1, from_logits=False))
    loss2 = K.mean(K.binary_crossentropy(y2_in, p2, from_logits=False))

    # 带权重的loss
    #     w_loss=K.softmax(K.variable([0.5,0.5]))

    #     loss=w_loss[0]*loss1+w_loss[1]*loss2
    loss = loss1 + loss2

    train_model.add_loss(loss)


    train_model.compile(optimizer=Adam(learning_rate))
    # print(train_model.summary())
    return train_model


from sklearn import metrics

'寻找最好的阀值'
def get_best_F1_score(pred, ture):
    f1_scores = []
    cut_offs = []
    for threshold in np.arange(0.01, 1, 0.01):
        pred_binary = (pred >= threshold) * 1
        f1_tmp = metrics.f1_score(y_true=ture, y_pred=pred_binary)
        f1_scores.append(f1_tmp)
        cut_offs.append(threshold)
    max_index = f1_scores.index(max(f1_scores))
    max_x_axis = cut_offs[max_index]
    max_y_axis_F1 = f1_scores[max_index]
    return max_x_axis, max_y_axis_F1


from keras.callbacks import Callback


class Evaluate(Callback):
    def __init__(self, X_train, X_valid, tag):
        self.X_valid = X_valid
        self.X_train = X_train
        self.Y1_train, self.Y2_train = [int(i) for i in X_train[:, 1]], [int(i) for i in X_train[:, 2]]
        self.Y1_valid, self.Y2_valid = [int(i) for i in X_valid[:, 1]], [int(i) for i in X_valid[:, 2]]
        self.tag = tag
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

    def on_epoch_end(self, epoch, logs=None):
        #         train_D = data_generator(self.X_train, shuffle=False)
        valid_D = data_generator(self.X_valid, shuffle=False)
        #         tra_1,tra_2 = self.model.predict_generator(train_D.__iter__(), steps=len(train_D),verbose=0)
        val_1, val_2 = self.model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=0)
        #         tra_1=[int(i>0.5) for i in tra_1]
        #         tra_2=[int(i>0.5) for i in tra_2]
        val_1 = [int(i > 0.5) for i in val_1]
        val_2 = [int(i > 0.5) for i in val_2]
        # train
        #         t_f1_1=f1_score(self.Y1_train,tra_1)
        #         t_f1_2=f1_score(self.Y2_train,tra_2)
        #         t_acc_1=accuracy_score(self.Y1_train,tra_1)
        #         t_acc_2=accuracy_score(self.Y2_train,tra_2)
        # val
        f1_1 = f1_score(self.Y1_valid, val_1)
        f1_2 = f1_score(self.Y2_valid, val_2)
        # acc_1 = accuracy_score(self.Y1_valid, val_1)
        # acc_2 = accuracy_score(self.Y2_valid, val_2)

        print(' ----val -f1_1:{:.5f} -f1_2:{:.5f}'.format(f1_1, f1_2))
        #         print('train -acc_1:{} -acc_2:{} ----val -acc_1:{} -acc_2:{}'.format(t_acc_1,t_acc_2,acc_1,acc_2))

        f_mean = f1_1 * 0.4 + f1_2 * 0.6
        if f_mean >= self.best:
            self.best = f_mean
            self.model.save_weights(save_model_path + save_mdoel_name_pre + '{}.hdf5'.format(self.tag))
            print('f_mean : {:.5f} best f_mean :{:.5f}'.format(f_mean, self.best))



#########################################训练部分########################################

def run_cv(nfold, data, data_label, data_test):
    f1_1_list = []
    f1_2_list = []

    val_best_threshold1 = []  # 预测1
    val_best_threshold2 = []  # 预测2
    kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019).split(data, train['entity_label_position3'])
    train_model_pred1 = np.zeros((len(data), 1))
    train_model_pred2 = np.zeros((len(data), 1))
    test_model_preds1 = np.zeros((len(data_test), nfold))
    test_model_preds2 = np.zeros((len(data_test), nfold))

    for i, (train_fold, test_fold) in enumerate(kf):
        #         if i<2:
        #             continue
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        print(X_train.shape)
        print(X_valid.shape)

        print(train_fold[:20])
        print(test_fold[:20])

        print('*' * 50, i + 1, '*' * 50)
        model = build_bert(1)
        #         early_stopping = EarlyStopping(monitor='val_loss', patience=3)   # val_acc
        #         plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=1)
        #         checkpoint = ModelCheckpoint('./model/' + str(i) + '.hdf5', monitor='val_loss',
        #                                          verbose=2, save_best_only=True, mode='min',save_weights_only=True)

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=False)
        test_D = data_generator(data_test, shuffle=False)

        evaluator = Evaluate(X_train, X_valid, i + 1)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            epochs=1,
            callbacks=[evaluator],
            verbose=2
        )

        model.load_weights(save_model_path + save_mdoel_name_pre + '{}.hdf5'.format(i + 1))

        # return model
        val_1, val_2 = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=0)

        # 保存预测结果
        train_model_pred1[test_fold, :] = val_1
        train_model_pred2[test_fold, :] = val_2

        val_y1 = [int(i) for i in X_valid[:, 1]]
        val_y2 = [int(i) for i in X_valid[:, 2]]
        best_threshold1, f1_1 = get_best_F1_score(val_1, val_y1)
        best_threshold2, f1_2 = get_best_F1_score(val_2, val_y2)

        val_best_threshold1.append(best_threshold1)
        val_best_threshold2.append(best_threshold2)

        f1_1_list.append(f1_1)
        f1_2_list.append(f1_2)

        print('validate  score -f1:{:.5f}  -val_best_threshold1:{:.5f} -f2{:.5f} -val_best_threshold2:{:.5f}'.format(
            f1_1_list[-1], val_best_threshold1[-1], f1_2_list[-1], val_best_threshold2[-1]))

        # 预测
        t1, t2 = model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=0)
        for j in range(len(t1)):
            test_model_preds1[j, i] = t1[j]
            test_model_preds2[j, i] = t2[j]

        del model;
        gc.collect()
        K.clear_session()
        print('*' * 50, i + 1, '-done-', '*' * 45)

    return train_model_pred1, train_model_pred2, test_model_preds1, test_model_preds2, val_best_threshold1, val_best_threshold2

train_model_pred1,train_model_pred2, test_pred_list1,test_pred_list2,val_best_threshold1,val_best_threshold2 = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)

######################################################预测部分################################################
'best train f1 and threshold'
train_best_threshold1,train_f1=get_best_F1_score(train_model_pred1,train['negative'].values)
print('train best_f1_1:',train_f1)
print('train best_threshold_1:',train_best_threshold1)

'best train f1 and threshold'
train_best_threshold2,train_f2=get_best_F1_score(train_model_pred2,train['label'].values)
print('train best_f1_2:',train_f2)
print('train best_threshold_2:',train_best_threshold2)

from sklearn.metrics import f1_score, accuracy_score

'获取5折的分数'
pred1 = [int(index[0] > 0.5) for index in train_model_pred1]
pred2 = [int(index[0] > 0.5) for index in train_model_pred2]
true1 = [int(index) for index in DATA_LIST[:, 1]]
true2 = [int(index) for index in DATA_LIST[:, 2]]

print('f1_1:{}'.format(f1_score(true1, pred1)))
print('f1_2:{}'.format(f1_score(true2, pred2)))

print('acc_1:{}'.format(accuracy_score(true1, pred1)))
print('acc_2:{}'.format(accuracy_score(true2, pred2)))

# 保存概率
train['pred1'] = train_model_pred1[:, 0]
train['pred2'] = train_model_pred2[:, 0]
train.to_csv('{}_train_preds.csv'.format(save_mdoel_name_pre), index=False)

# 线下模拟线上分数
train['preb1'] = train_model_pred1[:, 0]
train['preb2'] = train_model_pred2[:, 0]
df = train[['id', 'negative', 'label', 'preb1', 'preb2', 'entity_label']]
df['n_pred'] = df['preb1'].apply(lambda x: int(x > 0.5))
df['l_pred'] = df['preb2'].apply(lambda x: int(x > 0.5))

train_or = pd.read_csv(train_path, encoding='utf-8')
train2 = pd.read_csv(train2_path, encoding='utf-8')
train_or = pd.concat([train_or, train2], axis=0, sort=True)
train_or = train_or[train_or['entity'].notnull()]

# pred1
temp = df[['id', 'n_pred']].groupby('id')['n_pred'].agg(lambda x: list(x))
train_or['n_pred_list'] = train_or['id'].map(temp)
train_or = train_or[train_or['n_pred_list'].notnull()]

# entity label list
temp = df[['id', 'entity_label']].groupby('id')['entity_label'].agg(lambda x: list(x))
train_or['entity_label_list'] = train_or['id'].map(temp)

# pred2
temp = df[['id', 'l_pred']].groupby('id')['l_pred'].agg(lambda x: list(x))
train_or['l_pred_list'] = train_or['id'].map(temp)

train_or['n_pred_type'] = train_or['n_pred_list'].apply(lambda x: len(set(x)))


def get_negative_and_entity(n_pred_list, l_pred_list, entitys):
    assert len(entitys) == len(n_pred_list)
    sub = []
    neg = 0
    for i in range(len(n_pred_list)):
        if n_pred_list[i] == 1 and l_pred_list[i] == 1:
            sub.append(entitys[i])
            neg = 1
    if len(sub) == 0:
        return np.nan
    return ';'.join(sub)


train_or['preb_entity'] = list(map(lambda x, y, z: get_negative_and_entity(x, y, z),
                                   train_or['n_pred_list'], train_or['l_pred_list'], train_or['entity_label_list']))

train_or['pred_label'] = train_or['preb_entity'].apply(lambda x: 1 if str(x) != 'nan' else 0)


def get_f1(y1, y2, y1_pred, y2_pred):
    F1_1 = f1_score(y1, y1_pred)
    print('f1_1:{}'.format(F1_1))

    '计算实体F1'
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(y2)):
        if str(y2[i]) == 'nan':
            y2_i = set()
        else:
            y2_i = set(y2[i].split(';'))
        if str(y2_pred[i]) == 'nan':
            y2_pred_i = set()
        else:
            y2_pred_i = set(y2_pred[i].split(';'))
        TPi = len(y2_i & y2_pred_i)
        FNi = len(y2_i.difference(y2_pred_i))
        FPi = len(y2_pred_i.difference(y2_i))
        TP += TPi
        FN += FNi
        FP += FPi
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1_2 = 2 * P * R / (P + R)
    print('f1_2:{}'.format(F1_2))
    print('score：', 0.4 * F1_1 + 0.6 * F1_2)


print(get_f1(train_or['negative'].values, train_or['key_entity'].values, train_or['pred_label'].values,
       train_or['preb_entity'].values))


pred1_df=np.mean(test_pred_list1,axis=1)
pred2_df=np.mean(test_pred_list2,axis=1)

sub=test.copy()
sub['preb1']=list(pred1_df)
sub['preb2']=list(pred2_df)


df=sub[['id','preb1','preb2','entity_label']]
df['n_pred']=df['preb1'].apply(lambda x: int(x>0.5))
df['l_pred']=df['preb2'].apply(lambda x: int(x>0.5))


sub.to_csv('{}_test_preds.csv'.format(save_mdoel_name_pre),index=False)


test_or = pd.read_csv(test_path, encoding='utf-8')
test_or= test_or[test_or['entity'].notnull()]

#pred1
temp=df[['id','n_pred']].groupby('id')['n_pred'].agg(lambda x:list(x))
test_or['n_pred_list']=test_or['id'].map(temp)

#pred2
temp=df[['id','l_pred']].groupby('id')['l_pred'].agg(lambda x:list(x))
test_or['l_pred_list']=test_or['id'].map(temp)

#entitys
temp=df[['id','entity_label']].groupby('id')['entity_label'].agg(lambda x:list(x))
test_or['entity_label_list']=test_or['id'].map(temp)

#key_entity
test_or['preb_entity']=list(map(lambda x,y,z:get_negative_and_entity(x,y,z),
                             test_or['n_pred_list'],test_or['l_pred_list'],test_or['entity_label_list']))


test_or['pred_label']=test_or['preb_entity'].apply(lambda x: 1 if str(x)!='nan' else 0)

test_or=test_or[['id','pred_label','preb_entity']]

test_2 = pd.read_csv(test_path, encoding='utf-8')
submit = test_2[['id']]
submit = submit.merge(test_or, on='id', how='left')
submit.columns=['id', 'negative', 'key_entity']
submit['negative']=submit['negative'].apply(lambda x: int(x) if str(x)!='nan' else 0)
submit['negative']=submit['negative'].astype('int')
# submit['key_entity']=np.nan
submit.to_csv('{}_result_mean.csv'.format(save_mdoel_name_pre),index=False)
print(submit.isnull().sum())

