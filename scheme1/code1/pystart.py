# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:39:24 2019

@author: xiong
"""

import os
import pandas as pd
import numpy as np
import argparse

for i in range(10):
    params = '--model_type bert \
            --model_name_or_path chinese_roberta_wwm_large_ext \
            --do_train \
            --do_eval \
            --do_test \
            --data_dir %s \
            --output_dir %s \
            --max_seq_length 512 \
            --split_num 1 \
            --lstm_hidden_size 512 \
            --lstm_layers 1 \
            --lstm_dropout 0.1 \
            --eval_steps 1000 \
            --per_gpu_train_batch_size 16 \
            --gradient_accumulation_steps 8 \
            --warmup_steps 0 \
            --per_gpu_eval_batch_size 32 \
            --learning_rate 8e-6 \
            --adam_epsilon 1e-6 \
            --weight_decay 0 \
            --train_steps 40000 \
            --device_id %d' % ('./data/data_'+str(i), './model_roberta_wwm_large_ext'+str(i), 0)
    ex = os.system("python run_bert.py %s" %params)
    print('The fold:', i)

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default='./model_roberta_wwm_large_ext', type=str)
args = parser.parse_args()

k = 10
df = pd.read_csv('data/data_0/test.csv')
df['0'] = 0
df['1'] = 1
for i in range(k):
    temp = pd.read_csv('{}{}/test_pb.csv'.format(args.model_prefix, i))
    df['0'] += temp['label_0'] / k
    df['1'] += temp['label_1'] / k
print('The end for combining.')

df['pre_label'] = np.argmax(df[['0','1']].values, -1)
df['key_entity'] = np.nan
df.rename(columns={'pre_label':'negative'}, inplace=True)
df[['id','negative','key_entity']].to_csv('./result/submit_emotion.csv', encoding='utf-8', index=None)     #######right#######