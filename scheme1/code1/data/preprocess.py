import pandas as pd
import os
import random
import numpy as np
import re
import textdistance

train_df = pd.read_csv('./data/Round2_train.csv')
test_data = pd.read_csv('./data/round2_test.csv')

train_df['text'] = train_df.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)
test_data['text'] = test_data.apply(lambda index: index.title if index.text is np.nan else index.text, axis=1)

def entity_clear(df):
    for index, row in df.iterrows():
        if type(row.entity) == float or type(row.text) == float:
            continue
        entities = row.entity.split(';')
        entities.sort(key = lambda x : len(x))
        n = len(entities)
        tmp = entities.copy()
        for i in range(n):
            entity_tmp = entities[i]
            if i + 1 >= n:
                break
            for entity_tmp2 in entities[i+1:]:
                if entity_tmp2.find(entity_tmp) != -1 and row.text.replace(entity_tmp2, '').find(entity_tmp) == -1:
                    tmp.remove(entity_tmp)
                    break
        df.loc[index, 'entity'] = ';'.join(tmp)
    return df
train_data = entity_clear(train_df)
test_df = entity_clear(test_data)

train_data.dropna(subset = ['entity'], inplace=True)
train_data.reset_index(drop=True, inplace=True)
test_df.dropna(subset = ['entity'], inplace=True)
test_df.reset_index(drop=True, inplace=True)
test_df['negative'] = 0
train_data['title'] = train_data['title'].fillna('无')
train_data['text'] = train_data['text'].fillna('无')
test_df['title'] = test_df['title'].fillna('无')
test_df['text'] = test_df['text'].fillna('无')

train_data['text'] = train_data['text'].map(lambda index: re.sub(r'http.*$', "", index))
test_df['text'] = test_df['text'].map(lambda index: re.sub(r'http.*$', "", index))

train_data['title'] = train_data['title'].map(lambda index: index.replace(' ', ''))
train_data['text'] = train_data['text'].map(lambda index: index.replace(' ', ''))
train_data['title_len'] = train_data['title'].map(lambda index:len(index))

test_df['title'] = test_df['title'].map(lambda index: index.replace(' ', ''))
test_df['text'] = test_df['text'].map(lambda index: index.replace(' ', ''))
test_df['title_len'] = test_df['title'].map(lambda index:len(index))

distance = textdistance.Levenshtein(external = False)
train_data['distance'] = train_data.apply(lambda index: distance(index.title, index.text), axis=1)
test_df['distance'] = test_df.apply(lambda index: distance(index.title, index.text), axis=1)

train_data['title_in_text'] = train_data.apply(lambda index: 1 if index.text.find(index.title) != -1 else 0, axis=1)
test_df['title_in_text'] = test_df.apply(lambda index: 1 if index.text.find(index.title) != -1 else 0, axis=1)

train_data['content'] = train_data.apply(lambda index: index.title + index.text if (index.title_len != 0) & (index.title_in_text != 1) & (index.distance > 100) else index.text, axis=1)
test_df['content'] = test_df.apply(lambda index: index.title + index.text if (index.title_len != 0) & (index.title_in_text != 1) & (index.distance > 100) else index.text, axis=1)
    
def get_content(x ,y):
    try:
        if str(y) == 'nan':
            return x
        y = y.split(';')
        y = sorted(y, key=lambda i: len(i), reverse=True)
        for i in y:
            x = '实体词'.join(x.split(i))
        return x
    except:
        return x
train_data['content'] = list(map(lambda x,y: get_content(x,y), train_data['content'], train_data['entity']))
test_df['content'] = list(map(lambda x,y: get_content(x,y), test_df['content'], test_df['entity']))

train_data.rename(columns={'negative':'label'}, inplace=True)
test_df.rename(columns={'negative':'label'}, inplace=True)

features = ['id', 'content' ,'entity', 'label']
index = set(range(train_data.shape[0]))

K_fold = []
for i in range(10):
    if i == 9:
        tmp = index
    else:
        tmp = random.sample(index, int(1.0 /10 * train_data.shape[0]))
    index = index - set(tmp)
    print('number:', len(tmp))
    K_fold.append(tmp)

for i in range(10):
    print('Fold', i)
    os.system('mkdir ./data/data_{}'.format(i))
    dev_index = list(K_fold[i])
    train_index = []
    for j in range(10):
        if j != i:
            train_index += K_fold[j]
    train_data[features].iloc[train_index].to_csv('./data/data_{}/train.csv'.format(i))
    train_data[features].iloc[dev_index].to_csv('./data/data_{}/dev.csv'.format(i))
    test_df[features].to_csv('./data/data_{}/test.csv'.format(i))
    