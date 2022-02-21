import os
from torch import cuda
import numpy as np, os 
import pandas as pd, gc 
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score

RUN_KAGGLE = False
BASE_PATH = '../input' if RUN_KAGGLE else '../data'


# DECLARE HOW MANY GPUS YOU WISH TO USE. 
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
HOME = os.path.join(BASE_PATH, 'py-bigbird-v26')
os.makedirs(HOME, exist_ok=True)

# VERSION FOR SAVING MODEL WEIGHTS
VER=26

TRAIN_FILE = 'train_NER.csv'

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True
if len(os.listdir(os.path.join(BASE_PATH, 'feedback-prize-2021/test'))) > 5:
      COMPUTE_VAL_SCORE = False

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = os.path.join(BASE_PATH, 'py-bigbird-v26')

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'feedback-prize-2021/train.csv'))
    print(train_df.shape)
    train_df.head()

    # TEST and TRAIN DATA

    # test_names, test_texts = [], []
    # for f in list(os.listdir(os.path.join(BASE_PATH, 'feedback-prize-2021/test'))):
    #     test_names.append(f.replace('.txt', ''))
    #     test_texts.append(open(os.path.join(BASE_PATH, 'feedback-prize-2021/test', f), 'r').read())
    # test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
    # test_texts.head()

    test_names, train_texts = [], []
    for f in tqdm(list(os.listdir(os.path.join(BASE_PATH, 'feedback-prize-2021/train')))):
        test_names.append(f.replace('.txt', ''))
        train_texts.append(open(os.path.join(BASE_PATH, 'feedback-prize-2021/train', f), 'r').read())
    train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
    train_text_df.head()

    # Convert Train Text to NER Labels

    if not os.path.isfile(os.path.join(LOAD_TOKENS_FROM, TRAIN_FILE)):
        print(f"File {TRAIN_FILE} not found in {LOAD_TOKENS_FROM}. Creating it and saving it")
        all_entities = []
        for ii,i in enumerate(train_text_df.iterrows()):
            if ii%100==0: print(ii,', ',end='')
            total = i[1]['text'].split().__len__()
            entities = ["O"]*total
            for j in train_df[train_df['id'] == i[1]['id']].iterrows():
                discourse = j[1]['discourse_type']
                list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]: entities[k] = f"I-{discourse}"
            all_entities.append(entities)
        train_text_df['entities'] = all_entities
        train_text_df.to_csv(os.path.join(LOAD_TOKENS_FROM, TRAIN_FILE),index=False)
        
    print(train_text_df.shape)
    train_text_df.head()