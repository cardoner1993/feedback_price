from datetime import datetime
import os

RUN_KAGGLE = False
BASE_PATH = '../input' if RUN_KAGGLE else '../data'
HF_HOME = os.path.join(BASE_PATH, 'py-bigbird-v26', "models")
os.makedirs(HF_HOME, exist_ok=True)

DEBUG = False
EXPERIMENT_NAME = "pytorch_longformer"
TRAIN_MODEL = False
COMPUTE_VAL_SCORE = True
BUILD_SUBMISION = False

if COMPUTE_VAL_SCORE or BUILD_SUBMISION:
    os.environ["CUDA_VISIBLE_DEVICES"]="7" #0,1,2,3 for four gpu


# os.environ['TRANSFORMERS_CACHE'] = HF_HOME
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from tqdm import tqdm
from ast import literal_eval

import numpy as np
import pandas as pd

import json

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    MultiStepLR, 
    ReduceLROnPlateau
    )

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import torchmetrics

from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup,  AutoModelForTokenClassification, get_cosine_schedule_with_warmup
from torch.optim import Adam, AdamW

from sklearn.model_selection import KFold, train_test_split

from data_preparation import TRAIN_FILE, LOAD_TOKENS_FROM
from config import config
from inference import get_predictions, score_feedback_comp

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = None
# LOAD_MODEL_FROM = os.path.join(BASE_PATH, 'py-bigbird-v26')

# IF FOLLOWING IS NONE, THEN NOTEBOOK 
# USES INTERNET AND DOWNLOADS HUGGINGFACE 
# CONFIG, TOKENIZER, AND MODEL
# DOWNLOADED_MODEL_PATH = os.path.join(BASE_PATH, 'py-bigbird-v26') 

# if DOWNLOADED_MODEL_PATH is None:
#     DOWNLOADED_MODEL_PATH = 'model'    
MODEL_NAME = 'allenai/longformer-base-4096'
LOAD_CONFIG = None # None or path to config
LABEL_ALL_SUBTOKENS = True
FOLDS = 1
BATCH_SIZE = 4 # Same for train and valid


train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4,
                'pin_memory':True
                }

test_params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'num_workers': 4,
                'pin_memory':True
                }

def save_config(dict_conf, conf_path, conf_name='config_experiment.json'):
    # Function to save the config for each experiment.
    with open(os.path.join(conf_path, conf_name), 'w') as f:
        json.dump(dict_conf, f)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


# =============================
# Model
# =============================
def get_optimizer(cfg, parameters):
    opt = cfg['optimizer']
    if opt["optimizer"] == "AdamW":
        optimizer = AdamW(
            parameters,
            lr=opt["lr"],
            weight_decay=opt["weight_decay"]
            )
    
    elif opt["optimizer"] == "Adam":
        optimizer = Adam(
            parameters,
            lr=opt["lr"],
            weight_decay=opt["weight_decay"]
            )
    
    else:
        raise NotImplementedError
    
    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    sch = cfg['scheduler']
    if sch["scheduler"] == "get_linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=sch["num_warmup_steps"],
            num_training_steps=num_train_steps)
    
    elif sch["scheduler"] == "get_cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=sch["num_warmup_steps"],
            num_training_steps=num_train_steps,
            num_cycles=sch["num_cycles"]
            )

    elif sch["scheduler"] == "MultiStepLR":
        scheduler = MultiStepLR(
            optimizer, 
            milestones=sch["milestones"], 
            gamma=sch["gamma"]
        )

    else:
        raise NotImplementedError
    
    return scheduler


class LightningFeedBack(LightningModule):
    def __init__(self, model_name, cfg, num_classes=None):
        super(LightningFeedBack, self).__init__()

        self.cfg = cfg
        
        if not os.path.isfile(os.path.join(HF_HOME, 'pytorch_model.bin')):
            config_model = AutoConfig.from_pretrained(model_name) 
            config_model.num_labels = num_classes
            config_model.save_pretrained(HF_HOME)

            self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=config_model)
            self.model.save_pretrained(HF_HOME)
        else:
            config_model = AutoConfig.from_pretrained(os.path.join(HF_HOME, 'config.json')) 
            self.model = AutoModelForTokenClassification.from_pretrained(os.path.join(HF_HOME, 'pytorch_model.bin'), config=config_model)
            
        
        # Warm up wait n epochs
        self.warmup_steps = 0
        # threshold (float) – Threshold for transforming probability or logit predictions to binary (0,1) 
        self.metric = torchmetrics.F1(num_classes=num_classes)
    
    def setup(self, stage=None):
        if stage != "fit":
            return
        # calculate total steps
        train_dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
        gpus = 0 if self.trainer.gpus is None else self.trainer.gpus if isinstance(self.trainer.gpus, int) else len(self.trainer.gpus)
        tb_size = self.cfg['train_batch_size'] * max(1, gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_dataloader.dataset) // tb_size) // ab_size

    def forward(self, ids, mask, b_labels=None):
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=b_labels, return_dict=False)
        return outputs

    def on_train_batch_start(self, batch, batch_idx):
        # Logs learning
        self.log('learning_rate', self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0], prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_labels = batch['labels']
        z = self(b_input_ids, b_input_mask, b_labels)

        loss, logits = z[0], z[1]

        flattened_targets = b_labels.view(-1)
        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = b_labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.log('train_f1_score', self.metric(labels.cpu(), predictions.cpu()).item(), prog_bar=True)

        self.log('train_loss', loss.item(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_labels = batch['labels']
        z = self(b_input_ids, b_input_mask, b_labels)
        
        val_loss, logits = z[0], z[1]

        flattened_targets = b_labels.view(-1)
        active_logits = logits.view(-1, self.model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = b_labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.log('val_f1_score', self.metric(labels.cpu(), predictions.cpu()).item(), prog_bar=True)

        self.log('val_loss', val_loss.item(), prog_bar=True)

        return val_loss

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=0.01, eps=1e-8)

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.warmup_steps,
    #         num_training_steps=len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs,
    #     )

    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg, parameters=self.parameters())

        if self.cfg['scheduler'] is None:
            return [optimizer]
        else:
            scheduler = get_scheduler(self.cfg, optimizer, num_train_steps=self.total_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": self.cfg['scheduler']["interval"]}]


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_wids, cfg):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation

        self.output_labels = cfg['output_labels']

        self.labels_to_ids = cfg['labels_to_ids']
        self.ids_to_labels = cfg['ids_to_labels']


  def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                             is_split_into_words=True,
                             #return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:              
                    label_ids.append(self.labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(self.labels_to_ids[word_labels[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)
        
        return item

  def __len__(self):
        return self.len


class FeedbackDataNLP(LightningDataModule):
    def __init__(self, fold, batch_size, tokenizer, max_length, train_df, valid_df, cfg):
        super().__init__()
        self.fold = fold
        self.batch_size = batch_size
        self.max_len = max_length
        self.tokenizer = tokenizer

        self.train_df = train_df
        self.valid_df = valid_df

        # self.data_path = data_path
        # self.train_inputs = None
        # self.validation_inputs = None
        # self.train_labels = None
        # self.validation_labels = None
        # self.train_masks = None
        # self.validation_masks = None
        # self.num_labels = None

        self.training_set = None
        self.valid_set = None
        self.test_set = None

        self.output_labels = cfg['output_labels']

        self.num_labels = len(self.output_labels)

        self.labels_to_ids = cfg['labels_to_ids']
        self.ids_to_labels = cfg['ids_to_labels']

        # self.train_data = self.prepare_dataset()
    
    def prepare_dataset(self):
        data = pd.read_csv(os.path.join(LOAD_TOKENS_FROM, TRAIN_FILE))
        
        if self.debugging:
            data = data.sample(n=100, random_state=1)
        
        return data

    def prepare_train_valid(self, folds, ids):

        np.random.seed(42)

        if folds <= 1:
            IDS = ids.unique()
            print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')
            train_index = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False) 
            validation_index = np.setdiff1d(np.arange(len(IDS)),train_index)
        else:
            kf = KFold(FOLDS, shuffle=True, random_state=42)

            # Check it out. The code implements kf cross validation in this way ?¿
            for fold, (tr_idx, val_idx) in enumerate(kf.split(ids)):
                train_index = tr_idx
                validation_index = val_idx
                
                if fold == self.fold:
                    break

        return train_index, validation_index
    
    def align_tokens(self, word_ids, word_labels, get_wids):
        # CREATE TARGETS
        if not get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:              
                    label_ids.append(self.labels_to_ids[word_labels[word_idx]] )
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(self.labels_to_ids[word_labels[word_idx]] )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
        else:
            label_ids = [w if w is not None else -1 for w in word_ids]
        
        return label_ids

    def encode(self, text):
        encoding = self.tokenizer(text,
                             is_split_into_words=True,
                             #return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        word_ids = encoding.word_ids()  

        return encoding, word_ids
    
    def encode_and_align(self, texts, labels, get_wids):
        inputs, masks, label_list = list(), list(), list()
        for text, label in tqdm(zip(texts, labels), total=len(texts)):
            encoding, word_ids = self.encode(text)
            label_ids = self.align_tokens(word_ids, label, get_wids)
            inputs.append(encoding['input_ids'])
            masks.append(encoding['attention_mask'])
            label_list.append(label_ids)

        return inputs, masks, label_list

    def prepare_data(self):
        self.training_set = dataset(self.train_df, tokenizer, self.max_len, False, cfg=cfg)
        self.valid_set = dataset(self.valid_df, tokenizer, self.max_len, False,  cfg=cfg)
        self.test_set = dataset(self.valid_df, tokenizer, self.max_len, True,  cfg=cfg)
        

    # def setup(self, stage=None):

    #     self.train_data['text'] = self.train_data['text']

    #     self.train_data['text'] = self.train_data.text.str.split()
    #     self.train_data['entities'] = self.train_data.entities.apply(lambda x: literal_eval(x))

    #     IDS = self.train_data.id.unique()
    #     train_idx, valid_idx = self.prepare_train_valid(FOLDS, self.train_data.id)

    #     train_text, train_entities =  self.train_data['text'][self.train_data['id'].isin(IDS[train_idx])], self.train_data['entities'][self.train_data['id'].isin(IDS[train_idx])]
    #     valid_text, valid_entities = self.train_data['text'][self.train_data['id'].isin(IDS[valid_idx])], self.train_data['entities'][self.train_data['id'].isin(IDS[valid_idx])]
    #     train_inputs, train_masks, train_labels = self.encode_and_align(train_text, train_entities, False)
    #     validation_inputs, validation_masks, validation_labels = self.encode_and_align(valid_text, valid_entities, False)
    #     test_inputs, test_masks, test_labels = self.encode_and_align(valid_text, valid_entities, True)

    #     self.train_inputs = torch.tensor(train_inputs)
    #     self.validation_inputs = torch.tensor(validation_inputs)
    #     self.test_inputs = torch.tensor(test_inputs)
    #     self.train_labels = torch.tensor(train_labels)
    #     self.validation_labels = torch.tensor(validation_labels)
    #     self.test_labels = torch.tensor(test_labels)
    #     self.train_masks = torch.tensor(train_masks)
    #     self.validation_masks = torch.tensor(validation_masks)
    #     self.test_masks = torch.tensor(test_masks)
        

    def train_dataloader(self):
        # train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
        # train_sampler = RandomSampler(train_data)
        # return DataLoader(train_data, sampler=train_sampler, **params)
        return DataLoader(self.training_set, **train_params)

    def val_dataloader(self):
        # validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
        # validation_sampler = SequentialSampler(validation_data)
        # return DataLoader(validation_data, sampler=validation_sampler, **test_params)
        return DataLoader(self.valid_set, **test_params)
    
    def test_dataloader(self):
        # test_data = TensorDataset(self.test_inputs, self.test_masks, self.test_labels)
        # test_sampler = SequentialSampler(test_data)
        # return DataLoader(test_data, sampler=test_sampler, **test_params)
        return DataLoader(self.test_set, **test_params)

if __name__ == '__main__':
    
    experiment_id = f"type_{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join('./output_lightning', experiment_id)
    log_dir = os.path.join("logs", experiment_id)

    checkpoint_pth = './output_lightning/type_pytorch_longformer_20220219_211255/fold_0'

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if LOAD_CONFIG is not None:
        cfg = load_config(LOAD_CONFIG)
    else:
        cfg = config
        cfg_copy = config.copy()
        cfg_copy['logs_path'], cfg_copy['model_path'] = log_dir, [os.path.join(model_path, f'fold_{fold}') for fold in range(FOLDS)]
        save_config(cfg_copy, model_path, conf_name='config_experiment.json')
    
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'feedback-prize-2021', 'train.csv'))
    text_data_df = pd.read_csv(os.path.join(LOAD_TOKENS_FROM, TRAIN_FILE))  # Text text_data_df
        
    if DEBUG:
        text_data_df = text_data_df.sample(n=100, random_state=1)
        

    if not os.path.isfile(os.path.join(HF_HOME, 'pytorch_model.bin')):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
        tokenizer.save_pretrained(HF_HOME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_HOME, add_prefix_space=True)

    # CREATE TRAIN SUBSET AND VALID SUBSET
    text_data_df = text_data_df[['id','text', 'entities']]
    # pandas saves lists as string, we must convert back
    text_data_df.entities = text_data_df.entities.apply(lambda x: literal_eval(x) )

    IDS = train_df.id.unique()
    print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')
    
    if TRAIN_MODEL:
        for fold in range(FOLDS):
            
            # Save experiment in each fold. If no folds output is in fold 0
            model_fold_path = os.path.join(model_path, f'fold_{fold}')
            os.makedirs(model_fold_path, exist_ok=True)

            # TRAIN VALID SPLIT 90% 10%
            np.random.seed(42)
            train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
            valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
            np.random.seed(None)

            train_dataset = text_data_df.loc[text_data_df['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
            test_dataset = text_data_df.loc[text_data_df['id'].isin(IDS[valid_idx])].reset_index(drop=True)

            print("FULL Dataset: {}".format(text_data_df.shape))
            print("TRAIN Dataset: {}".format(train_dataset.shape))
            print("TEST Dataset: {}".format(test_dataset.shape))

            dm = FeedbackDataNLP(fold=fold, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_length=cfg['max_length'], train_df=train_dataset, valid_df=test_dataset, cfg=cfg)
            
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            chk_callback = ModelCheckpoint(
                monitor=cfg['monitor_metric'],
                dirpath=model_fold_path,
                filename='model_best',
                save_top_k=1,
                mode=cfg['mode'],
            )
            
            es_callback = EarlyStopping(
                monitor=cfg['monitor_metric'],
                min_delta=0.001,
                patience=5,
                verbose=True,
                mode=cfg['mode']
            )
            model = LightningFeedBack(model_name=MODEL_NAME, cfg=cfg, num_classes=dm.num_labels)

            tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

            trainer = Trainer(
                    # devices=[4, 5],  # error with scheduler from transformers if using more than one
                    devices=[5],
                    accelerator="gpu",
                    max_epochs=cfg['max_epochs'],
                    callbacks=[chk_callback, lr_monitor],  # es_callback
                    logger=tb_logger,
                    gradient_clip_val=cfg['gradient_clip_val'],
                    accumulate_grad_batches=cfg['accumulate_grad_batches'],
                )


            trainer.fit(model, dm)
    
    if COMPUTE_VAL_SCORE: # note this doesn't run during submit

        # TRAIN VALID SPLIT 90% 10%
        np.random.seed(42)
        train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
        valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
        np.random.seed(None)

        test_dataset = text_data_df.loc[text_data_df['id'].isin(IDS[valid_idx])].reset_index(drop=True)

        model = LightningFeedBack(model_name=MODEL_NAME, cfg=cfg, num_classes=len(cfg['output_labels']))
        checkpoint_path = os.path.join(checkpoint_pth, 'model_best.ckpt')
        model.load_from_checkpoint(checkpoint_path, model_name=MODEL_NAME, cfg=cfg)
        model.eval()
        model.to(cfg['device'])
        testing_loader = DataLoader(dataset(test_dataset, tokenizer, cfg['max_length'], True,  cfg=cfg), **test_params)

        # VALID TARGETS
        valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

        # OOF PREDICTIONS
        oof = get_predictions(test_dataset, testing_loader, model, cfg)

        # COMPUTE F1 SCORE
        f1s = []
        CLASSES = oof['class'].unique()
        
        for c in CLASSES:
            pred_df = oof.loc[oof['class']==c].copy()
            gt_df = valid.loc[valid['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            print(c,f1, flush=True)
            f1s.append(f1)
        
        print('Overall',np.mean(f1s), flush=True)

    if BUILD_SUBMISION:

        model = LightningFeedBack(model_name=MODEL_NAME, cfg=cfg, num_classes=len(cfg['output_labels']))
        checkpoint_path = os.path.join(checkpoint_pth, 'model_best.ckpt')
        model.load_from_checkpoint(checkpoint_path, model_name=MODEL_NAME, cfg=cfg)
        model.eval()
        model.to(cfg['device'])
        
        test_names, test_texts = [], []
        for f in list(os.listdir(os.path.join(BASE_PATH, 'feedback-prize-2021/test'))):
            test_names.append(f.replace('.txt', ''))
            test_texts.append(open(os.path.join(BASE_PATH, 'feedback-prize-2021/test', f), 'r').read())
        
        test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
        test_texts.head()
        
        # TEST DATASET
        test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True, cfg=cfg)
        test_texts_loader = DataLoader(test_texts_set, **test_params)

        sub = get_predictions(test_texts, test_texts_loader, model, cfg)
        sub.head()

        sub.to_csv("submission.csv", index=False)
