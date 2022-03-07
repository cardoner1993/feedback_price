from datetime import datetime
from email.policy import default
import os
import argparse

RUN_KAGGLE = False
BASE_PATH = '../input' if RUN_KAGGLE else '../data'
HF_HOME = os.path.join(BASE_PATH, 'py-bigbird-v26', "models")
os.makedirs(HF_HOME, exist_ok=True)

EXPERIMENT_NAME = "pytorch_longformer"

# os.environ['TRANSFORMERS_CACHE'] = HF_HOME
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from tqdm import tqdm
from ast import literal_eval

import numpy as np
import pandas as pd
import gc

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

from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup,  AutoModelForTokenClassification, get_cosine_schedule_with_warmup, LongformerConfig, LongformerModel, LongformerTokenizerFast
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

LOAD_CONFIG = None # None or path to config
LABEL_ALL_SUBTOKENS = True
FOLDS = 1


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
        self.num_labels = num_classes

        if self.cfg['build_custom_head']:
            # Create a head that will be trained for the custom problem.
            base_path = os.path.join(HF_HOME, 'pretrained_head')
            if not os.path.isfile(os.path.join(base_path, 'pytorch_model.bin')):
                config_model = LongformerConfig.from_pretrained(model_name) 
                config_model.num_labels = self.num_labels
                config_model.save_pretrained(base_path)

                self.model = LongformerModel.from_pretrained(model_name, config=config_model)
                self.model.save_pretrained(base_path)
            else:
                config_model = AutoConfig.from_pretrained(os.path.join(base_path, 'config.json')) 
                self.model = LongformerModel.from_pretrained(base_path, config=config_model)
        
        else:
            # Continue training the head.
            base_path = os.path.join(HF_HOME, 'pretrained_base')
            if not os.path.isfile(os.path.join(base_path, 'pytorch_model.bin')):
                config_model = AutoConfig.from_pretrained(model_name) 
                config_model.num_labels = self.num_labels
                config_model.save_pretrained(base_path)

                self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=config_model)
                self.model.save_pretrained(os.path.join(base_path, 'pytorch_model.bin'))
            else:
                config_model = AutoConfig.from_pretrained(os.path.join(base_path, 'config.json')) 
                self.model = AutoModelForTokenClassification.from_pretrained(base_path, config=config_model)
                
            
        # Warm up wait n epochs
        self.warmup_steps = 0
        # threshold (float) – Threshold for transforming probability or logit predictions to binary (0,1) 
        self.metric = torchmetrics.F1(num_classes=self.num_labels)

        self.loss = nn.CrossEntropyLoss()
        
        if self.cfg['build_custom_head']:
            self.linear = nn.Linear(self.cfg['hidden_size'], self.num_labels)
    
    def on_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
    
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
        if self.cfg['build_custom_head']:
            x = self.model(ids, mask)
            x = x[0]
            outputs = self.linear(x)
        
        else:
            outputs = self.model(input_ids=ids, attention_mask=mask, labels=b_labels, return_dict=False)
        
        return outputs

    def on_train_batch_start(self, batch, batch_idx):
        # Logs learning
        self.log('learning_rate', self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0], prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_labels = batch['labels']

        if self.cfg['build_custom_head']:
            logits = self(b_input_ids, b_input_mask, b_labels)
            loss = self.loss(logits.view(-1, self.num_labels), b_labels.view(-1))
        else:
            z = self(b_input_ids, b_input_mask, b_labels)
            loss, logits = z[0], z[1]

        flattened_targets = b_labels.view(-1)
        active_logits = logits.view(-1, self.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = b_labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.log('train_f1_score', self.metric(labels.cpu(), predictions.cpu()).item(), prog_bar=True, sync_dist=True)

        self.log('train_loss', loss.item(), prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_labels = batch['labels']
        
        if self.cfg['build_custom_head']:
            logits = self(b_input_ids, b_input_mask, b_labels)
            val_loss = self.loss(logits.view(-1, self.num_labels), b_labels.view(-1))
        else:
            z = self(b_input_ids, b_input_mask, b_labels)
            val_loss, logits = z[0], z[1]

        flattened_targets = b_labels.view(-1)
        active_logits = logits.view(-1, self.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = b_labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        self.log('val_f1_score', self.metric(labels.cpu(), predictions.cpu()).item(), prog_bar=True, sync_dist=True)

        self.log('val_loss', val_loss.item(), prog_bar=True, sync_dist=True)

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
    def __init__(self, fold, tokenizer, max_length, train_df, valid_df, cfg):
        super().__init__()
        self.fold = fold
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


def is_dir(path):
    """Check if file is path.

    Args:
        path (string): path to model

    Raises:
        ValueError: If path is not valid

    Returns:
        str: path to model
    """
    if not os.path.isdir(path):
        raise ValueError("The path is not a valid file")
    
    return path


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment to run. If not defined will default to pytorch_longformer and the date', default=None)
    parser.add_argument('-t', '--train', action='store_true', help='Train Model')
    parser.add_argument('-v', '--validate', action='store_true', help='Compute Validation Score')
    parser.add_argument('-m', '--model_path', type=is_dir, help='Compute Validation Score', default=None)
    parser.add_argument('-dt', '--device_train', required=False, help='Define the cuda device to use', default=None, nargs="+", type=int)
    parser.add_argument('-dv', '--device_valid', required=False, help='Define the cuda device to use in validation', default=None, type=int)

    args = parser.parse_args()

    if args.train and args.model_path is not None:
        print("Train set to True and model_path provided. The model path won't take effect")

    TRAIN_MODEL = args.train
    COMPUTE_VAL_SCORE = args.validate
    BUILD_SUBMISION = False

    print("TRAIN", TRAIN_MODEL, "VALIDATE", COMPUTE_VAL_SCORE, "BUILD Submision", BUILD_SUBMISION)

    if args.name is not None:
        EXPERIMENT_NAME = args.name
    
    experiment_id = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join('./output_lightning', experiment_id)
    log_dir = os.path.join("logs", experiment_id)

    checkpoint_pth = os.path.join(model_path, 'fold_0') if TRAIN_MODEL else args.model_path

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if LOAD_CONFIG is not None:
        cfg = load_config(LOAD_CONFIG)
    else:
        cfg = config
        cfg_copy = config.copy()
        cfg_copy['logs_path'], cfg_copy['model_path'] = log_dir, [os.path.join(model_path, f'fold_{fold}') for fold in range(FOLDS)]
        save_config(cfg_copy, model_path, conf_name='config_experiment.json')
    
    if args.device_valid is not None and 'cuda' in cfg['device']:
        cfg['device'] = f"cuda:{args.device_valid}"
    
    if args.device_train is not None and 'cuda' in cfg['device']:
        device = args.device_train
    else:
        device = None
    
    train_params = {'batch_size': cfg['train_batch_size'],
                'shuffle': True,
                'num_workers': config['num_workers'],
                'pin_memory':True
                }

    test_params = {'batch_size': config['valid_batch_size'],
                'shuffle': False,
                'num_workers': config['num_workers'],
                'pin_memory':True
                }

    train_df = pd.read_csv(os.path.join(BASE_PATH, 'feedback-prize-2021', 'train.csv'))
    text_data_df = pd.read_csv(os.path.join(LOAD_TOKENS_FROM, TRAIN_FILE))  # Text text_data_df
        
    if cfg['debug']:
        text_data_df = text_data_df.sample(n=100, random_state=1)
        
    if cfg['build_custom_head']:
        if not os.path.isfile(os.path.join(HF_HOME, 'pytorch_automodel.bin')):
            tokenizer = LongformerTokenizerFast.from_pretrained(cfg['model_name'], add_prefix_space=True)
            tokenizer.save_pretrained(os.path.join(HF_HOME, 'pretrained_head'))
        else:
            tokenizer = LongformerTokenizerFast.from_pretrained(os.path.join(HF_HOME, 'pretrained_head'), add_prefix_space=True)
    else:
        if not os.path.isfile(os.path.join(HF_HOME, 'pytorch_model.bin')):
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], add_prefix_space=True)
            tokenizer.save_pretrained(os.path.join(HF_HOME, 'pretrained_base'))
        else:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(HF_HOME, 'pretrained_base'), add_prefix_space=True)

    # CREATE TRAIN SUBSET AND VALID SUBSET
    text_data_df = text_data_df[['id','text', 'entities']]
    # pandas saves lists as string, we must convert back
    text_data_df.entities = text_data_df.entities.apply(lambda x: literal_eval(x) )

    IDS = train_df.id.unique()
    print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')
    
    if TRAIN_MODEL:
        for fold in range(FOLDS):

            print(f"Model will train for {cfg['max_epochs']} EPOCHS")
            
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

            dm = FeedbackDataNLP(fold=fold, tokenizer=tokenizer, max_length=cfg['max_length'], train_df=train_dataset, valid_df=test_dataset, cfg=cfg)
            
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            chk_callback = ModelCheckpoint(
                monitor=cfg['monitor_metric'],
                dirpath=model_fold_path,
                filename='model_best',
                save_top_k=1,
                mode=cfg['mode'],
            )

            callbacks = [chk_callback, lr_monitor]

            if cfg['early_stopping']:
            
                es_callback = EarlyStopping(
                    monitor=cfg['monitor_metric'],
                    min_delta=0.001,
                    patience=5,
                    verbose=True,
                    mode=cfg['mode']
                )

                callbacks += [es_callback]

            model = LightningFeedBack(model_name=cfg['model_name'], cfg=cfg, num_classes=dm.num_labels)

            tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

            trainer = Trainer(
                    strategy='dp',
                    devices=[5, 6] if device is None else device,
                    accelerator="gpu",
                    max_epochs=cfg['max_epochs'],
                    callbacks=callbacks,
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

        model = LightningFeedBack(model_name=cfg['model_name'], cfg=cfg, num_classes=len(cfg['output_labels']))
        checkpoint_path = os.path.join(checkpoint_pth, 'model_best.ckpt')
        model.load_from_checkpoint(checkpoint_path, model_name=cfg['model_name'], cfg=cfg, num_classes=len(cfg['output_labels']))
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

        model = LightningFeedBack(model_name=cfg['model_name'], cfg=cfg, num_classes=len(cfg['output_labels']))
        checkpoint_path = os.path.join(checkpoint_pth, 'model_best.ckpt')
        model.load_from_checkpoint(checkpoint_path, model_name=cfg['model_name'], cfg=cfg)
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
