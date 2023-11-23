import os

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class CFG:
    wandb=True
    competition='FB3'
    _wandb_kernel='nakama'
    debug=False
    apex=False
    print_freq=50
    num_workers=8
    model="cointegrated/rubert-tiny2"
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=3
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=64
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['Исполнитель', 'Группа тем', 'Тема']
    seed=42
    n_fold=5
    trn_fold=[0]
    train=True

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_score(y_trues, exec_predictions, topic_predictions, subtopic_predictions):
    exec_predictions = [np.argmax(el) for el in exec_predictions]
    topic_predictions = [np.argmax(el) for el in topic_predictions]
    subtopic_predictions = [np.argmax(el) for el in subtopic_predictions]

    exec_score = f1_score(y_trues[:, 0], exec_predictions, average="weighted")
    topic_score = f1_score(y_trues[:, 1], topic_predictions, average="weighted")
    subtopic_score = f1_score(y_trues[:, 2], subtopic_predictions, average="weighted")
    score = (exec_score + topic_score + subtopic_score) / 3
    return score, exec_score, topic_score, subtopic_score


def get_logger(filename=os.path.join(OUTPUT_DIR, 'train')):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

train = pd.read_csv("train_dataset_train.csv", delimiter=";")
train["len"] = train["Текст инцидента"].apply(len)
train = train[train.len > 4].reset_index(drop=True)
train.head()

executor_le = LabelEncoder()
topic_le = LabelEncoder()
subtopic_le = LabelEncoder()

executor_le.fit(train["Исполнитель"].tolist())
topic_le.fit(train["Группа тем"].tolist())
subtopic_le.fit(train["Тема"].tolist())

train["Исполнитель"] = executor_le.transform(train["Исполнитель"].tolist())
train["Группа тем"] = topic_le.transform(train["Группа тем"].tolist())
train["Тема"] = subtopic_le.transform(train["Тема"].tolist())
train.head()

Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'tokenizer'))
CFG.tokenizer = tokenizer

lengths = []
tk0 = tqdm(train['Текст инцидента'].fillna("").values, total=len(train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
CFG.max_len = max(lengths) + 2 # cls & sep
LOGGER.info(f"max_len: {CFG.max_len}")
CFG.max_len = 512

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['Текст инцидента'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label_exec = torch.tensor(self.labels[item][0], dtype=torch.long)
        label_topic = torch.tensor(self.labels[item][1], dtype=torch.long)
        label_subtopic = torch.tensor(self.labels[item][2], dtype=torch.long)
        return inputs, label_exec, label_topic, label_subtopic
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc_exec = nn.Linear(self.config.hidden_size, 10)
        self.fc_topic = nn.Linear(self.config.hidden_size, 26)
        self.fc_subtopic = nn.Linear(self.config.hidden_size, 195)
        self._init_weights(self.fc_exec)
        self._init_weights(self.fc_topic)
        self._init_weights(self.fc_subtopic)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output_exec = self.fc_exec(feature)
        output_topic = self.fc_topic(feature)
        output_subtopic = self.fc_subtopic(feature)
        
        return output_exec, output_topic, output_subtopic
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, label_exec, label_topic, label_subtopic) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        label_exec = label_exec.to(device)
        label_topic = label_topic.to(device)
        label_subtopic = label_subtopic.to(device)
        
        batch_size = label_exec.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            exec_pred, topic_pred, subtopic_pred = model(inputs)
            loss_exec = criterion(exec_pred, label_exec)
            loss_topic = criterion(topic_pred, label_topic)
            loss_subtopic = criterion(subtopic_pred, label_subtopic)
            loss = (loss_exec + loss_topic + loss_subtopic) / 3
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    exec_preds = []
    topic_preds = []
    subtopic_preds = []
    start = end = time.time()
    for step, (inputs, label_exec, label_topic, label_subtopic) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        label_exec = label_exec.to(device)
        label_topic = label_topic.to(device)
        label_subtopic = label_subtopic.to(device)

        batch_size = label_exec.size(0)
        with torch.no_grad():
            exec_pred, topic_pred, subtopic_pred = model(inputs)
            loss_exec = criterion(exec_pred, label_exec)
            loss_topic = criterion(topic_pred, label_topic)
            loss_subtopic = criterion(subtopic_pred, label_subtopic)
            loss = (loss_exec + loss_topic + loss_subtopic) / 3
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        exec_preds.append(exec_pred.to('cpu').numpy())
        topic_preds.append(topic_pred.to('cpu').numpy())
        subtopic_preds.append(subtopic_pred.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    exec_predictions = np.concatenate(exec_preds)
    topic_predictions = np.concatenate(topic_preds)
    subtopic_predictions = np.concatenate(subtopic_preds)
    return losses.avg, exec_predictions, topic_predictions, subtopic_predictions

def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, os.path.join(OUTPUT_DIR, 'config.pth'))
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss() # МБ добавить веса в лосс
    
    best_score = -1 * float('inf')

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, exec_predictions, topic_predictions, subtopic_predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score, exec_score, topic_score, subtopic_score = get_score(valid_labels, exec_predictions, topic_predictions, subtopic_predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {exec_score}, {topic_score}, {subtopic_score}')
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'exec_predictions': exec_predictions,
                        'topic_predictions': topic_predictions,
                        'subtopic_predictions': subtopic_predictions},
                        os.path.join(OUTPUT_DIR, f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"))

    exec_predictions = torch.load(os.path.join(OUTPUT_DIR, f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"), 
                             map_location=torch.device('cpu'))['exec_predictions']
    topic_predictions = torch.load(os.path.join(OUTPUT_DIR, f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"), 
                             map_location=torch.device('cpu'))['topic_predictions']
    subtopic_predictions = torch.load(os.path.join(OUTPUT_DIR, f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"), 
                             map_location=torch.device('cpu'))['subtopic_predictions']
    
    valid_folds["pred_Исполнитель"] = [np.argmax(el) for el in exec_predictions]
    valid_folds["pred_Группа тем"] = [np.argmax(el) for el in topic_predictions]
    valid_folds["pred_Тема"] = [np.argmax(el) for el in subtopic_predictions]
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

if __name__ == '__main__':
    
    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        
        exec_predictions = oof_df["pred_Исполнитель"].tolist()
        topic_predictions = oof_df["pred_Группа тем"].tolist()
        subtopic_predictions = oof_df["pred_Тема"].tolist()
        
        score, exec_score, topic_score, subtopic_score = get_score(labels, exec_predictions, topic_predictions, subtopic_predictions)
        LOGGER.info(f'Score: {score:.4f}  Scores: {exec_score}, {topic_score}, {subtopic_score}')
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(os.path.join(OUTPUT_DIR, 'oof_df.pkl'))