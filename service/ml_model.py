import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG:
    num_workers = 8
    path = "output_me5"
    config_path = os.path.join(path, "config.pth")
    model = "intfloat/multilingual-e5-large"
    gradient_checkpointing = False
    batch_size = 1
    target_cols = ["Исполнитель", "Группа тем", "Тема"]
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    max_len = 512


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
            self.config.hidden_dropout = 0.0
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_dropout = 0.0
            self.config.attention_probs_dropout_prob = 0.0
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
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
        feature = self.pool(last_hidden_states, inputs["attention_mask"])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output_exec = self.fc_exec(feature)
        output_topic = self.fc_topic(feature)
        output_subtopic = self.fc_subtopic(feature)

        return output_exec, output_topic, output_subtopic


class Model:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = CustomModel(self.cfg, config_path=self.cfg.config_path, pretrained=False)
        state = torch.load(
            os.path.join(self.cfg.path, f"{self.cfg.model.replace('/', '-')}_fold0_best.pth"),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.cfg.path, "tokenizer"))
        self.m = nn.Softmax(dim=1)

        with open("ml/output_me5/executor_le.pkl", "rb") as f:
            self.exec_le = pickle.load(f)

        with open("ml/output_me5/topic_le.pkl", "rb") as f:
            self.topic_le = pickle.load(f)

        with open("ml/output_me5/subtopic_le.pkl", "rb") as f:
            self.subtopic_le = pickle.load(f)

    def prepare_input(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.cfg.max_len,
            pad_to_max_length=True,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor([v], dtype=torch.long)
        return inputs

    def predict(self, text, th):
        inputs = self.prepare_input(text)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            exec_pred, topic_pred, subtopic_pred = self.model(inputs)

        exec_pred = self.m(exec_pred)
        topic_pred = self.m(topic_pred)
        subtopic_pred = self.m(subtopic_pred)

        exec_pred = exec_pred.to("cpu").numpy()[0]
        topic_pred = topic_pred.to("cpu").numpy()[0]
        subtopic_pred = subtopic_pred.to("cpu").numpy()[0]

        th = th / 100

        exec_th = False
        for el in exec_pred:
            if el > th:
                exec_th = True
        topic_th = False
        for el in topic_pred:
            if el > th:
                topic_th = True
        subtopic_th = False
        for el in subtopic_pred:
            if el > th:
                subtopic_th = True

        if exec_th:
            exec_label = np.argmax(exec_pred)
            str_exec_label = self.exec_le.inverse_transform([exec_label])[0]
        else:
            str_exec_label = None

        if topic_th:
            topic_label = np.argmax(topic_pred)
            str_topic_label = self.topic_le.inverse_transform([topic_label])[0]
        else:
            str_topic_label = None

        if subtopic_th:
            subtopic_label = np.argmax(subtopic_pred)
            str_subtopic_label = self.subtopic_le.inverse_transform([subtopic_label])[0]
        else:
            str_subtopic_label = None

        return str_exec_label, str_topic_label, str_subtopic_label
