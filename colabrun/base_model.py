import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy  as np
from sentence_transformers import SentenceTransformer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.core import Classifier_Lighting, Summarizer_Lighting
from src.models.utils import MultiHeadSelfAttention, SlidingWindowMultiHeadSelfAttention, retrieve_from_dict


class MHAClassifier(Classifier_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim,  max_len, lr, window,
                 intermediate=False, num_heads=4, dropout=False, class_weights=[],
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent = "", activation_attention="softmax"):
        super(MHAClassifier, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        if not intermediate:
            self.fc = nn.Linear(self.embed_dim, num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second linear layer

        self.dropout = dropout
        self.max_len = max_len
        self.lr = lr
        self.window = window
        self.intermediate = intermediate
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.temperature = 1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0

        # full MHA
        if window == 100:
            self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim,
                                                    num_heads, temperature=1, dropout=attn_dropout,
                                                    activation_attention=activation_attention)

        # sliding window MHA
        else:
            self.attention = SlidingWindowMultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(),
                                                                 embed_dim, num_heads, max_len=self.max_len, window_size=self.window, temperature=1,
                                                                 dropout=attn_dropout, activation_attention=activation_attention)

        #sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {} #{k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        self.save_hyperparameters(ignore=["invert_vocab_sent"]) ### for loading later

    def training_step(self, batch, batch_idx):
        return_val = super(MHAClassifier, self).training_step(batch, batch_idx)
        if self.temperature_scheduler == "anneal_decrease":
            # We recommend to set step with a very low value: 1e-3, 1e-4, 1e-5
            self.temperature = max(0.1, np.exp(- self.temperature_step * self.global_iter))
        self.global_iter += 1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHAClassifier, self).on_train_epoch_end()
        if self.temperature_scheduler == "step_decrease":
            self.temperature = self.temperature - self.temperature_step
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler == "step_decrease" or self.temperature_scheduler == "anneal_decrease":
            print("Temperature at the end of epoch:", self.temperature)

        return return_val

    ### only works for MHAClassifiers trained with the last changes
    def on_save_checkpoint(self, checkpoint) -> None:
        """Objects to include in checkpoint file"""
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        """Objects to retrieve from checkpoint file"""
        self.temperature = checkpoint["temperature_end_of_epoch"]

    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        x_emb = []
        for doc in doc_ids:
            source = self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc != 0]))
            complement = np.zeros((len(doc[doc == 0]), self.embed_dim))
            temp_emb = np.concatenate((source, complement))
            x_emb.append(temp_emb)

        x_emb = torch.tensor(x_emb)

        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask,
                                                   temperature=self.temperature)

        if self.dropout != False:
            attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        attn_output[matrix_mask[:, 0, :] != 0] = torch.nan
        attn_output = torch.nanmean(attn_output,
                                    dim=1)  # Mean pooling over the sequence length

        if not self.intermediate:
            logits = self.fc(attn_output)
        else:
            attn_output = torch.relu(self.fc1(attn_output))
            logits = self.fc2(attn_output)
        return logits, attn_weights


class MHASummarizer(Summarizer_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim, max_len, lr, window,
                 intermediate=False, num_heads=4, dropout=False, class_weights=[],  # criterion = nn.CrossEntropyLoss()
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent='', activation_attention="softmax"):
        super(MHASummarizer, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        if not intermediate:
            self.fc = nn.Linear(self.embed_dim, num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second linear layer

        self.dropout = dropout
        self.max_len = max_len
        self.lr = lr
        self.window = window
        self.intermediate = intermediate
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.temperature = 1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0

        # full MHA
        if window == 100:
            self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim,
                                                    num_heads, temperature=1, dropout=attn_dropout,
                                                    activation_attention=activation_attention)

        # sliding window MHA
        else:
            self.attention = SlidingWindowMultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(),
                                                                 embed_dim, num_heads, max_len=self.max_len,
                                                                 window_size=self.window, temperature=1,
                                                                 dropout=attn_dropout,
                                                                 activation_attention=activation_attention)

        # sent_dict_disk = pd.read_csv(path_invert_vocab_sent + "vocab_sentences.csv")
        # self.invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
        # self.save_hyperparameters(ignore=["invert_vocab_sent"])  ### for loading later

    def training_step(self, batch, batch_idx):
        return_val = super(MHASummarizer, self).training_step(batch, batch_idx)
        if self.temperature_scheduler == "anneal_decrease":
            # step should be a very low value, 1e-4 or ( 1e-3 or 1e-5)
            self.temperature = max(0.1, np.exp(- self.temperature_step * self.global_iter))
        self.global_iter += 1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHASummarizer, self).on_train_epoch_end()
        if self.temperature_scheduler == "step_decrease":
            self.temperature = self.temperature - self.temperature_step
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler == "step_decrease" or self.temperature_scheduler == "anneal_decrease":
            print("Temperature at the end of epoch:", self.temperature)

        return return_val

    def on_save_checkpoint(self, checkpoint) -> None:
        # Objects to include in checkpoint file
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        # Objects to retrieve from checkpoint file
        self.temperature = checkpoint["temperature_end_of_epoch"]

    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        x_emb = []
        for doc in doc_ids:
            source = self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc != 0]))
            complement = np.zeros((len(doc[doc == 0]), self.embed_dim))
            temp_emb = np.concatenate((source, complement))  # np.array
            x_emb.append(temp_emb)

        x_emb = torch.tensor(x_emb)
        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask,
                                                   temperature=self.temperature)

        if self.dropout != False:
            attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        if not self.intermediate:
            logits = self.fc(attn_output)
        # with 2 layers:
        else:
            attn_output = torch.relu(self.fc1(attn_output))
            logits = self.fc2(attn_output)

        return logits, attn_weights
