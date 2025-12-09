import torch
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Dataset, Data, DataLoader

from src.data.utils import documents_to_ids, clean_tokenization_sent

BIG_VAL = 1e9


class DocumentDataset(Dataset):
    def __init__(self, raw_documents, labels, max_len, article_identifiers,
                 vocab_sent, invert_vocab_sent, task="classification"):
        self.raw_documents = raw_documents   # <-- Store RAW TEXT, not tokenized IDs
        self.labels = labels
        self.max_len = max_len
        self.article_id = article_identifiers
        self.task = task
        self.vocab = vocab_sent
        self.inv_vocab = invert_vocab_sent

    def encode(self, sent_list):
        # convert a list of sentences to ids
        return [self.vocab.get(s, 0) for s in sent_list]  # O(1) per item

    def __len__(self):
        return len(self.raw_documents)

    def __getitem__(self, idx):
        # tokenization happens HERE instead of earlier
        sentence_list = self.raw_documents[idx]  # a list of strings
        document = self.encode(sentence_list)

        label = self.labels[idx]
        article_id = self.article_id[idx]
        original_len = len(document)
        num_pad_values = max(self.max_len - original_len, 0)

        if num_pad_values > 0:
            document = document + [0] * num_pad_values
            attention_mask = torch.tensor([0] * original_len + [-BIG_VAL] * num_pad_values)
            matrix_mask = torch.concat(
                [attention_mask[None, :].repeat(original_len, 1),
                 -BIG_VAL * torch.ones(num_pad_values, self.max_len)],
                dim=0)
        else:
            document = document[:self.max_len]
            attention_mask = torch.zeros(self.max_len)
            matrix_mask = torch.zeros(self.max_len, self.max_len)
            if self.task == "summarization":
                label = label[:self.max_len]

        return {
            'documents_ids': torch.tensor(document),
            'labels': torch.tensor(label),
            'src_key_padding_mask': attention_mask,
            'matrix_mask': matrix_mask,
            'article_id': article_id
        }



def create_loaders(df_full_train, df_test, max_len, batch_size, with_val=True, tokenizer_from_scratch=True,
                   path_ckpt='', df_val=None, task="classification", sent_tokenizer=False):
    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}

    if with_val:  ### If no validation partition is provided, split training data
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2)
        else:
            df_train = df_full_train
        print("Splitting sentences with sentence tokenizer?", sent_tokenizer)
        documents = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
                     list(df_train[source_column_name[task]])]
        documents_val = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc
                         in list(df_val[source_column_name[task]])]
        if task == "classification":
            labels = list(df_train[label_column_name[task]])
            labels_val = list(df_val[label_column_name[task]])
        else:
            labels = [clean_tokenization_sent(doc, "label") for doc in list(df_train[label_column_name[task]])]
            labels_val = [clean_tokenization_sent(doc, "label") for doc in list(df_val[label_column_name[task]])]

        article_identifiers = list(df_train[id_column_name[task]])
        article_identifiers_val = list(df_val[id_column_name[task]])

    else:
        documents = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
                     list(df_full_train[source_column_name[task]])]
        if task == "classification":
            labels = list(df_full_train[label_column_name[task]])
        else:
            labels = [clean_tokenization_sent(doc, "label") for doc in list(df_full_train[label_column_name[task]])]
        article_identifiers = list(df_full_train[id_column_name[task]])

    documents_test = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
                      list(df_test[source_column_name[task]])]
    if task == "classification":
        labels_test = list(df_test[label_column_name[task]])
    else:
        labels_test = [clean_tokenization_sent(doc, "label") for doc in list(df_test[label_column_name[task]])]
    article_identifiers_test = list(df_test[id_column_name[task]])

    if tokenizer_from_scratch == True:
        documents_ids, vocab_sent, invert_vocab_sent = documents_to_ids(documents)
        print("Train dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
        if with_val:
            documents_ids_val, vocab_sent, invert_vocab_sent = documents_to_ids(documents_val, from_scratch=False,
                                                                                dict_text_ids=vocab_sent,
                                                                                invert_text_ids=invert_vocab_sent,
                                                                                modifiable=True)
            print("Train + Val dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
            dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task)
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        documents_ids_test, vocab_sent, invert_vocab_sent = documents_to_ids(documents_test, from_scratch=False,
                                                                             dict_text_ids=vocab_sent,
                                                                             invert_text_ids=invert_vocab_sent,
                                                                             modifiable=True)
        print("Train + Val + Test dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")

    if tokenizer_from_scratch == False:
        if task == "classification":  # If the sentence-level vocabulary was previously created, load it from disk
            try:
                checkpoint = torch.load(path_ckpt)
                invert_vocab_sent = checkpoint['hyper_parameters']['invert_vocab_sent']
            except:
                sent_dict_disk = pd.read_csv(path_ckpt + "vocab_sentences.csv")
                invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
        else:
            sent_dict_disk = pd.read_csv(path_ckpt + "vocab_sentences.csv")
            invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}

        vocab_sent = {v: k for k, v in invert_vocab_sent.items()}
        #documents_ids = documents_to_ids(documents, from_scratch=False, dict_text_ids=vocab_sent,
        #                                 invert_text_ids=invert_vocab_sent)
        if with_val:
            documents_ids_val = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent,
                                                 invert_text_ids=invert_vocab_sent)
            #dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task)
            dataset_val = DocumentDataset(documents_val, labels_val, max_len, article_identifiers_val,
                                          vocab_sent, invert_vocab_sent, task)
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
        #documents_ids_test = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent,
        #                                      invert_text_ids=invert_vocab_sent)

    #dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers, task=task)
    dataset_train = DocumentDataset(documents, labels, max_len, article_identifiers,
                                    vocab_sent, invert_vocab_sent, task)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    #dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test, task=task)
    dataset_test = DocumentDataset(documents_test, labels_test, max_len, article_identifiers_test,
                                   vocab_sent, invert_vocab_sent, task)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    if with_val:
        return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent
    else:
        return loader_train, loader_test, vocab_sent, invert_vocab_sent
