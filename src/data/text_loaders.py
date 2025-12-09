import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize

from src.data.utils_vocab import (
    build_lmdb_vocab, load_lmdb_vocab,
    LmdbVocab, documents_to_ids,
)
from src.data.utils import clean_tokenization_sent

##############################################
# DocumentDataset – unchanged
##############################################

class DocumentDataset(Dataset):
    def __init__(self, documents, labels, max_len, article_identifiers, task="classification"):
        self.documents = documents
        self.labels = labels
        self.max_len = max_len
        self.article_id = article_identifiers
        self.task = task

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]
        article_id = self.article_id[idx]

        original_len = len(document)
        num_pad_values = max(self.max_len - original_len, 0)

        if num_pad_values > 0:
            document = document + [0] * num_pad_values
            attention_mask = torch.tensor([0] * original_len + [-1e9] * num_pad_values)
            matrix_mask = torch.cat(
                [attention_mask[None, :].repeat(original_len, 1),
                 -1e9 * torch.ones(num_pad_values, self.max_len)],
                dim=0
            )
        else:
            document = document[:self.max_len]
            attention_mask = torch.zeros(self.max_len)
            matrix_mask = torch.zeros(self.max_len, self.max_len)

        if self.task == "classification":
            return {
                'documents_ids': torch.tensor(document),
                'labels': torch.tensor(label),
                'src_key_padding_mask': attention_mask,
                'matrix_mask': matrix_mask,
                'article_id': article_id
            }
        else:
            # summarization: pad labels too
            return {
                'documents_ids': torch.tensor(document),
                'labels': torch.tensor(label + [-1] * num_pad_values),
                'src_key_padding_mask': attention_mask,
                'matrix_mask': matrix_mask,
                'article_id': article_id
            }


##############################################
# create_loaders — rewritten with LMDB vocab
##############################################

def create_loaders(df_full_train, df_test, max_len, batch_size, with_val=True,
                   tokenizer_from_scratch=True, path_ckpt='', df_val=None,
                   task="classification", sent_tokenizer=False):

    source_column_name = {
        "classification": 'article_text',
        "summarization": 'Cleaned_Article'
    }
    label_column_name = {
        "classification": 'article_label',
        "summarization": 'Calculated_Labels'
    }
    id_column_name = {
        "classification": 'article_id',
        "summarization": 'Article_ID'
    }

    # --------------------------------------------------------
    # LOAD LMDB VOCAB (required for both tokenizer_from_scratch True/False)
    # --------------------------------------------------------
    vocab_path = path_ckpt + "vocab_sentences.lmdb"
    print("Loading LMDB vocab from:", vocab_path)
    vocab = load_lmdb_vocab(vocab_path)

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------
    if with_val:
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2)
        else:
            df_train = df_full_train

        documents = [
            sent_tokenize(doc) if sent_tokenizer else clean_tokenization_sent(doc, "text")
            for doc in df_train[source_column_name[task]]
        ]
        documents_val = [
            sent_tokenize(doc) if sent_tokenizer else clean_tokenization_sent(doc, "text")
            for doc in df_val[source_column_name[task]]
        ]

        if task == "classification":
            labels = list(df_train[label_column_name[task]])
            labels_val = list(df_val[label_column_name[task]])
        else:
            labels = [
                clean_tokenization_sent(doc, "label")
                for doc in df_train[label_column_name[task]]
            ]
            labels_val = [
                clean_tokenization_sent(doc, "label")
                for doc in df_val[label_column_name[task]]
            ]

        article_identifiers = list(df_train[id_column_name[task]])
        article_identifiers_val = list(df_val[id_column_name[task]])

    else:
        documents = [...]
        labels = [...]
        article_identifiers = [...]

    documents_test = [
        sent_tokenize(doc) if sent_tokenizer else clean_tokenization_sent(doc, "text")
        for doc in df_test[source_column_name[task]]
    ]
    if task == "classification":
        labels_test = list(df_test[label_column_name[task]])
    else:
        labels_test = [
            clean_tokenization_sent(doc, "label")
            for doc in df_test[label_column_name[task]]
        ]
    article_identifiers_test = list(df_test[id_column_name[task]])

    # --------------------------------------------------------
    # CONVERT TO IDs USING LMDB
    # --------------------------------------------------------
    print("Converting documents to IDs...")

    documents_ids = documents_to_ids(documents, vocab, mode="text")
    documents_ids_test = documents_to_ids(documents_test, vocab, mode="text")

    if with_val:
        documents_ids_val = documents_to_ids(documents_val, vocab, mode="text")

    # --------------------------------------------------------
    # DATASETS + LOADERS
    # --------------------------------------------------------
    dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers, task=task)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    if with_val:
        dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test, task=task)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    if with_val:
        return loader_train, loader_val, loader_test, df_train, df_val, vocab, None
    else:
        return loader_train, loader_test, vocab, None
