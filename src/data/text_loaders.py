import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils_vocab import (
    build_lmdb_vocab,
    LmdbVocab,
)

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

    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name    = {"classification": 'article_id',    "summarization": 'Article_ID'}

    ##############################################
    # 1. Train/Val/Test splits and tokenization
    ##############################################

    if with_val:
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2)
        else:
            df_train = df_full_train

        documents = [clean_tokenization_sent(doc, "text")
                     for doc in df_train[source_column_name[task]]]
        documents_val = [clean_tokenization_sent(doc, "text")
                         for doc in df_val[source_column_name[task]]]

        labels = list(df_train[label_column_name[task]])
        labels_val = list(df_val[label_column_name[task]])

        article_identifiers = list(df_train[id_column_name[task]])
        article_identifiers_val = list(df_val[id_column_name[task]])

    else:
        df_train = df_full_train
        documents = [clean_tokenization_sent(doc, "text")
                     for doc in df_full_train[source_column_name[task]]]
        labels = list(df_full_train[label_column_name[task]])
        article_identifiers = list(df_full_train[id_column_name[task]])

    documents_test = [clean_tokenization_sent(doc, "text")
                      for doc in df_test[source_column_name[task]]]
    labels_test = list(df_test[label_column_name[task]])
    article_identifiers_test = list(df_test[id_column_name[task]])

    ##############################################
    # 2. Tokenizer logic (scratch or LMDB vocab)
    ##############################################

    if tokenizer_from_scratch:
        # Your old logic remains unchanged
        documents_ids, vocab_sent, invert_vocab_sent = documents_to_ids(documents)

        if with_val:
            documents_ids_val, _, _ = documents_to_ids(
                documents_val,
                from_scratch=False,
                dict_text_ids=vocab_sent,
                invert_text_ids=invert_vocab_sent
            )

        documents_ids_test, _, _ = documents_to_ids(
            documents_test,
            from_scratch=False,
            dict_text_ids=vocab_sent,
            invert_text_ids=invert_vocab_sent
        )

    else:
        ##############################################
        # *** LMDB VOCAB LOAD ***
        ##############################################

        vocab_csv  = os.path.join(path_ckpt, "vocab_sentences.csv")
        vocab_lmdb = os.path.join(path_ckpt, "vocab_sentences.lmdb")

        if not os.path.exists(vocab_lmdb):
            print("[INFO] LMDB vocab not found. Building it...")
            build_lmdb_vocab(vocab_csv, vocab_lmdb)

        print("[INFO] Loading LMDB vocab...")
        invert_vocab_sent = LmdbVocab(vocab_lmdb)

        # Now convert sentences → IDs using "documents_to_ids"
        # but this time, vocab_sent and invert_vocab_sent remain on disk
        vocab_sent = None  # no RAM dictionary

        documents_ids = documents_to_ids(
            documents, from_scratch=False,
            dict_text_ids=None,
            invert_text_ids=invert_vocab_sent
        )

        if with_val:
            documents_ids_val = documents_to_ids(
                documents_val, from_scratch=False,
                dict_text_ids=None,
                invert_text_ids=invert_vocab_sent
            )

        documents_ids_test = documents_to_ids(
            documents_test, from_scratch=False,
            dict_text_ids=None,
            invert_text_ids=invert_vocab_sent
        )

    ##############################################
    # 3. Create PyTorch datasets & loaders
    ##############################################

    dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers, task)
    loader_train  = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test, task)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    if with_val:
        dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent

    return loader_train, loader_test, vocab_sent, invert_vocab_sent
