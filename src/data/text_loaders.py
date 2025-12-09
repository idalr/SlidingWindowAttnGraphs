import os
import torch
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm

from src.data.utils import documents_to_ids, clean_tokenization_sent

BIG_VAL = 1e9


# class DocumentDataset(Dataset):
#     def __init__(self, raw_documents, labels, max_len, article_identifiers,
#                  vocab_sent, invert_vocab_sent, task="classification"):
#         self.raw_documents = raw_documents   # <-- Store RAW TEXT, not tokenized IDs
#         self.labels = labels
#         self.max_len = max_len
#         self.article_id = article_identifiers
#         self.task = task
#         self.vocab = vocab_sent
#         self.inv_vocab = invert_vocab_sent
#
#     def encode(self, sent_list):
#         # convert a list of sentences to ids
#         return [self.vocab.get(s, 0) for s in sent_list]  # O(1) per item
#
#     def __len__(self):
#         return len(self.raw_documents)
#
#     def __getitem__(self, idx):
#         # tokenization happens HERE instead of earlier
#         sentence_list = self.raw_documents[idx]  # a list of strings
#         document = self.encode(sentence_list)
#
#         label = self.labels[idx]
#         article_id = self.article_id[idx]
#         original_len = len(document)
#         num_pad_values = max(self.max_len - original_len, 0)
#
#         if num_pad_values > 0:
#             document = document + [0] * num_pad_values
#             attention_mask = torch.tensor([0] * original_len + [-BIG_VAL] * num_pad_values)
#             matrix_mask = torch.concat(
#                 [attention_mask[None, :].repeat(original_len, 1),
#                  -BIG_VAL * torch.ones(num_pad_values, self.max_len)],
#                 dim=0)
#         else:
#             document = document[:self.max_len]
#             attention_mask = torch.zeros(self.max_len)
#             matrix_mask = torch.zeros(self.max_len, self.max_len)
#             if self.task == "summarization":
#                 label = label[:self.max_len]
#
#         return {
#             'documents_ids': torch.tensor(document),
#             'labels': torch.tensor(label),
#             'src_key_padding_mask': attention_mask,
#             'matrix_mask': matrix_mask,
#             'article_id': article_id
#         }
#
#
#
# def create_loaders(df_full_train, df_test, max_len, batch_size, with_val=True, tokenizer_from_scratch=True,
#                    path_ckpt='', df_val=None, task="classification", sent_tokenizer=False):
#     source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
#     label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
#     id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}
#
#     if with_val:  ### If no validation partition is provided, split training data
#         if df_val is None:
#             df_train, df_val = train_test_split(df_full_train, test_size=0.2)
#         else:
#             df_train = df_full_train
#         print("Splitting sentences with sentence tokenizer?", sent_tokenizer)
#         documents = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
#                      list(df_train[source_column_name[task]])]
#         documents_val = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc
#                          in list(df_val[source_column_name[task]])]
#         if task == "classification":
#             labels = list(df_train[label_column_name[task]])
#             labels_val = list(df_val[label_column_name[task]])
#         else:
#             labels = [clean_tokenization_sent(doc, "label") for doc in list(df_train[label_column_name[task]])]
#             labels_val = [clean_tokenization_sent(doc, "label") for doc in list(df_val[label_column_name[task]])]
#
#         article_identifiers = list(df_train[id_column_name[task]])
#         article_identifiers_val = list(df_val[id_column_name[task]])
#
#     else:
#         documents = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
#                      list(df_full_train[source_column_name[task]])]
#         if task == "classification":
#             labels = list(df_full_train[label_column_name[task]])
#         else:
#             labels = [clean_tokenization_sent(doc, "label") for doc in list(df_full_train[label_column_name[task]])]
#         article_identifiers = list(df_full_train[id_column_name[task]])
#
#     documents_test = [sent_tokenize(doc) if sent_tokenizer == True else clean_tokenization_sent(doc, "text") for doc in
#                       list(df_test[source_column_name[task]])]
#     if task == "classification":
#         labels_test = list(df_test[label_column_name[task]])
#     else:
#         labels_test = [clean_tokenization_sent(doc, "label") for doc in list(df_test[label_column_name[task]])]
#     article_identifiers_test = list(df_test[id_column_name[task]])
#
#     if tokenizer_from_scratch == True:
#         documents_ids, vocab_sent, invert_vocab_sent = documents_to_ids(documents)
#         print("Train dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
#         if with_val:
#             documents_ids_val, vocab_sent, invert_vocab_sent = documents_to_ids(documents_val, from_scratch=False,
#                                                                                 dict_text_ids=vocab_sent,
#                                                                                 invert_text_ids=invert_vocab_sent,
#                                                                                 modifiable=True)
#             print("Train + Val dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
#             dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task)
#             loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
#
#         documents_ids_test, vocab_sent, invert_vocab_sent = documents_to_ids(documents_test, from_scratch=False,
#                                                                              dict_text_ids=vocab_sent,
#                                                                              invert_text_ids=invert_vocab_sent,
#                                                                              modifiable=True)
#         print("Train + Val + Test dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
#
#     if tokenizer_from_scratch == False:
#         if task == "classification":  # If the sentence-level vocabulary was previously created, load it from disk
#             try:
#                 checkpoint = torch.load(path_ckpt)
#                 invert_vocab_sent = checkpoint['hyper_parameters']['invert_vocab_sent']
#             except:
#                 sent_dict_disk = pd.read_csv(path_ckpt + "vocab_sentences.csv")
#                 invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
#         else:
#             sent_dict_disk = pd.read_csv(path_ckpt + "vocab_sentences.csv")
#             invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
#
#         vocab_sent = {v: k for k, v in invert_vocab_sent.items()}
#         #documents_ids = documents_to_ids(documents, from_scratch=False, dict_text_ids=vocab_sent,
#         #                                 invert_text_ids=invert_vocab_sent)
#         if with_val:
#             documents_ids_val = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent,
#                                                  invert_text_ids=invert_vocab_sent)
#             #dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task)
#             dataset_val = DocumentDataset(documents_val, labels_val, max_len, article_identifiers_val,
#                                           vocab_sent, invert_vocab_sent, task)
#             loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
#         #documents_ids_test = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent,
#         #                                      invert_text_ids=invert_vocab_sent)
#
#     #dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers, task=task)
#     dataset_train = DocumentDataset(documents, labels, max_len, article_identifiers,
#                                     vocab_sent, invert_vocab_sent, task)
#     loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
#     #dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test, task=task)
#     dataset_test = DocumentDataset(documents_test, labels_test, max_len, article_identifiers_test,
#                                    vocab_sent, invert_vocab_sent, task)
#     loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
#
#     if with_val:
#         return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent
#     else:
#         return loader_train, loader_test, vocab_sent, invert_vocab_sent

import sqlite3

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize  # or your own clean_tokenization_sent


# -------------------------
# 1️⃣ Helper: Create SQLite vocab DB
# -------------------------
def create_vocab_db(csv_path, db_path="/content/vocab.db"):
    if os.path.exists(db_path):
        print(f"[INFO] Using existing vocab DB: {db_path}")
        return db_path

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE vocab (sentence TEXT PRIMARY KEY, sentence_id INT)")

    iterator = pd.read_csv(csv_path, chunksize=100_000)
    for chunk in tqdm(iterator, desc="Creating vocab DB"):
        cur.executemany(
            "INSERT OR IGNORE INTO vocab VALUES (?, ?)",
            zip(chunk["Sentence"], chunk["Sentence_id"])
        )
        conn.commit()
    conn.close()
    print(f"[INFO] Vocabulary DB created at {db_path}")
    return db_path


# -------------------------
# 2️⃣ Helper: Stream documents → memmap
# -------------------------
def documents_to_ids_memmap(documents, db_path, memmap_prefix, max_len, labels=None, ids=None):
    n_docs = len(documents)
    print(f"[INFO] Converting {n_docs} docs → memmap: {memmap_prefix}")

    docs_mm = np.memmap(memmap_prefix + "_docs.mm", dtype="int32", mode="w+", shape=(n_docs, max_len))
    if labels is not None:
        labels_mm = np.memmap(memmap_prefix + "_labels.mm", dtype="int32", mode="w+", shape=(n_docs,))
    if ids is not None:
        ids_mm = np.memmap(memmap_prefix + "_ids.mm", dtype="int32", mode="w+", shape=(n_docs,))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for i in tqdm(range(n_docs), desc=f"Streaming docs → {memmap_prefix}", unit="doc"):
        doc = documents[i]
        doc_ids = []
        for sent in doc:
            cur.execute("SELECT sentence_id FROM vocab WHERE sentence=?", (sent,))
            res = cur.fetchone()
            doc_ids.append(res[0] if res is not None else 0)

        # pad/truncate
        if len(doc_ids) < max_len:
            doc_ids += [0] * (max_len - len(doc_ids))
        else:
            doc_ids = doc_ids[:max_len]

        docs_mm[i] = np.array(doc_ids, dtype="int32")
        if labels is not None:
            labels_mm[i] = int(labels[i])
        if ids is not None:
            ids_mm[i] = int(ids[i])

    docs_mm.flush()
    if labels is not None: labels_mm.flush()
    if ids is not None: ids_mm.flush()
    conn.close()

    return docs_mm.filename, labels_mm.filename if labels is not None else None, ids_mm.filename if ids is not None else None


# -------------------------
# 3️⃣ Dataset + DataLoader
# -------------------------
class MemmapDocumentDataset(Dataset):
    def __init__(self, docs_path, labels_path, ids_path, max_len):
        self.docs = np.memmap(docs_path, mode="r", dtype="int32").reshape(-1, max_len)
        self.labels = np.memmap(labels_path, mode="r", dtype="int32") if labels_path else None
        self.ids = np.memmap(ids_path, mode="r", dtype="int32") if ids_path else None
        self.N = len(self.docs)
        self.max_len = max_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        doc = torch.from_numpy(self.docs[idx]).long()
        label = int(self.labels[idx]) if self.labels is not None else None
        aid = int(self.ids[idx]) if self.ids is not None else None

        mask = (doc == 0).float() * -1e9
        matrix_mask = mask.unsqueeze(0).repeat(self.max_len, 1)

        return {
            "documents_ids": doc,
            "labels": label,
            "src_key_padding_mask": mask,
            "matrix_mask": matrix_mask,
            "article_id": aid,
        }


def make_loader(docs_path, labels_path, ids_path, max_len, batch_size):
    dataset = MemmapDocumentDataset(docs_path, labels_path, ids_path, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )


# -------------------------
# 4️⃣ Main create_loaders()
# -------------------------
def create_loaders(df_full_train, df_test, max_len, batch_size,
                   with_val=True, tokenizer_from_scratch=True,
                   path_ckpt='', df_val=None, task="classification",
                   sent_tokenizer=False):

    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}

    # -----------------------------
    # 1 — split
    # -----------------------------
    if with_val:
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2)
        else:
            df_train = df_full_train
    else:
        df_train = df_full_train

    # -----------------------------
    # 2 — tokenize documents
    # -----------------------------
    def proc(df):
        return [
            sent_tokenize(doc) if sent_tokenizer else clean_tokenization_sent(doc, "text")
            for doc in df[source_column_name[task]]
        ]

    documents_train = proc(df_train)
    documents_test = proc(df_test)
    if with_val:
        documents_val = proc(df_val)

    # labels
    if task == "classification":
        labels_train = list(df_train[label_column_name[task]])
        labels_test = list(df_test[label_column_name[task]])
        if with_val:
            labels_val = list(df_val[label_column_name[task]])
    else:
        labels_train = [clean_tokenization_sent(x, "label") for x in df_train[label_column_name[task]]]
        labels_test = [clean_tokenization_sent(x, "label") for x in df_test[label_column_name[task]]]
        if with_val:
            labels_val = [clean_tokenization_sent(x, "label") for x in df_val[label_column_name[task]]]

    ids_train = list(df_train[id_column_name[task]])
    ids_test = list(df_test[id_column_name[task]])
    if with_val:
        ids_val = list(df_val[id_column_name[task]])

    # -----------------------------
    # 3 — vocabulary DB
    # -----------------------------
    if tokenizer_from_scratch:
        raise RuntimeError("For large datasets, you MUST provide path_ckpt → vocab CSV")
    vocab_db = create_vocab_db("/content/drive/MyDrive/Colab Notebooks/datasets/arXiv/vocab_sentences.csv")

    # -----------------------------
    # 4 — memmap conversion
    # -----------------------------
    train_docs_mm, train_labels_mm, train_ids_mm = documents_to_ids_memmap(
        documents_train, vocab_db, "/content/mmap/train", max_len, labels_train, ids_train
    )

    test_docs_mm, test_labels_mm, test_ids_mm = documents_to_ids_memmap(
        documents_test, vocab_db, "/content/mmap/test", max_len, labels_test, ids_test
    )

    if with_val:
        val_docs_mm, val_labels_mm, val_ids_mm = documents_to_ids_memmap(
            documents_val, vocab_db, "/content/mmap/val", max_len, labels_val, ids_val
        )

    # -----------------------------
    # 5 — GPU-pinned DataLoaders
    # -----------------------------
    loader_train = make_loader(train_docs_mm, train_labels_mm, train_ids_mm, max_len, batch_size)
    loader_test = make_loader(test_docs_mm, test_labels_mm, test_ids_mm, max_len, batch_size)
    if with_val:
        loader_val = make_loader(val_docs_mm, val_labels_mm, val_ids_mm, max_len, batch_size)
        return loader_train, loader_val, loader_test, df_train, df_val
    else:
        return loader_train, loader_test
