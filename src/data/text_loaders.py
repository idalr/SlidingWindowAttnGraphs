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


def load_vocab_from_csv(path):

    path = os.path.join(path, "vocab_sentences.csv")
    print(f"\n[INFO] Loading vocabulary CSV:\n{path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vocabulary CSV not found: {path}")

    invert_vocab_sent = {}
    try:
        iterator = pd.read_csv(path, chunksize=200_000)
    except Exception as e:
        raise RuntimeError(f"Failed to load vocabulary CSV: {e}")

    for chunk in tqdm(iterator, desc="Reading vocab CSV in chunks", unit="chunk"):
        if "Sentence_id" not in chunk.columns or "Sentence" not in chunk.columns:
            raise ValueError("CSV must contain columns: Sentence_id, Sentence")
        invert_vocab_sent.update({int(k): str(v) for k, v in zip(chunk["Sentence_id"], chunk["Sentence"])})

    vocab_sent = {v: k for k, v in invert_vocab_sent.items()}
    print(f"[INFO] Vocabulary loaded: {len(vocab_sent)} entries\n")

    return vocab_sent, invert_vocab_sent

def build_memmap_dataset(doc_ids, labels, ids, max_len, prefix):
    n = len(doc_ids)

    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    docs_mm = np.memmap(prefix + "_docs.mm", dtype="int32", mode="w+", shape=(n, max_len))
    labels_mm = np.memmap(prefix + "_labels.mm", dtype="int32", mode="w+", shape=(n,))
    ids_mm = np.memmap(prefix + "_ids.mm", dtype="int32", mode="w+", shape=(n,))

    for i in tqdm(range(n), desc=f"Memmapping → {prefix}", unit="doc"):
        seq = doc_ids[i]
        L = len(seq)

        if L < max_len:
            padded = seq + [0] * (max_len - L)
        else:
            padded = seq[:max_len]

        docs_mm[i] = np.array(padded, dtype="int32")
        labels_mm[i] = int(labels[i])
        ids_mm[i] = int(ids[i])

    docs_mm.flush()
    labels_mm.flush()
    ids_mm.flush()

    return prefix + "_docs.mm", prefix + "_labels.mm", prefix + "_ids.mm"


class MemmapDocumentDataset(Dataset):
    def __init__(self, docs_path, labels_path, ids_path, max_len):
        self.docs = np.memmap(docs_path, mode="r", dtype="int32").reshape(-1, max_len)
        self.labels = np.memmap(labels_path, mode="r", dtype="int32")
        self.ids = np.memmap(ids_path, mode="r", dtype="int32")

        self.N = len(self.labels)
        self.max_len = max_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        doc = torch.from_numpy(self.docs[idx]).long()
        label = int(self.labels[idx])
        aid = int(self.ids[idx])

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
    ds = MemmapDocumentDataset(docs_path, labels_path, ids_path, max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )


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
    # 2 — tokenize documents into lists of sentences/tokens
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
    # 3 — vocabulary handling
    # -----------------------------
    if tokenizer_from_scratch:
        print("\n[INFO] Creating vocabulary from scratch is TOO RAM HEAVY — RECOMMENDED: use vocab_csv.\n")
        raise RuntimeError("You must supply vocab via CSV for large datasets.")
    else:
        # path_ckpt now MUST point to vocab CSV
        try:
            vocab_sent, invert_vocab_sent = load_vocab_from_csv(path_ckpt)
        except Exception as e:
            raise RuntimeError(f"Failed loading vocab CSV: {e}")

    # -----------------------------
    # 4 — convert documents to integer ids
    # -----------------------------
    def to_ids(docs):
        return documents_to_ids(
            docs,
            from_scratch=False,
            dict_text_ids=vocab_sent,
            invert_text_ids=invert_vocab_sent
        )

    print("[INFO] Converting documents to integer IDs...")
    documents_ids_train = to_ids(documents_train)
    documents_ids_test = to_ids(documents_test)
    if with_val:
        documents_ids_val = to_ids(documents_val)

    # -----------------------------
    # 5 — MEMMAP construction (RAM-safe)
    # -----------------------------
    print("\n[INFO] Building memmap datasets...\n")

    train_docs, train_labels, train_ids = build_memmap_dataset(
        documents_ids_train, labels_train, ids_train,
        max_len, "/content/mmap/train"
    )

    test_docs, test_labels, test_ids = build_memmap_dataset(
        documents_ids_test, labels_test, ids_test,
        max_len, "/content/mmap/test"
    )

    if with_val:
        val_docs, val_labels, val_ids = build_memmap_dataset(
            documents_ids_val, labels_val, ids_val,
            max_len, "/content/mmap/val"
        )

    # -----------------------------
    # 6 — GPU-pinned DataLoaders
    # -----------------------------
    loader_train = make_loader(train_docs, train_labels, train_ids, max_len, batch_size)
    loader_test = make_loader(test_docs, test_labels, test_ids, max_len, batch_size)

    if with_val:
        loader_val = make_loader(val_docs, val_labels, val_ids, max_len, batch_size)
        return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent
    else:
        return loader_train, loader_test, vocab_sent, invert_vocab_sent
