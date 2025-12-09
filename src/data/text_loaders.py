import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from src.data.utils import clean_tokenization_sent

# -------------------------
# Dataset Class for Memmap
# -------------------------
class MemmapDocumentDataset(Dataset):
    def __init__(self, docs_path, labels_path=None, ids_path=None, max_len=None):
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
# Helper: Stream documents → memmap
# -------------------------
def documents_to_ids_memmap_chunked(documents, vocab_dict, memmap_prefix, max_len,
                                    labels=None, ids=None, chunk_size=5000, default_id=0):
    """
    Convert documents to IDs in chunks using vocab_dict.
    Unknown sentences are replaced with default_id (e.g., 0).
    Warn once about missing sentences.
    """
    n_docs = len(documents)
    os.makedirs(os.path.dirname(memmap_prefix), exist_ok=True)

    # Create or load memmaps
    docs_mm = np.memmap(memmap_prefix + "_docs.mm", dtype="int32", mode="w+", shape=(n_docs, max_len))
    if labels is not None:
        labels_mm = np.memmap(memmap_prefix + "_labels.mm", dtype="int32", mode="w+", shape=(n_docs,))
    if ids is not None:
        ids_mm = np.memmap(memmap_prefix + "_ids.mm", dtype="int32", mode="w+", shape=(n_docs,))

    missing_sentences = set()  # store unique missing sentence IDs

    for start_idx in tqdm(range(0, n_docs, chunk_size), desc=f"Processing memmap {memmap_prefix}"):
        end_idx = min(start_idx + chunk_size, n_docs)
        chunk_docs = documents[start_idx:end_idx]
        if labels is not None:
            chunk_labels = labels[start_idx:end_idx]
        if ids is not None:
            chunk_ids = ids[start_idx:end_idx]

        for i, doc in enumerate(chunk_docs):
            doc_ids = []
            for sent in doc:
                if sent in vocab_dict:
                    doc_ids.append(vocab_dict[sent])
                else:
                    doc_ids.append(default_id)
                    missing_sentences.add(sent)

            if len(doc_ids) < max_len:
                doc_ids += [default_id] * (max_len - len(doc_ids))
            else:
                doc_ids = doc_ids[:max_len]

            docs_mm[start_idx + i] = np.array(doc_ids, dtype="int32")
            if labels is not None:
                labels_mm[start_idx + i] = int(chunk_labels[i])
            if ids is not None:
                ids_mm[start_idx + i] = int(chunk_ids[i])

    docs_mm.flush()
    if labels is not None: labels_mm.flush()
    if ids is not None: ids_mm.flush()

    if missing_sentences:
        print(f"[WARNING] {len(missing_sentences)} unique sentences not found in vocab. Replaced with default ID={default_id}.")

    return docs_mm.filename, labels_mm.filename if labels is not None else None, ids_mm.filename if ids is not None else None


# -------------------------
# Main create_loaders function
# -------------------------
def create_loaders(df_full_train, df_test, max_len, batch_size,
                   with_val=True, tokenizer_from_scratch=False,
                   path_ckpt='', df_val=None, task="classification",
                   sent_tokenizer=False, chunk_size=5000):
    """
    Memory-safe DataLoader creation using chunked streaming + memmap.
    vocab_dict: preloaded small dictionary (from CSV or checkpoint)
    """
    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}

    # -----------------------------
    # 1 — split train/val
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
    # 3 — load vocabulary CSV as streaming dict
    # -----------------------------
    vocab_dict = {}
    print("[INFO] Loading vocabulary CSV in chunks...")
    iterator = pd.read_csv(path_ckpt, chunksize=200_000)
    for chunk in tqdm(iterator, desc="Loading vocab"):
        vocab_dict.update(dict(zip(chunk["Sentence"], chunk["Sentence_id"])))
    print(f"[INFO] Loaded vocab size: {len(vocab_dict)} sentences")

    # -----------------------------
    # 4 — memmap conversion
    # -----------------------------
    train_docs_mm, train_labels_mm, train_ids_mm = documents_to_ids_memmap_chunked(
        documents_train, vocab_dict, "/content/mmap/train", max_len, labels_train, ids_train, chunk_size
    )

    test_docs_mm, test_labels_mm, test_ids_mm = documents_to_ids_memmap_chunked(
        documents_test, vocab_dict, "/content/mmap/test", max_len, labels_test, ids_test, chunk_size
    )

    if with_val:
        val_docs_mm, val_labels_mm, val_ids_mm = documents_to_ids_memmap_chunked(
            documents_val, vocab_dict, "/content/mmap/val", max_len, labels_val, ids_val, chunk_size
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
