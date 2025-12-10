import lmdb
import csv
import os
import shutil
import pandas as pd

class LmdbVocab:
    def __init__(self, path, readonly=True):
        self.env = lmdb.open(
            path,
            readonly=readonly,
            lock=not readonly,
            map_size=1024 * 1024 * 1024 * 512,  # 512GB virtual
            subdir=False,
            max_dbs=2
        )
        self.text2id = self.env.open_db(b"text2id")
        self.id2text = self.env.open_db(b"id2text")

    def get_id(self, sentence: str):
        with self.env.begin(db=self.text2id) as txn:
            return txn.get(sentence.encode("utf8"))

    def get_text(self, sid: int):
        key = str(sid).encode("utf8")
        with self.env.begin(db=self.id2text) as txn:
            return txn.get(key)


# =========================================
# build_lmdb_vocab_safe.py
# =========================================

def build_lmdb_vocab(csv_path, lmdb_path, chunk_size=100_000):
    """
    Build a safe LMDB vocab from CSV of Sentence -> Sentence_id.

    csv_path: str, path to 'vocab_sentences.csv'
    lmdb_path: str, path to LMDB file (local Colab path)
    chunk_size: int, number of rows per commit
    """

    # remove old LMDB if exists
    if os.path.exists(lmdb_path):
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)

    env = lmdb.open(
        lmdb_path,
        map_size=1024 * 1024 * 1024 * 512,  # 512 GB virtual
        subdir=False,
        max_dbs=2
    )

    # Named DBs
    text2id = env.open_db(b"text2id", dupsort=False)
    id2text = env.open_db(b"id2text", dupsort=False)

    print("Building LMDB vocab from CSV in chunks...")

    txn = env.begin(write=True)
    inserted = 0

    # Read CSV in chunks to avoid huge RAM usage
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for idx, row in chunk.iterrows():
            sentence = row['Sentence'].strip()
            sid = int(row['Sentence_id'])

            key_text = sentence.encode("utf8")
            val_text = str(sid).encode("utf8")
            key_sid = str(sid).encode("utf8")
            val_sid = sentence.encode("utf8")

            txn.put(key_text, val_text, db=text2id)
            txn.put(key_sid, val_sid, db=id2text)
            inserted += 1

            # commit periodically
            if inserted % chunk_size == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"Inserted {inserted:,} sentences...")

    # final commit
    txn.commit()
    env.close()
    print(f"LMDB vocab build complete! Total sentences: {inserted:,}")



def load_lmdb_vocab(lmdb_path):
    return LmdbVocab(lmdb_path)


def documents_to_ids(documents, vocab, mode="text"):
    results = []
    missing_total = 0

    if mode == "text":
        for doc in documents:
            doc_ids = []
            for sent in doc:
                val = vocab.get_id(sent)
                if val is None:
                    doc_ids.append(0)
                    missing_total += 1
                else:
                    doc_ids.append(int(val.decode("utf8")))
            results.append(doc_ids)
    else:  # label mode
        for doc in documents:
            doc_ids = []
            for sid in doc:
                try:
                    doc_ids.append(int(sid))
                except:
                    doc_ids.append(0)
                    missing_total += 1
            results.append(doc_ids)

    if missing_total > 0:
        print(f"[WARNING] {missing_total} sentences missing from LMDB.")

    return results


def retrieve_from_lmdb(vocab, list_ids):
    """Safely retrieve sentences by ID from LMDB."""
    results = []
    missing = 0

    with vocab.env.begin() as txn:
        for idx in list_ids:
            key = str(int(idx)).encode("utf-8")
            val = txn.get(key)
            if val is None:
                missing += 1
                results.append("[MISSING]")
            else:
                results.append(val.decode("utf8"))

    if missing > 0:
        print(f"[WARNING] {missing}/{len(list_ids)} IDs not found in LMDB.")

    return results
