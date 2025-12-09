import os
import csv
import lmdb

##############################################
# (A) Build LMDB from large CSV (run once)
##############################################

def build_lmdb_vocab(csv_path, lmdb_path):
    """
    Streams the giant vocab CSV and stores key=Sentence_id, value=Sentence in LMDB.
    Safe for >2GB CSV. Zero RAM growth.
    """
    print(f"[INFO] Building LMDB vocab at: {lmdb_path}")
    print(f"[INFO] Source CSV: {csv_path}")

    # Large map_size is fine: LMDB allocates sparsely
    env = lmdb.open(lmdb_path, map_size=1024**3 * 120)  # 120GB max virtual size

    with env.begin(write=True) as txn:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # ['Sentence_id', 'Sentence']

            for i, row in enumerate(reader):
                if i % 500_000 == 0:
                    print(f"[INFO] Processed {i:,} rows...")

                try:
                    key = int(row[0])
                    val = row[1]
                except:
                    continue

                txn.put(str(key).encode(), val.encode())

    env.sync()
    env.close()
    print("[INFO] LMDB vocab build complete.")


##############################################
# (B) LMDB reader wrapper
##############################################

class LmdbVocab:
    """
    Thin wrapper: because LMDB returns bytes, we decode.
    """
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=2048)

    def get(self, key: int):
        with self.env.begin() as txn:
            val = txn.get(str(key).encode())
            if val is None:
                return None
            return val.decode()


##############################################
# (C) Safe lookup for your model
##############################################

def retrieve_from_lmdb(vocab: LmdbVocab, list_ids):
    """
    Converts a list of token IDs → list of sentences, skipping missing entries.

    Returns: list of sentences (strings)
    """
    missing = 0
    results = []

    for tid in list_ids:
        tid_int = int(tid.item())
        val = vocab.get(tid_int)
        if val is None:
            missing += 1
            continue
        results.append(val)

    if missing > 0:
        print(f"[WARNING] {missing}/{len(list_ids)} IDs missing from vocab.")

    return results
