import yaml
import argparse
#from graph_data_loaders import UnifiedAttentionGraphs_Sum  # , AttentionGraphs_Sum
import torch
import os
import time
import pandas as pd
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from eval_models import retrieve_parameters
from preprocess_data import load_data
from data_loaders import create_loaders, check_dataframe
from base_model import MHASummarizer  # , MHASummarizer_extended

import gc
from torch_geometric.data import Data, Dataset
import torch
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from eval_models import filtering_matrices, get_threshold, clean_tokenization_sent, filtering_matrix
from base_model import MHASummarizer, retrieve_from_dict, MHAClassifier #, MHASummarizer_extended
from utils import solve_by_creating_edge

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def process_single(args):
    return _process_one(*args)

def _process_one(idx, pred, full_matrix, doc_ids, labels_sample, article_id,
                 filter_type, K, binarized, normalized,
                 model_window, max_len, sent_model, model_vocab):

    label = clean_tokenization_sent(labels_sample, "label")
    valid_sents = min(len(label), max_len)

    if filter_type is not None:
        filtered_matrix = filtering_matrix(
            full_matrix, valid_sents=valid_sents, window=model_window,
            degree_std=K, with_filtering=True, filtering_type=filter_type
        )
    else:
        filtered_matrix = filtering_matrix(
            full_matrix, valid_sents=valid_sents, window=model_window,
            degree_std=K, with_filtering=False
        )

    doc_ids = [int(x) for x in doc_ids[1:-1].split(",")]
    doc_ids = torch.tensor(doc_ids)
    cropped_doc = doc_ids[:valid_sents]

    match_ids = {k: v.item() for k, v in zip(range(len(filtered_matrix)), cropped_doc)}
    final = list(dict.fromkeys(x.item() for x in cropped_doc))
    dict_orig_to_graph = {k: v for v, k in enumerate(final)}

    final_label = []
    processed_ids = []
    source_list, target_list = [], []
    orig_source_list, orig_target_list = [], []
    edge_attrs = []
    all_edges = []

    skip_extra_edges = filter_type == "full"

    for i in range(len(filtered_matrix)):
        for j in range(len(filtered_matrix)):
            if filtered_matrix[i, j] != 0 and i != j:
                a, b = match_ids[i], match_ids[j]
                if a != b and (a, b) not in all_edges:
                    orig_source_list.append(a)
                    orig_target_list.append(b)
                    all_edges.append((a, b))
                    source_list.append(dict_orig_to_graph[a])
                    target_list.append(dict_orig_to_graph[b])
                    edge_attrs.append(1.0 if binarized else filtered_matrix[i, j].item())
                elif (a, b) in all_edges:
                    pos = all_edges.index((a, b))
                    edge_attrs[pos] = max(edge_attrs[pos], filtered_matrix[i, j].item())
            elif not skip_extra_edges and filtered_matrix[i, j] != 0 and i == j:
                if match_ids[i] not in orig_target_list:
                    for neighbor in [j - 1, j + 1]:
                        if 0 <= neighbor < len(filtered_matrix):
                            a = match_ids[i]
                            b = match_ids[neighbor]
                            weight = 1.0 if binarized else filtered_matrix[i, j].item() / 2
                            orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, _ = solve_by_creating_edge(
                                a, b, orig_source_list, orig_target_list, all_edges,
                                source_list, target_list, edge_attrs,
                                processed_ids, dict_orig_to_graph, weight
                            )

        if match_ids[i] not in processed_ids:
            processed_ids.append(match_ids[i])
            final_label.append(label[i])

    node_fea = sent_model.encode(
        retrieve_from_dict(model_vocab, torch.tensor(processed_ids))
    )

    final_source = source_list + target_list
    final_target = target_list + source_list
    final_orig_source = orig_source_list + orig_target_list
    final_orig_target = orig_target_list + orig_source_list
    edge_attrs = edge_attrs + edge_attrs

    edge_index = torch.tensor([final_source, final_target], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if normalized and edge_index.shape[1] > 0:
        num_nodes = edge_index.max().item() + 1
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[edge_index[0], edge_index[1]] = edge_attr
        for row in range(num_nodes):
            if adj[row].max() > 0:
                adj[row] /= adj[row].max()
        edge_attr = adj[adj != 0]

    node_fea = torch.tensor(node_fea, dtype=torch.float)
    data = Data(
        x=node_fea,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(final_label, dtype=torch.int),
    )
    data.article_id = torch.tensor(article_id) if not isinstance(article_id, torch.Tensor) else article_id
    data.orig_edge_index = torch.tensor([final_orig_source, final_orig_target], dtype=torch.long)

    return idx, data

##llamar con loader usando batch size 1
class UnifiedAttentionGraphs_Sum(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, degree=0.5, model_ckpt="", mode="train",
                 transform=None, normalized=False, binarized=False, multi_layer_model=False,
                 pre_transform=None):  # input_matrices, invert_vocab_sent
        ### df_train, df_test, max_len, batch_size tambien en init?
        # data loader es input_matrices, path_invert_vocab_sent='' puede ser model_ckpt, test=False es mode

        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data).
        """

        self.filename = filename #a crear post predict
        self.filter_type = filter_type
        self.data_loader = data_loader
        self.K = degree
        self.model_ckpt = model_ckpt
        self.mode = mode
        self.normalized = normalized
        self.binarized = binarized
        self.multi_layer_mha = multi_layer_model

        # Ensure the root is an absolute path
        self.root = os.path.abspath(root)
        # Create the full path for the filename (relative to the root folder)
        raw_folder_path = os.path.join(self.root, "raw")
        self.filename = os.path.join(raw_folder_path, filename)
        # Normalize paths to ensure correct separators on different OS (Windows/Unix)
        self.filename = os.path.normpath(self.filename)

        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
        super(UnifiedAttentionGraphs_Sum, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename  # +".csv"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.mode == "test":
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        if self.mode == "val":
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        all_doc_as_ids = self.data['doc_as_ids']
        all_labels = self.data['label']
        all_article_ids = self.data['article_id']
        all_batches = self.data_loader  # DO NOT pass this into multiprocessing workers

        print("Loading MHASummarizer model...")
        model = MHASummarizer.load_from_checkpoint(self.model_ckpt)
        model_window = model.window
        max_len = model.max_len
        invert_vocab_sent = model.invert_vocab_sent
        print("Model correctly loaded.")

        print(f"[{self.mode.upper()}] Predicting in batches...")
        predictions = []
        for batch_sample in tqdm(all_batches, total=len(all_batches)):
            pred_batch, matrix_batch = model.predict_single(batch_sample)
            predictions.extend(zip(pred_batch, matrix_batch))

        print(f"[{self.mode.upper()}] Processing with multiprocessing...")

        args_list = [
            (
                idx,
                pred,
                matrix,
                all_doc_as_ids[idx],
                all_labels[idx],
                all_article_ids[idx],
                self.filter_type,
                self.K,
                self.binarized,
                self.normalized,
                model_window,
                max_len,
                self.sent_model,
                invert_vocab_sent,
                )
            for idx, (pred, matrix) in enumerate(predictions)
        ]

        num_workers = len(os.sched_getaffinity(0))
        with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single, args_list), total=len(args_list)))

        print(f"[{self.mode.upper()}] Saving processed graphs...")
        for idx, data in results:
            out_name = f"data_{self.mode}_{idx}.pt" if self.mode in ["test", "val"] else f"data_{idx}.pt"
            torch.save(data, os.path.join(self.processed_dir, out_name))

    def len(self):
        return self.data.shape[0]  ##tamaño del dataset

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'), weights_only=False)
        if self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'), weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

        return data

# TODO: mp seems to work, but need to check quality of the code and outputs, esp compare to non-multiprocessing

def main_run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger_name = "df_logger_cw.csv"
    model_name = "Extended_NoTemp"

    path_logger = "/scratch2/rldallitsako/HomoGraphs_GovReports/"
    root_graph = "/scratch2/rldallitsako/datasets/AttnGraphs_GovReports/"
    folder_results = "/home/rldallitsako/AttGraphs/GNN_Results_Summarizer/"
    filter_type = "full"
    save_flag = True #config_file["saving_file"]
    unified_flag = True #config_file["unified_nodes"]
    flag_binary = False #config_file["binarized"]
    file_results = folder_results + "Doc_Level_Results"
    with_val = False
    in_path = "scratch2/rldallitsako/datasets/GovReport-Sum/"

    # create folder if not exist
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    path_df_logger = os.path.join(path_logger, logger_name)
    df_logger = pd.read_csv(path_df_logger)

    #if flag_binary:
    #else:
    #    if unified_flag:
    path_root = os.path.join(root_graph, model_name, filter_type + "_unified")

    filename_train = "predict_train_documents.csv"
    #filename_val = "predict_val_documents.csv"
    filename_test = "predict_test_documents.csv"

    # import loader_data
    print("Loading data...")
    # if config_file["load_data_paths"]["with_val"]:
    #     df_train, df_val, df_test = load_data(**config_file["load_data_paths"])
    #
    #     ids2remove_val = check_dataframe(df_val)
    #     for id_remove in ids2remove_val:
    #         df_val = df_val.drop(id_remove)
    #     df_val.reset_index(drop=True, inplace=True)
    #     print("Val shape:", df_val.shape)
    #
    # else:
    #     df_train, df_test = load_data(**config_file["load_data_paths"])

    df_train, df_test = load_data(in_path, "", "", "", "")

    ids2remove_train = check_dataframe(df_train)
    for id_remove in ids2remove_train:
        df_train = df_train.drop(id_remove)
    df_train.reset_index(drop=True, inplace=True)
    print("Train shape:", df_train.shape)

    ids2remove_test = check_dataframe(df_test)
    for id_remove in ids2remove_test:
        df_test = df_test.drop(id_remove)
    df_test.reset_index(drop=True, inplace=True)
    print("Test shape:", df_test.shape)

    max_len = 1000  # Maximum number of sentences in a document
    print("Max number of sentences allowed in document:", max_len)

    # #################################minirun
    df_train, df_test = df_train[-20:], df_test[-20:]
    # #################################minirun

    # if config_file["load_data_paths"]["with_val"]:
    #     loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len, 1, df_val=df_val,
    #                                                                        task="summarization",
    #                                                                        tokenizer_from_scratch=False,
    #                                                                        path_ckpt=config_file["load_data_paths"][
    #                                                                            "in_path"])
    # else:
    loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, 1, with_val=False,
                                                         task="summarization", tokenizer_from_scratch=False,
                                                         path_ckpt=in_path)

    path_checkpoint = os.path.join(path_logger, model_name, "Extended_NoTemp-epoch=11-Val_f1-ma=0.52.ckpt")
    print("\nLoading", model_name, "from:", path_checkpoint)
    model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
    model_window = model_lightning.window
    print("Model temperature", model_lightning.temperature)
    print("Done")

    path_filename_train = os.path.join(path_root, "raw", filename_train)
    if os.path.exists(path_filename_train):
        print("Requirements satisfied in:", path_root)
    else:  # if not os.path.exists(path_filename_train)::
        if save_flag:
            print("\nCreating required files for PyTorch Geometric Dataset Creation from pre-trained MHASummarizer...")
        # else:
        #     print("\nObtaining predictions from pre-trained Single-layer MHASummarizer...")

        with open(file_results + '_' + model_name + '.txt', 'a') as f:
            if save_flag:
                #print("\n\n================================================", file=f)
                #print("Creating prediction files from pre-trained model:", model_name, file=f)
                print("Creating prediction files from pre-trained model:", model_name)
                #print("================================================", file=f)
            else:
                #print("\n\n================================================", file=f)
                #print("Obtaining predictions from pre-trained model:", model_name, file=f)
                print("Obtaining predictions from pre-trained model:", model_name)
                #print("================================================", file=f)
            start_creation = time.time()
            partition = "TRAIN"
            accs_tr, f1s_tr = model_lightning.predict_to_file(loader_train, saving_file=save_flag,
                                                              filename=filename_train,
                                                              path_root=path_root)  # /scratch/datasets/AttnGraphs_GovReports/Extended_Anneal/Attention/mean
            creation_time = time.time() - start_creation
            #print("[" + partition + "] File creation time:", creation_time, file=f)
            #print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item(), file=f)
            #print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy(),
            #      file=f)
            #print("================================================", file=f)
            print("[" + partition + "] File creation time:", creation_time)
            print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item())
            print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy())
            print("================================================")

            # if config_file["load_data_paths"]["with_val"]:
            #     start_creation = time.time()
            #     partition = "VAL"
            #     accs_v, f1s_v = model_lightning.predict_to_file(loader_val, saving_file=save_flag,
            #                                                     filename=filename_val, path_root=path_root)
            #     creation_time = time.time() - start_creation
            #     print("[" + partition + "] File creation time:", creation_time, file=f)
            #     print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item(), file=f)
            #     print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy(), file=f)
            #     print("================================================", file=f)
            #     print("[" + partition + "] File creation time:", creation_time)
            #     print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item())
            #     print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy())
            #     print("================================================")
            # else:
            #     pass

            start_creation = time.time()
            partition = "TEST"
            accs_t, f1s_t = model_lightning.predict_to_file(loader_test, saving_file=save_flag, filename=filename_test,
                                                            path_root=path_root)
            creation_time = time.time() - start_creation
            #print("[" + partition + "] File creation time:", creation_time, file=f)
            #print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item(), file=f)
            #print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy(), file=f)
            #print("================================================", file=f)
            print("[" + partition + "] File creation time:", creation_time)
            print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item())
            print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy())
            #print("================================================")
            f.close()

        print("Finished and saved in:", path_root)

    with open(file_results + '_' + model_name + '.txt', 'a') as f:

        tolerance = 0.5
        print("\n\n-----------------------------------------------", file=f)
        if flag_binary:
            print("Creating graphs with filter:", filter_type, "-- binarized", file=f)
            print("Creating graphs with filter:", filter_type, "-- binarized")
        else:
            print("Creating graphs with filter:", filter_type, file=f)
            print("Creating graphs with filter:", filter_type)
        print("-----------------------------------------------", file=f)
        start_creation = time.time()
        filename_train = "predict_train_documents.csv"
        dataset_train = UnifiedAttentionGraphs_Sum(path_root, filename_train, filter_type, loader_train,
                                                   degree=tolerance, model_ckpt=path_checkpoint, mode="train",
                                                   binarized=flag_binary)  # , multi_layer_model=multi_flag)
        creation_train = time.time() - start_creation
        print("Creation time for Train Graph dataset:", creation_train, file=f)
        print("Creation time for Train Graph dataset:", creation_train)

        # # graphs val
        # start_creation = time.time()
        # filename_val = "predict_val_documents.csv"
        # dataset_val = UnifiedAttentionGraphs_Sum(path_root, filename_val, filter_type, loader_val, degree=tolerance,
        #                                          model_ckpt=path_checkpoint, mode="val",
        #                                          binarized=flag_binary)  # , multi_layer_model=multi_flag)
        # creation_val = time.time() - start_creation
        # print("Creation time for Val Graph dataset:", creation_val, file=f)
        # print("Creation time for Val Graph dataset:", creation_val)

        # start_creation = time.time()
        # filename_test = "predict_test_documents.csv"
        # dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, filter_type, loader_test, degree=tolerance,
        #                                           model_ckpt=path_checkpoint, mode="test",
        #                                           binarized=flag_binary)  # , multi_layer_model=multi_flag)
        # creation_test = time.time() - start_creation
        # print("Creation time for Test Graph dataset:", creation_test, file=f)
        # print("Creation time for Test Graph dataset:", creation_test)
        # print("================================================", file=f)

        f.close()

if __name__ == "__main__":
    main_run()