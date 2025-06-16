import yaml
import argparse
# from graph_data_loaders import UnifiedAttentionGraphs_Sum  # , AttentionGraphs_Sum
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
import multiprocessing as mp

import gc
from torch_geometric.data import Data, Dataset
import torch
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from eval_models import filtering_matrices, get_threshold, clean_tokenization_sent, filtering_matrix
from base_model import MHASummarizer, retrieve_from_dict, MHAClassifier  # , MHASummarizer_extended
from utils import solve_by_creating_edge

os.environ["TOKENIZERS_PARALLELISM"] = "False"

# Global variable to hold shared config in workers
_shared_config = None


def _init_worker(config):
    global _shared_config
    _shared_config = config


# def process_single(args):
#     return _process_one(*args)

def _process_one(matrix, embeddings, article_id, doc_tensor, label):  # ,
    # filter_type, K, normalized, binarized,
    # model_ckpt, sent_model, mode, processed_dir):
    """
    Loads model, encodes embeddings, predicts adjacency matrix in batch,
    calls process_single, saves the resulting PyG Data object.
    """

    # Call graph construction logic
    graph_data = process_single(
        full_matrix=matrix,
        x=embeddings,
        doc_ids=doc_tensor,
        labels_sample=label,
        filter_type=_shared_config['filter_type'],
        K=_shared_config['K'],
        binarized=_shared_config['binarized'],
        normalized=_shared_config['normalized'],
        model_window=_shared_config['model_window'],
        max_len=_shared_config['max_len'],
    )

    # Save graph data file (name depends on mode)
    filename = f"data_{_shared_config['mode']}_{article_id}.pt" if _shared_config['mode'] in ["test",
                                                                                              "val"] else f"data_{article_id}.pt"
    torch.save(graph_data, os.path.join(_shared_config['processed_dir'], filename))


def process_single(full_matrix, x, doc_ids, labels_sample,
                   filter_type, K, normalized, binarized,
                   model_window, max_len):
    """
    Process one graph sample:
    - Filters adjacency matrix
    - Builds graph edges and attributes
    - Creates node features tensor from id_to_embedding dict
    - Returns PyG Data object with x, edge_index, edge_attr, y
    """

    label = clean_tokenization_sent(labels_sample, "label")
    #doc_ids = [int(x) for x in doc_as_ids[1:-1].split(",")]
    #doc_ids = torch.tensor(doc_ids)
    valid_sents = min(len(label), max_len)
    cropped_doc = doc_ids[:valid_sents]

    if filter_type is not None:
        filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window, degree_std=K, with_filtering=True,
                                           filtering_type=filter_type)
    else:
        filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window, degree_std=K,
                                           with_filtering=False)

    """"calculating edges"""
    match_ids = {k: v.item() for k, v in
                 zip(range(len(filtered_matrix)), cropped_doc)}  # key pos-i, value orig. sentence-id

    # Get unique IDs and build a mapping from original sentence ID to unique node index
    unique_ids = []
    for v in match_ids.values():
        if v not in unique_ids:
            unique_ids.append(v)
    id_to_node_idx = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}

    # final_label = []
    processed_ids = []
    source_list = []
    target_list = []
    orig_source_list = []
    orig_target_list = []
    edge_attrs = []
    all_edges = []

    final = []
    for elem in cropped_doc:
        if elem.item() not in final:
            final.append(elem.item())

    dict_orig_to_ide_graph = {k: v for k, v in
                              zip(final, range(len(final)))}  # key orig. sentence-id, value node-id for PyGeo

    for i in range(len(filtered_matrix)):
        only_one_entry = False
        fix_with_extra_edges = False
        if np.count_nonzero(filtered_matrix[i]) == 1:
            only_one_entry = True

        for j in range(len(filtered_matrix)):
            if filtered_matrix[i, j] != 0 and i != j:  # no self loops
                a = match_ids[i]  # original ID
                b = match_ids[j]
                if a != b:
                    if (a,
                        b) not in all_edges:  # do not aggregate an edge if matched source and target are the same sentence
                        orig_source_list.append(a)
                        orig_target_list.append(b)
                        all_edges.append((a, b))
                        # has_edges = True

                        # === CHANGED: use unique node idx, NOT processed_ids.index or dict_orig_to_ide_graph ===
                        source_list.append(id_to_node_idx[a])
                        target_list.append(id_to_node_idx[b])

                        if binarized:
                            edge_attrs.append(1.0)
                        else:
                            edge_attrs.append(
                                filtered_matrix[i, j].item())  # attention weight: as calculated by MHA trained model

                    else:  # (a,b) in all_edges:
                        pos_edge = all_edges.index((a, b))  # check update edge weights
                        if edge_attrs[pos_edge] < filtered_matrix[i, j].item():
                            edge_attrs[pos_edge] = filtered_matrix[i, j].item()
                else:
                    # if only_one_entry == True:
                    fix_with_extra_edges = True

            if filtered_matrix[i, j] != 0 and i == j and (
                    only_one_entry == True):  # check self-loop potentially producing disconnected graphs
                fix_with_extra_edges = True

            if fix_with_extra_edges == True:
                if match_ids[i] not in orig_target_list:
                    try:  # Adding edges to previous neighbor
                        b = match_ids[j - 1]
                        a = match_ids[i]
                        if binarized:
                            new_weight = 1.0
                        else:
                            new_weight = filtered_matrix[i, j].item() / 2
                        orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted = solve_by_creating_edge(
                            a, b, orig_source_list, orig_target_list, all_edges, source_list, target_list,
                            edge_attrs, processed_ids, dict_orig_to_ide_graph, new_weight)
                    except:
                        pass

                    try:
                        b = match_ids[j + 1]
                        a = match_ids[i]
                        if binarized:
                            new_weight = 1.0
                        else:
                            new_weight = filtered_matrix[i, j].item() / 2
                        orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted = solve_by_creating_edge(
                            a, b, orig_source_list, orig_target_list, all_edges, source_list, target_list,
                            edge_attrs, processed_ids, dict_orig_to_ide_graph, new_weight)
                    except:
                        pass

        # === CHANGED: append unique nodes only once ===
        if match_ids[i] not in processed_ids:
            processed_ids.append(match_ids[i])
            # final_label.append(label[i])

    # Assuming `label` is aligned to filtered_matrix rows, pick labels for unique nodes
    # If multiple duplicate nodes, choose the first occurrence's label (you can change logic if needed)
    unique_labels = []
    for uid in unique_ids:
        # Find first index where match_ids has uid, get corresponding label
        idx = next(k for k, v in match_ids.items() if v == uid)
        unique_labels.append(label[idx])

    x_unique = []
    for uid in unique_ids:
        idx = next(k for k, v in match_ids.items() if v == uid)
        x_unique.append(x[idx])

    x_unique = torch.stack(x_unique, dim=0)

    if len(source_list) != len(orig_source_list) or len(orig_source_list) != len(edge_attrs):
        print("Error in edge creation -- source and edge don't match")
        print("numero de edges acummulados:")
        print("source:", len(source_list))
        print("target:", len(target_list))
        print("orig_source:", len(orig_source_list))
        print("orig_target:", len(orig_target_list))
        print("edge_attrs:", len(edge_attrs))

    # reverse edges for all heuristics
    final_source = source_list + target_list
    final_target = target_list + source_list
    indexes = [final_source, final_target]
    final_orig_source_list = orig_source_list + orig_target_list
    final_orig_target_list = orig_target_list + orig_source_list
    edge_attrs = edge_attrs + edge_attrs  ### for reverse edge attributes (same weight - simetric matrix)

    all_indexes = torch.tensor(indexes).long()
    edge_attrs = torch.tensor(edge_attrs).float()

    if normalized:
        if all_indexes.shape[1] != 0:
            num_sent = torch.max(all_indexes[0].max(), all_indexes[1].max()) + 1
            adj_matrix = torch.zeros(num_sent, num_sent)
            for i in range(len(all_indexes[0])):
                adj_matrix[all_indexes[0][i], all_indexes[1][i]] = edge_attrs[i]
            for row in range(len(adj_matrix)):
                adj_matrix[row] = adj_matrix[row] / adj_matrix[row].max()

            temp = adj_matrix.reshape(-1)
            edge_attrs = temp[temp.nonzero()].reshape(-1)

    generated_data = Data(x=x_unique, edge_index=all_indexes, edge_attr=edge_attrs,
                          y=torch.tensor(unique_labels).int())  # labels adjusted for unique nodes

    generated_data.orig_edge_index = torch.tensor([final_orig_source_list, final_orig_target_list]).long()

    if generated_data.has_isolated_nodes():
        print("Error in graph -- isolated nodes detected")
        print(generated_data)
        print(generated_data.edge_index)
        print("Isolated nodes:", isolated_nodes.tolist())
        print('debug')
    elif generated_data.contains_self_loops():
        print("Error in graph -- self loops detected")
        print(generated_data)
        print(generated_data.edge_index)

    # out_name = f"data_{mode}_{article_id}.pt" if mode in ["test", "val"] else f"data_{article_id}.pt"
    # torch.save(generated_data, os.path.join(processed_dir, out_name))

    return generated_data


##llamar con loader usando batch size 1
class UnifiedAttentionGraphs_Sum(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, degree=0.5, model_ckpt="", mode="train",
                 transform=None, normalized=False, binarized=False, multi_layer_model=False,
                 pre_transform=None):  # input_matrices, invert_vocab_sent
        ### df_train, df_test, max_len, batch_size tambien en init?
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data).
        """

        self.filename = filename
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

        print("Loading model...")
        model = MHASummarizer.load_from_checkpoint(self.model_ckpt)
        model.eval()
        print('Model Loaded.')

        shared_config = {
            'model_ckpt': self.model_ckpt,
            'filter_type': self.filter_type,
            'K': self.K,
            'binarized': self.binarized,
            'normalized': self.normalized,
            'sent_model': self.sent_model,
            'mode': self.mode,
            'model_window': model.window,
            'max_len': model.max_len,
            'processed_dir': self.processed_dir
        }

        # starting multiprocessing
        mp.set_start_method("spawn", force=True)
        num_workers = min(4, len(os.sched_getaffinity(0)))

        # splitting data
        num_splits = 32
        total_samples = len(self.data_loader)
        split_size = np.ceil(total_samples / num_splits)
        print(f'Splitting data into {num_splits} chunks.')

        for split_idx in range(num_splits):
            start = split_idx * split_size
            end = min(start + split_size, total_samples)

            args_buffer = []
            for idx, batch_sample in enumerate(tqdm(self.data_loader)):
                if idx < start:
                    continue
                if idx >= end:
                    break

                # Convert doc_as_ids string to list of ints
                doc_as_ids = all_doc_as_ids[idx]
                doc_ids = [int(i) for i in doc_as_ids[1:-1].split(',') if i.strip().isdigit()]
                doc_tensor = torch.tensor(doc_ids)

                # Retrieve sentences using model's invert vocab and encode to embeddings
                sentences = retrieve_from_dict(model.invert_vocab_sent, doc_tensor)
                with torch.no_grad():
                    embeddings = self.sent_model.encode(sentences, convert_to_tensor=True)

                # Get attention weights
                _, matrix = model.predict_single(batch_sample)

                args = (
                    matrix.squeeze(0).cpu().numpy(),
                    embeddings,
                    all_article_ids[idx],
                    doc_tensor,
                    all_labels[idx],
                )
                args_buffer.append(args)

            # Process this chunk with multiprocessing as before

            print("Starting multiprocessing pool...")
            try:
                pool = mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(shared_config,))
                pool.starmap(_process_one, args_buffer)
            finally:
                print("Closing pool...")
                pool.close()  # No more tasks
                pool.join()  # Wait for workers to finish
                print("Pool closed and joined.")

            args_buffer.clear()


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


# TODO: check quality of the data
# TODO: this is the current log, must study and improve it?
## all VAL data (972 pt's) #970 # 801 and 950 not exist
'''
[VAL] Creating Graphs Objects with multiprocessing...
100%|██████████| 970/970 [06:27<00:00,  2.50it/s]
  0%|          | 0/970 [00:00<?, ?it/s]Starting multiprocessing pool...
 93%|█████████▎| 905/970 [2:38:40<20:00, 18.47s/it]
100%|██████████| 970/970 [3:09:06<00:00, 11.70s/it] 
Closing pool...
Pool closed and joined.
Creation time for Val Graph dataset: 11737.002381324768
Done!

[TEST] Creating Graphs Objects with multiprocessing...
100%|██████████| 972/972 [06:35<00:00,  2.46it/s]
  0%|          | 0/972 [00:00<?, ?it/s]Starting multiprocessing pool...
100%|██████████| 972/972 [2:15:43<00:00,  8.38s/it]
Closing pool...
Pool closed and joined.
Done!
Creation time for Test Graph dataset: 8543.168995141983

'''
'''
[VAL] Creating Graphs Objects with multiprocessing...
100%|██████████| 100/100 [00:49<00:00,  2.01it/s]
  0%|          | 0/100 [00:00<?, ?it/s]Starting multiprocessing pool...
100%|██████████| 100/100 [12:57<00:00,  7.78s/it]
Closing pool...
Pool closed and joined.
Creation time for Val Graph dataset: 830.56130027771
Done!
'''
## this is log before the latest edit
'''
Done
Requirements satisfied in: /scratch2/rldallitsako/datasets/AttnGraphs_GovReports/Extended_NoTemp/full_unified
Creating graphs with filter: full
Processing...
Model correctly loaded.
[VAL] Creating Graphs Objects with multiprocessing...
100%|██████████| 970/970 [42:10<00:00,  2.61s/it]
Starting multiprocessing pool...
  0%|          | 0/970 [00:15<?, ?it/s]
Closing pool...
'''


def main_run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger_name = "df_logger_cw.csv"
    model_name = "Extended_NoTemp"  ####### Extended_NoTemp_w30 is old data

    path_logger = "/scratch2/rldallitsako/HomoGraphs_GovReports/"
    root_graph = "/scratch2/rldallitsako/datasets/AttnGraphs_GovReports/"
    folder_results = "/home/rldallitsako/AttGraphs/GNN_Results_Summarizer/"
    filter_type = "full"
    save_flag = True  # config_file["saving_file"]
    unified_flag = True  # config_file["unified_nodes"]
    flag_binary = False  # config_file["binarized"]
    file_results = folder_results + "Doc_Level_Results"
    with_val = True  # False
    in_path = "scratch2/rldallitsako/datasets/GovReport-Sum/"

    # create folder if not exist
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    path_df_logger = os.path.join(path_logger, logger_name)
    df_logger = pd.read_csv(path_df_logger)

    # if flag_binary:
    # else:
    #    if unified_flag:
    path_root = os.path.join(root_graph, model_name, filter_type + "_unified")

    filename_train = "predict_train_documents.csv"
    # filename_val = "predict_val_documents.csv"
    filename_test = "predict_test_documents.csv"

    # import loader_data
    print("Loading data...")
    # if config_file["load_data_paths"]["with_val"]:
    df_train, df_val, df_test = load_data(in_path, "", "", "", "", True)

    # ids2remove_val = check_dataframe(df_val)
    # for id_remove in ids2remove_val:
    #     df_val = df_val.drop(id_remove)
    # df_val.reset_index(drop=True, inplace=True)
    # print("Val shape:", df_val.shape)
    #
    # else:
    #     df_train, df_test = load_data(**config_file["load_data_paths"])

    # df_train, df_test = load_data(in_path, "", "", "", "")

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
    # df_train, df_test = df_train[:20], df_test[:20]
    # df_val = df_val[100:110] #df_val[:100]

    # #################################minirun

    # if config_file["load_data_paths"]["with_val"]:
    # loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len, 1, df_val=df_val,
    # loader_train, _, _, _, _, _, _ = create_loaders(df_train, df_test, max_len, 1, df_val=df_val,
    #                                                                       task="summarization",
    #                                                                      tokenizer_from_scratch=False,
    #                                                                     path_ckpt=in_path)
    # else:
    loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, 1, with_val=False,
                                                     task="summarization", tokenizer_from_scratch=False,
                                                     path_ckpt=in_path)

    path_checkpoint = os.path.join(path_logger, model_name, "Extended_NoTemp-epoch=11-Val_f1-ma=0.52.ckpt")
    # print("\nLoading", model_name, "from:", path_checkpoint)
    # model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
    # model_window = model_lightning.window
    # print("Model temperature", model_lightning.temperature)
    # print("Done")

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
                print("\n\n================================================", file=f)
                print("Creating prediction files from pre-trained model:", model_name, file=f)
                print("Creating prediction files from pre-trained model:", model_name)
                print("================================================", file=f)
            else:
                print("\n\n================================================", file=f)
                print("Obtaining predictions from pre-trained model:", model_name, file=f)
                print("Obtaining predictions from pre-trained model:", model_name)
                print("================================================", file=f)
            start_creation = time.time()
            partition = "TRAIN"
            accs_tr, f1s_tr = model_lightning.predict_to_file(loader_train, saving_file=save_flag,
                                                              filename=filename_train,
                                                              path_root=path_root)  # /scratch/datasets/AttnGraphs_GovReports/Extended_Anneal/Attention/mean
            creation_time = time.time() - start_creation
            # print("[" + partition + "] File creation time:", creation_time, file=f)
            # print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item(), file=f)
            # print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy(),
            #      file=f)
            # print("================================================", file=f)
            print("[" + partition + "] File creation time:", creation_time)
            print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item())
            print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy())
            print("================================================")

            # # if config_file["load_data_paths"]["with_val"]:
            # start_creation = time.time()
            # partition = "VAL"
            # accs_v, f1s_v = model_lightning.predict_to_file(loader_val, saving_file=save_flag,
            #                                                     filename=filename_val, path_root=path_root)
            # creation_time = time.time() - start_creation
            # print("[" + partition + "] File creation time:", creation_time, file=f)
            # print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item(), file=f)
            # print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy(), file=f)
            # print("================================================", file=f)
            # print("[" + partition + "] File creation time:", creation_time)
            # print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item())
            # print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy())
            # print("================================================")
            # # else:
            # #     pass
            #
            # start_creation = time.time()
            # partition = "TEST"
            # accs_t, f1s_t = model_lightning.predict_to_file(loader_test, saving_file=save_flag, filename=filename_test,
            #                                                 path_root=path_root)
            # creation_time = time.time() - start_creation
            # print("[" + partition + "] File creation time:", creation_time, file=f)
            # print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item(), file=f)
            # print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy(), file=f)
            # print("================================================", file=f)
            # print("[" + partition + "] File creation time:", creation_time)
            # print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item())
            # print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy())
            # print("================================================")
            # f.close()

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
                                                   # degree=tolerance, model=model_lightning, mode="train",
                                                   degree=tolerance, model_ckpt=path_checkpoint, mode="train",
                                                   binarized=flag_binary)  # , multi_layer_model=multi_flag)
        creation_train = time.time() - start_creation
        print("Creation time for Train Graph dataset:", creation_train, file=f)
        print("Creation time for Train Graph dataset:", creation_train)

        # # graphs val
        # start_creation = time.time()
        # filename_val = "predict_val_documents.csv"
        # dataset_val = UnifiedAttentionGraphs_Sum(path_root, filename_val, filter_type, loader_val, degree=tolerance,
        #                                          model=model_lightning, mode="val",
        #                                          binarized=flag_binary)  # , multi_layer_model=multi_flag)
        # creation_val = time.time() - start_creation
        # print("Creation time for Val Graph dataset:", creation_val, file=f)
        # print("Creation time for Val Graph dataset:", creation_val)

        # start_creation = time.time()
        # filename_test = "predict_test_documents.csv"
        # dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, filter_type, loader_test, degree=tolerance,
        #                                           model=model_lightning, mode="test",
        #                                           binarized=flag_binary)  # , multi_layer_model=multi_flag)
        # creation_test = time.time() - start_creation
        # print("Creation time for Test Graph dataset:", creation_test, file=f)
        # print("Creation time for Test Graph dataset:", creation_test)
        # print("================================================", file=f)

        f.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_run()