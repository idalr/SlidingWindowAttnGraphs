import re
from torch_geometric.data import Data, Dataset
import torch
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from base_model import MHASummarizer, retrieve_from_dict, MHAClassifier
from src.data.graph_utils import solve_by_creating_edge, filtering_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class UnifiedAttentionGraphs_Class(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, degree=0.5, model="", mode="train",
                 transform=None, normalized=False, binarized=False, multi_layer_model=False, pre_transform=None):

        """
        root = Where the graph dataset should be stored. This folder is split into raw_dir (featured dataset) and processed_dir (processed graph data).
        filename = CSV file with the dataset to be processed. It should contain the following columns: article_id, label, doc_as_ids.
                    A row example would be: 0, 1, "[32603, 32604, 32611, 32612, 32613, 0, 0, 0, 0, 0]" for a document with article_id=0, label=1 and 5 sentences.
        filter_type = type of filtering to be applied to the attention matrices. If None, no filtering is applied.
        data_loader = data loader for MHA model forward. It should be a list of batches.
        degree = tolerance degree of filtering to be applied to the attention matrices. Preliminary experiments suggest a value of 0.5.
        model_ckpt = path to the trained MHA-based model checkpoint to be loaded --- The models is required to forward the data and obtain the corresponding attention matrices.
        mode = "train", "val" or "test" - to specify the mode of the graph dataset.
        transform = default None. Do not change.
        normalized = if True, the edges are normalized (divided by the maximum value of each row) instead of using the original attention weights as edge attributes.
        binarized = if True, the edges are binarized (1.0) instead of using the attention weights as edge attributes.
        pre_transform = default None. Do not change.
        """

        self.filename = filename
        self.filter_type = filter_type
        self.data_loader = data_loader
        self.K = degree
        self.model = model
        self.mode = mode
        self.normalized = normalized
        self.binarized = binarized
        self.multi_layer_mha = multi_layer_model
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
        super(UnifiedAttentionGraphs_Class, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename  # CSV filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.mode == "test":
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.mode == "val":
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        all_doc_as_ids = self.data['doc_as_ids']
        all_labels = self.data['label']
        all_article_identifiers = self.data['article_id']
        all_batches = self.data_loader

        print("Loading MHAClassifier model...")
        #model_lightning = MHAClassifier.load_from_checkpoint(self.model_ckpt)
        model_lightning = self.model
        model_window = int(self.model.window)
        print("Model correctly loaded.")

        if self.mode == "test":
            print("\n[TEST] Creating Graphs Objects...")
        elif self.mode == "val":
            print("\n[VAL] Creating Graphs Objects...")
        else:
            print("\n[TRAIN] Creating Graphs Objects...")

        ide = 0
        for batch_sample in tqdm(all_batches, total=len(all_batches)):
            pred_batch, full_matrix_batch = model_lightning.predict_single(batch_sample)

            for pred, full_matrix in zip(pred_batch, full_matrix_batch):
                doc_ids = all_doc_as_ids[ide]
                label = all_labels[ide]
                doc_ids = [int(element) for element in doc_ids[1:-1].split(",")]
                doc_ids = torch.tensor(doc_ids)

                try:
                    valid_sents = (doc_ids == 0).nonzero()[0]
                    cropped_doc = doc_ids[:valid_sents]
                except:
                    valid_sents = len(doc_ids)
                    cropped_doc = doc_ids

                if self.filter_type is not None:
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, degree_std=self.K,
                                                       window=model_window, with_filtering=True,
                                                       filtering_type=self.filter_type)
                else:
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, degree_std=self.K,
                                                       window=model_window, with_filtering=False)

                """"calculating edges"""
                match_ids = {k: v.item() for k, v in
                             zip(range(len(filtered_matrix)), cropped_doc)}  # key pos-i, value orig. sentence-id

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

                dict_orig_to_ide_graph = {k: v for k, v in zip(final, range(
                    len(final)))}  # key orig. sentence-id, value node-id for PyGeo

                for i in range(len(filtered_matrix)):
                    only_one_entry = False
                    fix_with_extra_edges = False
                    if filtered_matrix[i].count_nonzero().item() == 1:
                        only_one_entry = True

                    for j in range(len(filtered_matrix)):
                        if filtered_matrix[i, j] != 0 and i != j:  # no self loops
                            a = match_ids[i]  # check match original ids
                            b = match_ids[j]
                            if a != b:
                                if (a,
                                    b) not in all_edges:  # do not aggregate an edge if matched source and target are the same sentence
                                    orig_source_list.append(a)
                                    orig_target_list.append(b)
                                    all_edges.append((a, b))
                                    # has_edges = True

                                    if a in processed_ids:  # adapt id to the next position in processed
                                        source_list.append(processed_ids.index(a))  # requires int from 0 to len
                                    else:
                                        source_list.append(dict_orig_to_ide_graph[a])  # requires int from 0 to len

                                    if b in processed_ids:  # adapt id to the next position in processed
                                        target_list.append(processed_ids.index(b))  # requires int from 0 to len
                                    else:
                                        target_list.append(dict_orig_to_ide_graph[b])  # requires int from 0 to len

                                    if self.binarized:
                                        edge_attrs.append(1.0)
                                    else:
                                        edge_attrs.append(filtered_matrix[
                                                              i, j].item())  # attention weight: as calculated by MHA trained model

                                else:  # (a,b) in all_edges:
                                    pos_edge = all_edges.index((a, b))  # check update edge weights
                                    if edge_attrs[pos_edge] < filtered_matrix[i, j].item():
                                        edge_attrs[pos_edge] = filtered_matrix[i, j].item()
                            else:
                                # if only_one_entry == True:
                                fix_with_extra_edges = True

                        if filtered_matrix[i, j] != 0 and i == j and (
                                only_one_entry == True):  # check self-loop potentialy producing disconnected graphs
                            fix_with_extra_edges = True

                        if fix_with_extra_edges == True:
                            if match_ids[i] not in orig_target_list:
                                try:  # add edges to previous and/or next neighbor
                                    b = match_ids[j - 1]
                                    a = match_ids[i]
                                    if self.binarized:
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
                                    if self.binarized:
                                        new_weight = 1.0
                                    else:
                                        new_weight = filtered_matrix[i, j].item() / 2
                                    orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted = solve_by_creating_edge(
                                        a, b, orig_source_list, orig_target_list, all_edges, source_list, target_list,
                                        edge_attrs, processed_ids, dict_orig_to_ide_graph, new_weight)
                                except:
                                    pass

                    if match_ids[i] not in processed_ids:
                        processed_ids.append(match_ids[i])

                if len(source_list) != len(orig_source_list) or len(orig_source_list) != len(edge_attrs):
                    print("Error when creating edges -- source and edge do not match")
                    print("Number of edges acummulated:")
                    print("source:", len(source_list))
                    print("target:", len(target_list))
                    print("orig_source:", len(orig_source_list))
                    print("orig_target:", len(orig_target_list))
                    print("edge_attrs:", len(edge_attrs))
                    break

                """"calculating node features"""
                node_fea = self.sent_model.encode(
                    retrieve_from_dict(model_lightning.invert_vocab_sent, torch.tensor(processed_ids)))

                # reverse edges for undirected graphs
                final_source = source_list + target_list
                final_target = target_list + source_list
                indexes = [final_source, final_target]
                final_orig_source_list = orig_source_list + orig_target_list
                final_orig_target_list = orig_target_list + orig_source_list
                edge_attrs = edge_attrs + edge_attrs
                all_indexes = torch.tensor(indexes).long()
                edge_attrs = torch.tensor(edge_attrs).float()

                if self.normalized:
                    if all_indexes.shape[1] != 0:
                        num_sent = torch.max(all_indexes[0].max(), all_indexes[1].max()) + 1
                        adj_matrix = torch.zeros(num_sent, num_sent)
                        for i in range(len(all_indexes[0])):
                            adj_matrix[all_indexes[0][i], all_indexes[1][i]] = edge_attrs[i]
                        for row in range(len(adj_matrix)):
                            adj_matrix[row] = adj_matrix[row] / adj_matrix[row].max()

                        temp = adj_matrix.reshape(-1)
                        edge_attrs = temp[temp.nonzero()].reshape(-1)

                node_fea = torch.tensor(node_fea).float()
                generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=edge_attrs,
                                      y=torch.tensor(label).int())

                try:
                    generated_data.article_id = torch.tensor(all_article_identifiers[ide])
                except:
                    generated_data.article_id = all_article_identifiers[ide]

                generated_data.orig_edge_index = torch.tensor([final_orig_source_list, final_orig_target_list]).long()

                if generated_data.has_isolated_nodes():
                    print("Error in graph -- isolated nodes detected")
                    print(generated_data)
                    print(generated_data.edge_index)
                elif generated_data.contains_self_loops():
                    print("Error in graph -- self loops detected")
                    print(generated_data)
                    print(generated_data.edge_index)

                if self.mode == "test":
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
                elif self.mode == "val":
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_val_{ide}.pt'))
                else:
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))

                ide += 1

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'), weights_only=False)
        elif self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'), weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

        return data



class UnifiedAttentionGraphs_Sum(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, degree=0.5, model_ckpt="", mode="train",
                 transform=None, normalized=False, binarized=False, multi_layer_model=False,
                 pre_transform=None):
        """
        root = Where the graph dataset should be stored. This folder is split into raw_dir (featured dataset) and processed_dir (processed graph data).
        filename = CSV file with the dataset to be processed. It should contain the following columns: article_id, label, doc_as_ids.
                    A row example would be: 0, 1, "[32603, 32604, 32611, 32612, 32613, 0, 0, 0, 0, 0]" for a document with article_id=0, label=1 and 5 sentences.
        filter_type = type of filtering to be applied to the attention matrices. If None, no filtering is applied.
        data_loader = data loader for MHA model forward. It should be a list of batches.
        degree = tolerance degree of filtering to be applied to the attention matrices. Preliminary experiments suggest a value of 0.5.
        model_ckpt = path to the trained MHA-based model checkpoint to be loaded --- The models is required to forward the data and obtain the corresponding attention matrices.
        mode = "train", "val" or "test" - to specify the mode of the graph dataset.
        transform = default None. Do not change.
        normalized = if True, the edges are normalized (divided by the maximum value of each row) instead of using the original attention weights as edge attributes.
        binarized = if True, the edges are binarized (1.0) instead of using the attention weights as edge attributes.
        pre_transform = default None. Do not change.
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
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
        super(UnifiedAttentionGraphs_Sum, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename                                                                        # CSV filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.mode == "test":
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.mode == "val":
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]


    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        all_doc_as_ids = self.data['doc_as_ids']
        all_labels = self.data['label']
        all_article_identifiers = self.data['article_id']
        all_batches = self.data_loader

        print("Loading MHASummarizer model...")
        model_lightning = MHASummarizer.load_from_checkpoint(self.model_ckpt)
        model_window = model_lightning.window
        max_len = model_lightning.max_len
        print("Model correctly loaded.")

        if self.mode == "test":
            print("\n[TEST] Creating Graphs Objects...")
        elif self.mode == "val":
            print("\n[VAL] Creating Graphs Objects...")
        else:
            print("\n[TRAIN] Creating Graphs Objects...")

        ide = 0
        for batch_sample in tqdm(all_batches, total=len(all_batches)):
            pred_batch, full_matrix_batch = model_lightning.predict_single(batch_sample)

            for pred, full_matrix in zip(pred_batch, full_matrix_batch):
                doc_ids = all_doc_as_ids[ide]
                label = all_labels[ide]
                doc_ids = [int(element) for element in doc_ids[1:-1].split(",")]
                doc_ids = torch.tensor(doc_ids)
                valid_sents = len(label)
                valid_sents = min(valid_sents, max_len)

                try:
                    cropped_doc = doc_ids[:valid_sents]
                except:
                    cropped_doc = doc_ids

                if self.filter_type is not None:
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window,
                                                       degree_std=self.K, with_filtering=True, filtering_type=self.filter_type)
                else:
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window,
                                                       degree_std=self.K, with_filtering=False)

                """"calculating edges"""
                match_ids = {k: v.item() for k, v in zip(range(len(filtered_matrix)), cropped_doc)} #key pos-i, value orig. sentence-id

                final_label = []
                processed_ids = []
                source_list = []
                target_list = []
                orig_source_list = []
                orig_target_list = []
                edge_attrs = []
                all_edges= []

                final = []
                for elem in cropped_doc:
                    if elem.item() not in final:
                        final.append(elem.item())

                dict_orig_to_ide_graph = {k: v for k, v in zip(final, range(len(final)))}           #key orig. sentence-id, value node-id for PyGeo


                for i in range(len(filtered_matrix)):
                    only_one_entry= False
                    fix_with_extra_edges = False
                    if filtered_matrix[i].count_nonzero().item()==1:
                        only_one_entry = True

                    for j in range(len(filtered_matrix)):
                        if filtered_matrix[i,j]!=0 and i!=j:                                        #no self loops
                            a= match_ids[i]                                                         #check match original ids
                            b= match_ids[j]
                            if a!=b:
                                if (a,b) not in all_edges:                                          #do not aggregate an edge if matched source and target are the same sentence
                                    orig_source_list.append(a)
                                    orig_target_list.append(b)
                                    all_edges.append((a,b))
                                    #has_edges = True

                                    if a in processed_ids:                                          #adapt id to the next position in processed
                                        source_list.append(processed_ids.index(a))                  #requires int from 0 to len
                                    else:
                                        source_list.append(dict_orig_to_ide_graph[a])               #requires int from 0 to len

                                    if b in processed_ids:                                          #adapt id to the next position in processed
                                        target_list.append(processed_ids.index(b))                  #requires int from 0 to len
                                    else:
                                        target_list.append(dict_orig_to_ide_graph[b])               #requires int from 0 to len

                                    if self.binarized:
                                        edge_attrs.append(1.0)
                                    else:
                                        edge_attrs.append(filtered_matrix[i,j].item())              #attention weight: as calculated by MHA trained model

                                else:                                                               #(a,b) in all_edges:
                                    pos_edge = all_edges.index((a,b))                               #check update edge weights
                                    if edge_attrs[pos_edge] < filtered_matrix[i,j].item():
                                        edge_attrs[pos_edge]= filtered_matrix[i,j].item()
                            else:
                                #if only_one_entry == True:
                                fix_with_extra_edges = True


                        if filtered_matrix[i,j]!=0 and i==j and (only_one_entry==True):             #check self-loop potentialy producing disconnected graphs
                            fix_with_extra_edges = True

                        if fix_with_extra_edges== True:
                            if match_ids[i] not in orig_target_list:
                                try:                                                                #add edges to previous and/or next neighbor
                                    b = match_ids[j-1]
                                    a = match_ids[i]
                                    if self.binarized:
                                        new_weight = 1.0
                                    else:
                                        new_weight = filtered_matrix[i,j].item()/2
                                    orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted = solve_by_creating_edge(a,b, orig_source_list, orig_target_list, all_edges, source_list, target_list,
                                                                                                                                                edge_attrs, processed_ids, dict_orig_to_ide_graph, new_weight)
                                except:
                                    pass

                                try:
                                    b = match_ids[j+1]
                                    a = match_ids[i]
                                    if self.binarized:
                                        new_weight = 1.0
                                    else:
                                        new_weight = filtered_matrix[i,j].item()/2
                                    orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted = solve_by_creating_edge(a,b, orig_source_list, orig_target_list, all_edges, source_list, target_list,
                                                                                                                                                edge_attrs, processed_ids, dict_orig_to_ide_graph, new_weight)
                                except:
                                    pass

                    if match_ids[i] not in processed_ids:
                        processed_ids.append(match_ids[i])
                        final_label.append(label[i])

                if len(source_list)!=len(orig_source_list) or len(orig_source_list)!=len(edge_attrs):
                    print ("Error when creating edges -- source and edge do not match")
                    print ("Number of edges acummulated:")
                    print ("source:", len(source_list))
                    print ("target:", len(target_list))
                    print ("orig_source:", len(orig_source_list))
                    print ("orig_target:", len(orig_target_list))
                    print ("edge_attrs:", len(edge_attrs))
                    break

                """"calculating node features"""
                node_fea=self.sent_model.encode(retrieve_from_dict(model_lightning.invert_vocab_sent, torch.tensor(processed_ids)))

                #reverse edges for undirected graphs
                final_source = source_list+target_list
                final_target = target_list+source_list
                indexes=[final_source,final_target]
                final_orig_source_list = orig_source_list+orig_target_list
                final_orig_target_list = orig_target_list+orig_source_list
                edge_attrs = edge_attrs+edge_attrs
                all_indexes=torch.tensor(indexes).long()
                edge_attrs = torch.tensor(edge_attrs).float()

                if self.normalized:
                    if all_indexes.shape[1] != 0:
                        num_sent= torch.max(all_indexes[0].max(), all_indexes[1].max()) + 1
                        adj_matrix = torch.zeros(num_sent, num_sent)
                        for i in range(len(all_indexes[0])):
                            adj_matrix[all_indexes[0][i], all_indexes[1][i]] = edge_attrs[i]
                        for row in range(len(adj_matrix)):
                            adj_matrix[row] = adj_matrix[row]/adj_matrix[row].max()

                        temp = adj_matrix.reshape(-1)
                        edge_attrs = temp[temp.nonzero()].reshape(-1)

                node_fea=torch.tensor(node_fea).float()
                generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=edge_attrs,
                                      y=torch.tensor(final_label).int())

                try:
                    generated_data.article_id = torch.tensor(all_article_identifiers[ide])
                except:
                    generated_data.article_id = all_article_identifiers[ide]

                generated_data.orig_edge_index = torch.tensor([final_orig_source_list,final_orig_target_list]).long()

                if generated_data.has_isolated_nodes():
                    print ("Error in graph -- isolated nodes detected")
                    print (generated_data)
                    print (generated_data.edge_index)
                elif generated_data.contains_self_loops():
                    print ("Error in graph -- self loops detected")
                    print (generated_data)
                    print (generated_data.edge_index)

                if self.mode == "test":
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
                elif self.mode == "val":
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_val_{ide}.pt'))
                else:
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))

                ide+=1

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'), weights_only=False)
        elif self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'), weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

        return data
