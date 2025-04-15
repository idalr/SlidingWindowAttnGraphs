import gc
from torch_geometric.data import Data, Dataset
import torch
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from eval_models import filtering_matrices, get_threshold, clean_tokenization_sent, filtering_matrix
from base_model import MHASummarizer_extended, MHASummarizer, retrieve_from_dict, MHAClassifier
from utils import solve_by_creating_edge


class AttentionGraphs(Dataset):
    def __init__(self, root, filename, filter_type, input_matrices, path_invert_vocab_sent='', window='', degree=0, test=False, transform=None, normalized=False, pre_transform=None): #filename is df_raw del dataset
        ### df_train, df_test, max_len, batch_size tambien en init?
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename #a crear post predict
        self.filter_type = filter_type
        self.K = degree
        self.input_matrices = input_matrices
        if path_invert_vocab_sent:
            sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
            self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        self.window = window
        self.normalized = normalized

        # Ensure the root is an absolute path
        self.root = os.path.abspath(root)
        # Create the full path for the filename (relative to the root folder)
        raw_folder_path = os.path.join(self.root, "raw")
        self.filename = os.path.join(raw_folder_path, filename)
        # Normalize paths to ensure correct separators on different OS (Windows/Unix)
        self.filename = os.path.normpath(self.filename)

        super(AttentionGraphs, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        """ list of files in the raw_dir which needs to be found in order to skip the download."""
        return self.filename  # +".csv"


    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        all_doc_as_ids = self.data['doc_as_ids'].apply(pd.eval)
        all_labels = self.data['label']
        all_article_identifiers = self.data['article_id']

        list_valid_sents = [(doc_as_ids != 0).sum() for doc_as_ids in all_doc_as_ids]

        if self.filter_type is not None:
            filtered_matrices, total_nodes, total_edges, deletions = filtering_matrices(self.input_matrices,
                                                                                    all_article_identifiers,
                                                                                    list_valid_sents, self.window, degree_std=self.K,
                                                                                    with_filtering=True,
                                                                                    filtering_type=self.filter_type)
        else:
            filtered_matrices, total_nodes, total_edges = filtering_matrices(self.input_matrices, all_article_identifiers,
                                                                         list_valid_sents, self.window, degree_std=self.K,
                                                                         with_filtering=False)

        if self.test:
            print("[Test] Creating Graphs Objects...")
        else:
            print("[Train] Creating Graphs Objects...")

        ide = 0
        for doc_ids, filtered, label in tqdm(zip(all_doc_as_ids, filtered_matrices, all_labels), total=len(all_doc_as_ids)):

            # doc_ids= [int(element) for element in doc_ids[1:-1].split(",")]
            # TODO: ask Margarita about this
            ## doc_ids= [element for element in doc_ids[1:-1]] # would lose its first token!
            doc_ids = [element for element in doc_ids if element != 0]  # array to list
            doc_ids = torch.tensor(doc_ids)  # list to tensor

            no_valid_sents = list_valid_sents[ide]
            cropped_doc = doc_ids[:no_valid_sents]

            """"calculating node features"""
            sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            for name, param in sent_model.named_parameters():
                param.requires_grad = False

            node_fea = []
            for sent in cropped_doc:
                node_fea.append(sent_model.encode(self.invert_vocab_sent[sent.item()]))

            """"calculating edges"""
            # for id in cropped_doc:
            match_ids = {k: v.item() for k, v in zip(range(len(filtered)), cropped_doc)}

            source_list = []
            target_list = []
            orig_source_list = []
            orig_target_list = []
            edge_attrs = []
            for i in range(len(filtered)):
                for j in range(len(filtered)):
                    if filtered[i, j] != 0 and i != j:  # no self loops
                        orig_source_list.append(match_ids[i])
                        orig_target_list.append(match_ids[j])
                        source_list.append(i)  # requires int from 0 to len
                        target_list.append(j)  # requires int from 0 to len
                        edge_attrs.append(filtered[i, j].item())  # attention weight -- as calculated by trained model -- TRY NORMALIZATION

            # reverse edges for all heuristics
            final_source = source_list + target_list
            final_target = target_list + source_list
            indexes = [final_source, final_target]
            edge_attrs = edge_attrs + edge_attrs  ### for reverse edge attributes (same weight - simetric matrix)
            # indexes=[source_list,target_list]

            all_indexes = torch.tensor(indexes).long()
            edge_attrs = torch.tensor(edge_attrs).float()

            if self.normalized:
                """try:
                    print ("Normalizing edge en origiwn", all_indexes[0].max(), "en target", all_indexes[1].max())
                    print ("Normalizing edge attributes", torch.max(all_indexes[0].max(), all_indexes[1].max()) + 1)
                except:
                    print ("matriz cropped:", cropped_doc)
                    print ("in dictionary 2 text:", self.invert_vocab_sent[cropped_doc[0].item()])
                    print ("doc ide article:", all_article_identifiers[ide])
                    print ("final_source", final_source)
                    print ("final_target", final_target)
                    print ("all_indexes", all_indexes.shape[1])
                """
                if all_indexes.shape[1] != 0:
                    num_sent = torch.max(all_indexes[0].max(), all_indexes[1].max()) + 1
                    adj_matrix = torch.zeros(num_sent, num_sent)
                    for i in range(len(all_indexes[0])):
                        adj_matrix[all_indexes[0][i], all_indexes[1][i]] = edge_attrs[i]
                    for row in range(len(adj_matrix)):
                        adj_matrix[row] = adj_matrix[row] / adj_matrix[row].max()

                    temp = adj_matrix.reshape(-1)
                    edge_attrs = temp[temp.nonzero()].reshape(-1)

                else:
                    pass

            node_fea = torch.tensor(node_fea).float()
            generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=edge_attrs, y=torch.tensor(label).int())
            generated_data.article_id = torch.tensor(all_article_identifiers[ide])
            generated_data.orig_edge_index = torch.tensor([orig_source_list, orig_target_list]).long()

            if self.test:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
            else:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))
            ide += 1

            del generated_data, node_fea
            torch.cuda.empty_cache()
            gc.collect()

    def len(self):
        return self.data.shape[0]   ##tamaño del dataset

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
        return data


##llamar con loader usando batch size 1
class UnifiedAttentionGraphs_Class(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, window, degree=0.5, model_ckpt="", mode="train",
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
        self.window = window
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
        super(UnifiedAttentionGraphs_Class, self).__init__(root, transform, pre_transform)

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
        all_article_identifiers = self.data['article_id']
        all_batches = self.data_loader


        print("Loading MHAClassifier model...")
        model_lightning = MHAClassifier.load_from_checkpoint(self.model_ckpt)
        model_window = model_lightning.window
        print("Model correctly loaded.")

        if self.mode == "test":
            print("\n[TEST] Creating Graphs Objects...")
        elif self.mode == "val":
            print("\n[VAL] Creating Graphs Objects...")
        else:
            print("\n[TRAIN] Creating Graphs Objects...")

        ide = 0
        for batch_sample in tqdm(all_batches, total=len(all_batches)):
            pred_batch, full_matrix_batch = model_lightning.predict_single(
                batch_sample)  ### este no existe en classifier .-- adaptarlo!

            for pred, full_matrix in zip(pred_batch, full_matrix_batch):
                doc_ids = all_doc_as_ids[ide]
                label = all_labels[ide]
                # label = clean_tokenization_sent(labels_sample, "label")

                doc_ids = [int(element) for element in doc_ids[1:-1].split(",")]
                doc_ids = torch.tensor(doc_ids)

                try:
                    valid_sents = (doc_ids == 0).nonzero()[0]
                    cropped_doc = doc_ids[:valid_sents]
                except:
                    valid_sents = len(doc_ids)
                    cropped_doc = doc_ids

                if self.filter_type is not None:
                    # filtered_matrix = filtering_matrix(full_matrix[0], valid_sents=valid_sents, degree_std=self.K, with_filtering=True, filtering_type=self.filter_type)
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window, degree_std=self.K,
                                                       with_filtering=True, filtering_type=self.filter_type)
                else:
                    # filtered_matrix = filtering_matrix(full_matrix[0], valid_sents=valid_sents, degree_std=self.K, with_filtering=False)
                    filtered_matrix = filtering_matrix(full_matrix, valid_sents=valid_sents, window=model_window, degree_std=self.K,
                                                       with_filtering=False)

                """"calculating edges"""
                match_ids = {k: v.item() for k, v in
                             zip(range(len(filtered_matrix)), cropped_doc)}  # key pos-i, value orig. sentence-id

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

                                    if a in processed_ids:  # adaptar ide a la posicion del processed
                                        source_list.append(processed_ids.index(a))  # requires int from 0 to len
                                    else:
                                        source_list.append(dict_orig_to_ide_graph[a])  # requires int from 0 to len

                                    if b in processed_ids:  # adaptar id a posicion en processed
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
                                try:  # Adding edges to previous neighbor
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
                        # final_label.append(label[i])

                del match_ids, all_edges, dict_orig_to_ide_graph, filtered_matrix, final
                gc.collect()

                if len(source_list) != len(orig_source_list) or len(orig_source_list) != len(edge_attrs):
                    print("Error in edge creation -- source and edge don't match")
                    print("numero de edges acummulados:")
                    print("source:", len(source_list))
                    print("target:", len(target_list))
                    print("orig_source:", len(orig_source_list))
                    print("orig_target:", len(orig_target_list))
                    print("edge_attrs:", len(edge_attrs))
                    break

                """"calculating node features"""  # processed_ids === dict_orig_to_ide_graph.keys()
                node_fea = self.sent_model.encode(
                    retrieve_from_dict(model_lightning.invert_vocab_sent, torch.tensor(processed_ids)))

                # reverse edges for all heuristics
                final_source = source_list + target_list
                final_target = target_list + source_list
                indexes = [final_source, final_target]
                final_orig_source_list = orig_source_list + orig_target_list
                final_orig_target_list = orig_target_list + orig_source_list
                edge_attrs = edge_attrs + edge_attrs  ### for reverse edge attributes (same weight - simetric matrix)

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
                                      y=torch.tensor(label).int())  # label in original graphs class
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
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'), weights_only=False)
                if self.mode == "val":
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_val_{ide}.pt'), weights_only=False)
                else:
                    torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'), weights_only=False)

                ide += 1

                del generated_data, node_fea
                torch.cuda.empty_cache()
                gc.collect()

    def len(self):
        return self.data.shape[0]  ##tamaño del dataset

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        if self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

        return data


###################### Heuristic Graphs ######################
# rules for constructing the graphs:
# order: Each sentence is connected to the next sentence.
# window: Each sentence is connected to every sentence in a window of size ws.
# semantic: Each sentence is connected to the closest sentences, considering a min threshold.
###### require: 
# path a MHA trained model for loading invert dictionary (ids a sentence texts) when semantic is specified as mode

class HeuristicGraphs(Dataset):
    def __init__(self, root, filename, heuristic, path_invert_vocab_sent='', window_size=2, mode="train", task="classification", transform=None, pre_transform=None): 
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.task = task
        self.mode = mode
        self.filename = filename   
        self.heuristic = heuristic
        self.window_size = window_size #forwards and backwards -- hops
        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #self.normalized = normalized
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False

        super(HeuristicGraphs, self).__init__(root, transform, pre_transform)
     
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename #+".csv"

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
        all_article_identifiers = self.data['article_id']

        if self.mode == "test":
            print ("\n[TEST] Creating Graphs Objects...")
        elif self.mode == "val":
            print ("\n[VAL] Creating Graphs Objects...")
        else: 
            print ("\n[TRAIN] Creating Graphs Objects...")

        ide=0
        for doc_ids, label in tqdm(zip(all_doc_as_ids, all_labels), total=len(all_doc_as_ids)):
            
            doc_ids= [int(element) for element in doc_ids[1:-1].split(",")]   ### filename list is saved as string
            doc_ids= torch.tensor(doc_ids)

            try:
                valid_sents= (doc_ids == 0).nonzero()[0]
                cropped_doc = doc_ids[:valid_sents]
            except:
                valid_sents = len(doc_ids)
                cropped_doc = doc_ids

            """"calculating node features"""
            # nodes are fixed - do not change with heuristic                 
            node_fea=[]
            for sent in cropped_doc:
                node_fea.append(self.sent_model.encode(self.invert_vocab_sent[sent.item()]))

            node_fea=torch.tensor(node_fea).float() 

            """"calculating edges"""
            #for id in cropped_doc:
            match_ids = {k: v.item() for k, v in zip(range(len(cropped_doc)),cropped_doc)}

            source_list = []
            target_list = []
            orig_source_list = []
            orig_target_list = []
            edge_attrs = []

            if self.heuristic == 'order':
                for i in range(len(cropped_doc)-1):
                    orig_source_list.append(match_ids[i])
                    orig_target_list.append(match_ids[i+1])
                    source_list.append(i)   # PyG requires int from 0 to len
                    target_list.append(i+1) # PyG requires int from 0 to len
                    #edge_attrs.append(1.0)

            elif self.heuristic == 'window':
                for i in range(len(cropped_doc)):
                    for j in range(i+1, min(i+self.window_size+1, len(cropped_doc))):
                        orig_source_list.append(match_ids[i])
                        orig_target_list.append(match_ids[j])
                        source_list.append(i) # PyG requires int from 0 to len
                        target_list.append(j) # PyG requires int from 0 to len
                        #edge_attrs.append(1.0)
                        
            elif self.heuristic == 'max_semantic':                
                #create similarity "matrix" among sentences from node features tensor -- solo para triangulo superior  
                sim_matrix= torch.zeros(len(cropped_doc),len(cropped_doc))
                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        sim_matrix[i,j]= torch.cosine_similarity(node_fea[i],node_fea[j],dim=0)
                
                # definir funcion de threshold minimo + filtering entries with low cosine similarity 
                _, _, _, threshold_min = get_threshold(sim_matrix, 0.5, type='max', mode='local')  
                for i in range(len(sim_matrix)):
                    sim_matrix[i]= torch.where(sim_matrix[i] < threshold_min[i], 0., sim_matrix[i])

                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        if sim_matrix[i,j]!=0:
                            orig_source_list.append(match_ids[i])
                            orig_target_list.append(match_ids[j])
                            source_list.append(i) # PyG requires int from 0 to len
                            target_list.append(j) # PyG requires int from 0 to len
                            edge_attrs.append(sim_matrix[i,j].item())  # cosine similarity

            elif self.heuristic == 'mean_semantic':                
                #create similarity "matrix" among sentences from node features tensor -- solo para triangulo superior  
                sim_matrix= torch.zeros(len(cropped_doc),len(cropped_doc))
                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        sim_matrix[i,j]= torch.cosine_similarity(node_fea[i],node_fea[j],dim=0)
                
                # definir funcion de threshold minimo + filtering entries with low cosine similarity 
                _, _, _, threshold_min = get_threshold(sim_matrix, 0.5, type='mean', mode='local')  
                for i in range(len(sim_matrix)):
                    sim_matrix[i]= torch.where(sim_matrix[i] < threshold_min[i], 0., sim_matrix[i])

                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        if sim_matrix[i,j]!=0:
                            orig_source_list.append(match_ids[i])
                            orig_target_list.append(match_ids[j])
                            source_list.append(i) # PyG requires int from 0 to len
                            target_list.append(j) # PyG requires int from 0 to len
                            edge_attrs.append(sim_matrix[i,j].item())  # cosine similarity

            else: 
                print ("Heuristic not defined - Aborting")
                return 
            
            #reverse edges for all heuristics            
            final_source = source_list+target_list
            final_target = target_list+source_list
            indexes=[final_source,final_target]       
            final_orig_source_list = orig_source_list+orig_target_list
            final_orig_target_list = orig_target_list+orig_source_list         
            all_indexes=torch.tensor(indexes).long()
            concat_edge_attrs = edge_attrs+edge_attrs  ### for reverse edge attributes (same weight - simetric matrix)
            final_edge_attrs = torch.tensor(concat_edge_attrs).float()          
            
            if self.task=="summarization":
                label = clean_tokenization_sent(label, "label")

            generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=final_edge_attrs, y=torch.tensor(label).int())
            generated_data.article_id = torch.tensor(all_article_identifiers[ide])
            generated_data.orig_edge_index = torch.tensor([final_orig_source_list,final_orig_target_list]).long()
            generated_data.article_id = torch.tensor(all_article_identifiers[ide])
            
            if self.mode == "test":
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
            if self.mode == "val":  
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_val_{ide}.pt'))
            else:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))
            
            ide+=1

    def len(self):
        return self.data.shape[0]   ##tamaño del dataset

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        if self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))        
            
        return data
    


##llamar con loader usando batch size 1
class AttentionGraphs_Sum(Dataset):
    def __init__(self, root, filename, filter_type, data_loader, degree=0.5, model_ckpt="", mode= "train", transform=None, normalized=False, binarized=False, multi_layer_model=False, pre_transform=None): #input_matrices, invert_vocab_sent
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
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
        super(AttentionGraphs_Sum, self).__init__(root, transform, pre_transform)
     
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename #+".csv"

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
        all_article_identifiers = self.data['article_id']
        all_batches = self.data_loader

        if self.multi_layer_mha:
            model_lightning = MHASummarizer_extended.load_from_checkpoint(self.model_ckpt)
        else: 
            model_lightning = MHASummarizer.load_from_checkpoint(self.model_ckpt)

        if self.mode == "test":
            print ("\n[TEST] Creating Graphs Objects...")
        elif self.mode == "val":
            print ("\n[VAL] Creating Graphs Objects...")
        else: 
            print ("\n[TRAIN] Creating Graphs Objects...")
        

        ide=0
        for doc_ids, batch_sample, labels_sample in tqdm(zip(all_doc_as_ids, all_batches, all_labels), total=len(all_doc_as_ids)):

            pred, full_matrix = model_lightning.predict_single(batch_sample)
            label = clean_tokenization_sent(labels_sample, "label")
            valid_sents = len(label)

            if self.filter_type is not None:
                filtered_matrix = filtering_matrix(full_matrix[0], valid_sents=valid_sents, degree_std=self.K, with_filtering=True, filtering_type=self.filter_type)
            else:
                filtered_matrix = filtering_matrix(full_matrix[0], valid_sents=valid_sents, degree_std=self.K, with_filtering=False)
                
            doc_ids= [int(element) for element in doc_ids[1:-1].split(",")]
            doc_ids= torch.tensor(doc_ids)

            try:
                #valid_sents= (doc_ids == 0).nonzero()[0]
                cropped_doc = doc_ids[:valid_sents]
            except:
                #valid_sents = len(doc_ids)
                cropped_doc = doc_ids

            """"calculating node features"""
            node_fea=self.sent_model.encode(retrieve_from_dict(model_lightning.invert_vocab_sent, cropped_doc))

            """"calculating edges"""
            #for id in cropped_doc:
            match_ids = {k: v.item() for k, v in zip(range(len(filtered_matrix)), cropped_doc)}

            source_list = []
            target_list = []
            orig_source_list = []
            orig_target_list = []
            edge_attrs = []
            for i in range(len(filtered_matrix)):
                for j in range(len(filtered_matrix)):
                    if filtered_matrix[i,j]!=0 and i!=j: #no self loops
                        orig_source_list.append(match_ids[i])
                        orig_target_list.append(match_ids[j])
                        source_list.append(i) #requires int from 0 to len
                        target_list.append(j) #requires int from 0 to len
                        if self.binarized:
                            edge_attrs.append(1.0)
                        else:
                            edge_attrs.append(filtered_matrix[i,j].item())  # attention weight -- as calculated by trained model -- TRY NORMALIZATION

            
            #reverse edges for all heuristics            
            final_source = source_list+target_list
            final_target = target_list+source_list
            indexes=[final_source,final_target]     
            final_orig_source_list = orig_source_list+orig_target_list
            final_orig_target_list = orig_target_list+orig_source_list
            edge_attrs = edge_attrs+edge_attrs ### for reverse edge attributes (same weight - simetric matrix)     

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

            else:
                pass

            node_fea=torch.tensor(node_fea).float() 
            generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=edge_attrs, y=torch.tensor(label).int())
            generated_data.article_id = torch.tensor(all_article_identifiers[ide])
            generated_data.orig_edge_index = torch.tensor([final_orig_source_list,final_orig_target_list]).long()
            
            if self.mode == "test":
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
            if self.mode == "val":  
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_val_{ide}.pt'))
            else:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))
            
            ide+=1


    def len(self):
        return self.data.shape[0]   ##tamaño del dataset

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset """
        if self.mode == "test":
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        if self.mode == "val":
            data = torch.load(os.path.join(self.processed_dir, f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))        
            
        return data
    


"""
class HeuristicGraphs(Dataset):
    def __init__(self, root, filename, heuristic, path_invert_vocab_sent='', window_size=2, test=False, transform=None, pre_transform=None): 
        
        #root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data). 
        
        self.test = test
        self.filename = filename   
        self.heuristic = heuristic
        self.window_size = window_size #forwards and backwards -- hops
        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #self.normalized = normalized
        super(HeuristicGraphs, self).__init__(root, transform, pre_transform)
     
    @property
    def raw_file_names(self):
        #"" If this file exists in raw_dir, the download is not triggered. ""
        return self.filename #+".csv"

    @property
    def processed_file_names(self):
        #"" If these files are found in raw_dir, processing is skipped""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() 
        all_doc_as_ids = self.data['doc_as_ids']
        all_labels = self.data['label']
        all_article_identifiers = self.data['article_id']

        if self.test:
            print ("[Test] Creating Graphs Objects...")
        else: 
            print ("[Train] Creating Graphs Objects...")

        ide=0
        for doc_ids, label in tqdm(zip(all_doc_as_ids, all_labels), total=len(all_doc_as_ids)):
            
            doc_ids= [int(element) for element in doc_ids[1:-1].split(",")]   ### filename list is saved as string
            doc_ids= torch.tensor(doc_ids)

            try:
                valid_sents= (doc_ids == 0).nonzero()[0]
                cropped_doc = doc_ids[:valid_sents]
            except:
                valid_sents = len(doc_ids)
                cropped_doc = doc_ids

            #""calculating node features""
            # nodes are fixed - do not change with heuristic 
            sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            for name, param in sent_model.named_parameters():
                param.requires_grad = False
                
            node_fea=[]
            for sent in cropped_doc:
                node_fea.append(sent_model.encode(self.invert_vocab_sent[sent.item()]))

            node_fea=torch.tensor(node_fea).float() 

            #""calculating edges""
            #for id in cropped_doc:
            match_ids = {k: v.item() for k, v in zip(range(len(cropped_doc)),cropped_doc)}

            source_list = []
            target_list = []
            orig_source_list = []
            orig_target_list = []
            edge_attrs = []

            if self.heuristic == 'order':
                for i in range(len(cropped_doc)-1):
                    orig_source_list.append(match_ids[i])
                    orig_target_list.append(match_ids[i+1])
                    source_list.append(i)   # PyG requires int from 0 to len
                    target_list.append(i+1) # PyG requires int from 0 to len
                    #edge_attrs.append(1.0)

            elif self.heuristic == 'window':
                for i in range(len(cropped_doc)):
                    for j in range(i+1, min(i+self.window_size+1, len(cropped_doc))):
                        orig_source_list.append(match_ids[i])
                        orig_target_list.append(match_ids[j])
                        source_list.append(i) # PyG requires int from 0 to len
                        target_list.append(j) # PyG requires int from 0 to len
                        #edge_attrs.append(1.0)
                        
            elif self.heuristic == 'max_semantic':                
                #create similarity "matrix" among sentences from node features tensor -- solo para triangulo superior  
                sim_matrix= torch.zeros(len(cropped_doc),len(cropped_doc))
                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        sim_matrix[i,j]= torch.cosine_similarity(node_fea[i],node_fea[j],dim=0)
                
                # definir funcion de threshold minimo + filtering entries with low cosine similarity 
                _, _, _, threshold_min = get_threshold(sim_matrix, 0.5, type='max', mode='local')  
                for i in range(len(sim_matrix)):
                    sim_matrix[i]= torch.where(sim_matrix[i] < threshold_min[i], 0., sim_matrix[i])

                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        if sim_matrix[i,j]!=0:
                            orig_source_list.append(match_ids[i])
                            orig_target_list.append(match_ids[j])
                            source_list.append(i) # PyG requires int from 0 to len
                            target_list.append(j) # PyG requires int from 0 to len
                            edge_attrs.append(sim_matrix[i,j].item())  # cosine similarity

            elif self.heuristic == 'mean_semantic':                
                #create similarity "matrix" among sentences from node features tensor -- solo para triangulo superior  
                sim_matrix= torch.zeros(len(cropped_doc),len(cropped_doc))
                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        sim_matrix[i,j]= torch.cosine_similarity(node_fea[i],node_fea[j],dim=0)
                
                # definir funcion de threshold minimo + filtering entries with low cosine similarity 
                _, _, _, threshold_min = get_threshold(sim_matrix, 0.5, type='mean', mode='local')  
                for i in range(len(sim_matrix)):
                    sim_matrix[i]= torch.where(sim_matrix[i] < threshold_min[i], 0., sim_matrix[i])

                for i in range(len(cropped_doc)):
                    for j in range(i+1, len(cropped_doc)):
                        if sim_matrix[i,j]!=0:
                            orig_source_list.append(match_ids[i])
                            orig_target_list.append(match_ids[j])
                            source_list.append(i) # PyG requires int from 0 to len
                            target_list.append(j) # PyG requires int from 0 to len
                            edge_attrs.append(sim_matrix[i,j].item())  # cosine similarity

            else: 
                print ("Heuristic not defined - Aborting")
                return 
            
            #reverse edges for all heuristics            
            final_source = source_list+target_list
            final_target = target_list+source_list
            indexes=[final_source,final_target]            
            all_indexes=torch.tensor(indexes).long()
            edge_attrs = edge_attrs+edge_attrs  ### for reverse edge attributes (same weight - simetric matrix)
            edge_attrs = torch.tensor(edge_attrs).float()          
            
            generated_data = Data(x=node_fea, edge_index=all_indexes, edge_attr=edge_attrs, y=torch.tensor(label).int())
            generated_data.article_id = torch.tensor(all_article_identifiers[ide])
            generated_data.orig_edge_index = torch.tensor([orig_source_list,orig_target_list]).long()
            
            if self.test:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_test_{ide}.pt'))
            else:
                torch.save(generated_data, os.path.join(self.processed_dir, f'data_{ide}.pt'))
            
            ide+=1

    def len(self):
        return self.data.shape[0]   ##tamaño del dataset

    def get(self, idx):
        #"" - Equivalent to __getitem__ in pytorch - Is not needed for PyG's InMemoryDataset ""
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
"""