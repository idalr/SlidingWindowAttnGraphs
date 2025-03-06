# half from test_train_GNN.py with class definition of AttentionGraphs (from graph_data_loaders.py)
# debugging to create the correct graph, starting from full attention graphs


import yaml
import argparse
import torch
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from gnn_model import GAT_model, GCN_model, partitions
from base_model import MHAClassifier
from eval_models import retrieve_parameters, eval_results
from preprocess_data import load_data
from data_loaders import create_loaders, get_class_weights

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score

from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from eval_models import filtering_matrices, get_threshold, clean_tokenization_sent, filtering_matrix

################################## CLASS DEFINITION
class AttentionGraphs_debug(Dataset):
    def __init__(self, root, filename, filter_type, input_matrices, path_invert_vocab_sent='', degree=0, test=False, transform=None, normalized=False, pre_transform=None): #filename is df_raw del dataset
        ### df_train, df_test, max_len, batch_size tambien en init?
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        # TODO: also explore relative dir
        self.test = test
        self.filename = filename #a crear post predict
        self.filter_type = filter_type
        self.K = degree
        self.input_matrices = input_matrices
        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        self.normalized = normalized
        super(AttentionGraphs_debug, self).__init__(root, transform, pre_transform)

    # TODO: also explore relative dir
    # class AttentionGraphs(Dataset):
    #     def __init__(self, root, filename, filter_type, input_matrices, path_invert_vocab_sent='', degree=0, test=False,
    #                  transform=None, normalized=False, pre_transform=None):
    #         self.test = test
    #         self.filename = filename
    #         self.filter_type = filter_type
    #         self.K = degree
    #         self.input_matrices = input_matrices
    #
    #         # Ensure the root is an absolute path
    #         self.root = os.path.abspath(root)
    #         # Create the full path for the filename (relative to the root folder)
    #         raw_folder_path = os.path.join(self.root, "raw")
    #         self.filename = os.path.join(raw_folder_path, filename)
    #         # Normalize paths to ensure correct separators on different OS (Windows/Unix)
    #         self.filename = os.path.normpath(self.filename)
    #
    #         # Check if the raw folder exists
    #         if not os.path.exists(raw_folder_path):
    #             raise FileNotFoundError(f"Raw folder {raw_folder_path} not found!")
    #         # Check if the file exists at the computed path
    #         if not os.path.exists(self.filename):
    #             raise FileNotFoundError(f"File {self.filename} not found!")
    #
    #         # Handling path for vocab_sentences.csv
    #         if path_invert_vocab_sent:
    #             vocab_sentences_path = os.path.join(path_invert_vocab_sent, "vocab_sentences.csv")
    #         else:
    #             # If path_invert_vocab_sent is empty, we look for the file in the same directory as `this.csv`
    #             vocab_sentences_path = os.path.join(raw_folder_path, "vocab_sentences.csv")
    #         # Normalize the vocab_sentences path
    #         vocab_sentences_path = os.path.normpath(vocab_sentences_path)
    #         # Read the CSV and process the data
    #         #        if not os.path.exists(vocab_sentences_path):
    #         #            raise FileNotFoundError(f"Vocab Sentences file not found at {vocab_sentences_path}")
    #
    #         # checl raw path
    #         print('raw paths', self.raw_paths)
    #         print('raw paths 0', self.raw_paths[0])
    #
    #         sent_dict_disk = pd.read_csv(vocab_sentences_path)
    #         self.invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
    #         self.normalized = normalized
    #
    #         # Call the superclass initialization
    #         super(AttentionGraphs, self).__init__(self.root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        return self.filename  # +".csv"


    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
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
                                                                                    list_valid_sents, degree_std=self.K,
                                                                                    with_filtering=True,
                                                                                    filtering_type=self.filter_type)
        else:
            filtered_matrices, total_nodes, total_edges = filtering_matrices(self.input_matrices, all_article_identifiers,
                                                                         list_valid_sents, degree_std=self.K,
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

            # try:
            #     valid_sents= (doc_ids == 0).nonzero()[0]
            #     cropped_doc = doc_ids[:valid_sents]
            # except:
            #     valid_sents = len(doc_ids)
            #     cropped_doc = doc_ids

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
                        edge_attrs.append(filtered[
                                          i, j].item())  # attention weight -- as calculated by trained model -- TRY NORMALIZATION

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


################################## Creating graphs from existing data

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"]= "0" #config_file["cuda_visible_devices"]
logger_name = "df_logger_cw.csv" # config_file["logger_name"] #"df_logger_cw.csv"
model_name = "Extended_NoTemp_100" # config_file["model_name"]  #Extended_Anneal
type_model = "GAT" # config_file["type_model"] #GAT or GCN

root_graph = "AttnGraphs_HND/" # config_file["data_paths"]["root_graph_dataset"] #"/scratch/datasets/AttnGraphs_HND/"
path_results = "AttGraphs/GNN_Results/" # config_file["data_paths"]["results_folder"] #"/home/mbugueno/AttGraphs/GNN_Results/" #folder for GNN results (Heuristic_Results for baselines)
path_logger = "HomoGraphs_HND/" # config_file["data_paths"]["path_logger"]  #path_logger = "/scratch/mbugueno/HomoGraphs_HND/"

if not os.path.exists(root_graph):
    os.makedirs(root_graph)  # Creates the folder
    print(f"Folder '{root_graph}' created.")

if not os.path.exists(path_results):
    os.makedirs(path_results)  # Creates the folder
    print(f"Folder '{path_results}' created.")

if not os.path.exists(path_logger):
    os.makedirs(path_logger)  # Creates the folder
    print(f"Folder '{path_logger}' created.")

    
df_logger = pd.read_csv(path_logger+logger_name)
path_checkpoint, model_score = retrieve_parameters(model_name, df_logger)

file_to_save = model_name+"_"+str(model_score)[:5]
type_graph = "full" #config_file["type_graph"] #full, mean, max
path_models = path_logger+model_name+"/"
path_root= os.path.join(root_graph, model_name, "Attention", type_graph) #root_graph+model_name+"/Attention/"+type_graph
project_name= model_name+"2"+type_model+"_"+type_graph
file_results = path_results+file_to_save+"_2"+type_model+"_"+type_graph
            
filename="post_predict_train_documents.csv"
filename_test= "post_predict_test_documents.csv"

# TODO: problem with AttentionGraph
## calling already processed files shouldn't need path/to/vocab, so none has to be passed, just have to call data in /processed/data.pt
## if no processed files, then call the vocab and process data from /raw/data.csv
## have to dig into torch_geometric.Dataset to properly define class property

######################################################### run either try or except, but both has to work

################ try
# TODO: look at the current error
##  FileNotFoundError: [Errno 2] No such file or directory: 'vocab_sentences.csv'
###dataset = AttentionGraphs_debug(root=path_root, filename=filename, filter_type="", input_matrices=None, path_invert_vocab_sent='', degree=0.5, test=False)
###dataset_test = AttentionGraphs_debug(root=path_root, filename=filename_test, filter_type="", input_matrices=None, path_invert_vocab_sent='', degree=0.5, test=True)

################ except
print ("Type graph:", type_graph)
print ("Root graph:", path_root)
print ("Loading data...")

df_full_train, df_test = load_data("datasets/HyperNews/", "articles-training-byarticle-20181122.xml", "ground-truth-training-byarticle-20181122.xml",
                                   "articles-test-byarticle-20181207.xml", "ground-truth-test-byarticle-20181207.xml")

sent_lengths=[]
for i, doc in enumerate(df_full_train['article_text']):
    sent_in_doc = sent_tokenize(doc)
    if len(sent_in_doc)==0:
        print ("Empty doc en:", i)
    sent_lengths.append(len(sent_in_doc))
max_len = max(sent_lengths)  # Maximum number of sentences in a document

############################################################################ mini run
df_full_train = df_full_train.head(40)
df_test = df_test.head(40)
sent_lengths = sent_lengths[:40]
##sent_lengths_test = sent_lengths_test[:50]
############################################################################# mini run

batch_size = 32
loader_train, loader_test, _, _ = create_loaders(df_full_train, df_test, max_len, batch_size, with_val=False, task="classification",
                                                                                               tokenizer_from_scratch=False, path_ckpt="datasets/HyperNews/")

print ("\nLoading", model_name, "({0:.3f}".format(model_score),") from:", path_checkpoint)
model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
print ("Done")

preds_t, full_attn_weights_t, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(loader_train, cpu_store=False)
preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(loader_test, cpu_store=False)
print("Done with predicting")


path_dataset_file = os.path.abspath(os.path.join(path_root,"raw",filename))
path_dataset_file_test = os.path.abspath(os.path.join(path_root,"raw",filename_test))
print ("\nCreating files for PyG dataset in:", path_root+'/raw/')

post_predict_train_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
post_predict_train_docs.to_csv(path_dataset_file, index=False)
for article_id, label, doc_as_ids in zip(all_article_identifiers_t, all_labels_t, all_doc_ids_t):
    post_predict_train_docs.loc[len(post_predict_train_docs)] = {
    "article_id": article_id.item(),
    "label": label.item(),
    "doc_as_ids": doc_as_ids.tolist()
    }
post_predict_train_docs.to_csv(path_dataset_file, index=False)
print ("Finished and saved in:", path_dataset_file)

post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
post_predict_test_docs.to_csv(path_dataset_file_test, index=False)
for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test, all_doc_ids_test):
    post_predict_test_docs.loc[len(post_predict_test_docs)] = {
    "article_id": article_id.item(),
    "label": label.item(),
    "doc_as_ids": doc_as_ids.tolist()
    }
post_predict_test_docs.to_csv(path_dataset_file_test, index=False)
print ("Finished and saved in:", path_dataset_file_test)

#### create data.pt
if type_graph=="full":
        filter_type=None
else:
    filter_type=type_graph

# normalization
normalized = False # True

# TODO: current error
##  FileNotFoundError: [Errno 2] No such file or directory: '/AttnGraphs_HND\\Extended_NoTemp_100\\Attention\\full\\raw\\post_predict_train_documents.csv'
start_creation = time.time()
dataset = AttentionGraphs_debug(root=path_root, filename=filename, filter_type=filter_type, input_matrices=full_attn_weights_t,
                              path_invert_vocab_sent="datasets/HyperNews/", degree=0.5, normalized=normalized, test=False)
creation_train = time.time()-start_creation
start_creation = time.time()
dataset_test = AttentionGraphs_debug(root=path_root, filename=filename_test, filter_type=filter_type, input_matrices=full_attn_weights_test,
                                   path_invert_vocab_sent="datasets/HyperNews/", degree=0.5, normalized=normalized, test=True)
creation_test = time.time()-start_creation

print('Creation Time for train data:', creation_train)
print('Creation Time for test data:', creation_test)

