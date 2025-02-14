import torch
import torch.nn as nn
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.utils import class_weight
from collections import Counter
from eval_models import filtering_matrices
from torch_geometric.data import Dataset, Data, DataLoader

"""
class DocumentDataset(Dataset):
    def __init__(self, documents, labels, max_len, article_identifiers, task="classification"): #tokenizer_sentence
        self.documents = documents
        self.labels = labels
        self.max_len = max_len
        self.article_id = article_identifiers
        self.task = task
        ### definir model pre trained y crear dataset con los embeddings de sentencias ya hechos! asi no hay strings problems  
        #self.tokenizer = tokenizer_sentence

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]
        article_id = self.article_id[idx]
        
        # Padding the document if it's shorter than max_len
        original_len = len(document)
        num_pad_values = np.maximum(self.max_len - original_len, 0)
        if num_pad_values > 0:
            document = document + [0] * num_pad_values  # Assume padding index is 0
            attention_mask = torch.tensor([0]*original_len + [-BIG_VAL]*num_pad_values)
            matrix_mask = torch.concat([attention_mask[None,:].repeat(original_len, 1), -BIG_VAL*torch.ones(num_pad_values, self.max_len)], dim=0)
        else:
            print ("--- cortando el documento en:", idx)
            document = document[:self.max_len]
            if self.task=="summarization": ##summarization task defines label as a list
                label = label[:self.max_len]
                print ("si se crtoooooo labels en:", idx)
            attention_mask = torch.zeros(self.max_len)
            matrix_mask = torch.zeros(self.max_len, self.max_len)        
        
        if self.task=="classification":
            return {'documents_ids': torch.tensor(document), 'labels': torch.tensor(label), 
                'src_key_padding_mask': attention_mask,
                "matrix_mask": matrix_mask, 'article_id': article_id}
        else:
            return {'documents_ids': torch.tensor(document), 'labels': torch.tensor(label+[0]*num_pad_values), #[-BIG_VAL]
                'src_key_padding_mask': attention_mask,
                "matrix_mask": matrix_mask, 'article_id': article_id}  ### summarization as sent-clasification -- labels with same length
"""

BIG_VAL = 1e9

class DocumentDataset(Dataset):
    def __init__(self, documents, labels, max_len, article_identifiers, task="classification"): #tokenizer_sentence
        self.documents = documents
        self.labels = labels
        self.max_len = max_len
        self.article_id = article_identifiers
        self.task = task
        ### definir model pre trained y crear dataset con los embeddings de sentencias ya hechos! asi no hay strings problems  
        #self.tokenizer = tokenizer_sentence

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]
        article_id = self.article_id[idx]
        
        original_len = len(document)     
        if self.task=="summarization":
            labels_len = len(label)
            if original_len != labels_len:
                print("Error in lengths of document and labels in dataframe row-id:", idx, "with label lens as ", labels_len, "and doc len as", original_len)          
        num_pad_values = np.maximum(self.max_len - original_len, 0)

        if num_pad_values > 0:
            document = document + [0] * num_pad_values  # Assume padding index is 0
            attention_mask = torch.tensor([0]*original_len + [-BIG_VAL]*num_pad_values) #torch.inf
            matrix_mask = torch.concat([attention_mask[None,:].repeat(original_len, 1), -BIG_VAL*torch.ones(num_pad_values, self.max_len)], dim=0) #torch.inf
        else:
            document = document[:self.max_len]
            if self.task=="summarization": ##summarization task defines label as a list
                label = label[:self.max_len]
            attention_mask = torch.zeros(self.max_len)
            matrix_mask = torch.zeros(self.max_len, self.max_len)        
        
        if self.task=="classification":
            return {'documents_ids': torch.tensor(document), 'labels': torch.tensor(label), 
                'src_key_padding_mask': attention_mask,
                "matrix_mask": matrix_mask, 'article_id': article_id}
        else:
            return {'documents_ids': torch.tensor(document), 'labels': torch.tensor(label+[-1]*num_pad_values), #[-BIG_VAL]
                'src_key_padding_mask': attention_mask,
                "matrix_mask": matrix_mask, 'article_id': article_id}  ### summarization as sent-clasification -- labels with same length
        

###binary classification
def get_class_weights(df_, task="classification"):
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    
    if task=="summarization":
        all_class_weights=torch.zeros(len(df_),2)
        counts={1:0, 0:0}

        for i, doc in enumerate(df_[label_column_name[task]]):
            labels = clean_tokenization_sent(doc, 'label')
            class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(torch.tensor(labels)), y=torch.tensor(labels).numpy())
            new= Counter(labels)
            for k in new.keys():
                counts[k]+=new[k]

            class_weights=torch.tensor(class_weights,dtype=torch.float)
            all_class_weights[i]=class_weights

        t_class_weights = all_class_weights.mean(axis=0)
    
    else:
        labels = df_[label_column_name[task]].tolist()
        class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(torch.tensor(labels)), y=torch.tensor(labels).numpy())
        t_class_weights= torch.tensor(class_weights,dtype=torch.float)
        counts = Counter(labels)
    
    return t_class_weights, counts 

"""def documents_to_ids(documents, from_scratch=True, dict_text_ids=None, invert_text_ids=None):
    # Assume documents is a list of sentences
    # Tokenize each document into sentences --- optionally: list of lists of words
    if from_scratch:
        dict_text_ids={}
        documents_as_ids = []
        i=1
        for doc in documents:
            doc_as_ids = []
            for sent in doc:
                if sent not in dict_text_ids:
                    dict_text_ids[sent]=i
                    i+=1 
                doc_as_ids.append(dict_text_ids[sent])
            documents_as_ids.append(doc_as_ids)
        
        invert_text_ids = {v: k for k, v in dict_text_ids.items()}

        return documents_as_ids, dict_text_ids, invert_text_ids

    else:
        modified = False
        documents_as_ids = []
        i =  max(invert_text_ids.keys())+1
        for doc in documents:
            doc_as_ids = []
            for sent in doc:
                if sent not in dict_text_ids:
                    #print ("Error: sentence not in dictionary")
                    modified = True
                    dict_text_ids[sent]=i
                    invert_text_ids[i]=sent
                    i+=1 
                doc_as_ids.append(dict_text_ids[sent])
            documents_as_ids.append(doc_as_ids)
        if modified:
            return documents_as_ids, dict_text_ids, invert_text_ids
        else: 
            return documents_as_ids
"""

def clean_tokenization_sent(list_as_document, object):    
    if object=="text":
        tok_document = list_as_document.split(' [St]')
        return tok_document
    else:
        tok_document = list_as_document.split(", ")
        tok_document[0] = tok_document[0][1:]
        tok_document[-1] = tok_document[-1][:-1]
        return [int(element) for element in tok_document]
    
def check_dataframe(df_):
    to_remove = []
    for i in range(len(df_)):
        articulos = df_["Cleaned_Article"][i]
        lista_lab= df_["Calculated_Labels"][i]
        if len(clean_tokenization_sent(articulos, 'text')) != len(clean_tokenization_sent(lista_lab, 'label')):
            #print ("Number of sentences and calculated labels do not match")
            #print("Removing entry ", df_.index[i], " - with Article ID:", df_["Article_ID"][i])
            to_remove.append(df_.index[i])
    
    return to_remove

def check_dimensions(dataset, max_allowed):
    print ("Checking dataset dimensions -- Max allowed:", max_allowed)

    for i in range(len(dataset)):
        sample = dataset[i]
        if len(sample['documents_ids']) > max_allowed:
            print ("More sentences than allowed in Dataset Object with id:", i)
            print ("Length of doc:", len(sample['documents_ids']))
            

        if len(sample['labels']) > max_allowed:
            print ("More labels than allowed in Dataset Object with id:", i)
            print ("Length of labels:", len(sample['labels']))
            
        else:
            continue
    print ("Dimensions are OK!")
    return


def documents_to_ids(documents, from_scratch=True, dict_text_ids=None, invert_text_ids=None, modifiable=False):
    # Assume documents is a list of sentences
    # Tokenize each document into sentences --- optionally: list of lists of words
    if from_scratch:
        dict_text_ids={}
        documents_as_ids = []
        i=1
        for doc in documents:
            doc_as_ids = []
            for sent in doc:
                if sent not in dict_text_ids:
                    dict_text_ids[sent]=i
                    i+=1 
                doc_as_ids.append(dict_text_ids[sent])
            documents_as_ids.append(doc_as_ids)
        
        invert_text_ids = {v: k for k, v in dict_text_ids.items()}
        
        return documents_as_ids, dict_text_ids, invert_text_ids

    else:
        if modifiable:
            modified = False
            documents_as_ids = []
            i =  max(invert_text_ids.keys())+1
            for doc in documents:
                doc_as_ids = []
                for sent in doc:
                    if sent not in dict_text_ids:
                        modified = True
                        dict_text_ids[sent]=i
                        invert_text_ids[i]=sent
                        i+=1 
                    doc_as_ids.append(dict_text_ids[sent])
                documents_as_ids.append(doc_as_ids)
            if modified:
                return documents_as_ids, dict_text_ids, invert_text_ids
            else: 
                return documents_as_ids, _, _ ##same and same
        else:
            documents_as_ids = []
            for doc in documents:
                doc_as_ids = []
                for sent in doc:
                    if sent not in dict_text_ids:
                        print ("Error: sentence not in dictionary")
                        print ("sentence:", sent)
                        doc_as_ids.append(0)
                    else:
                        doc_as_ids.append(dict_text_ids[sent])
                documents_as_ids.append(doc_as_ids)
            return documents_as_ids  ##dicts not modifiable
        

def create_loaders(df_full_train, df_test, max_len, batch_size, with_val=True, tokenizer_from_scratch=True, path_ckpt='', df_val=None, task="classification"): ### when val partition doesn't exist in original dataset

    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}

    if with_val:         ### if val doesn't exist, split training data
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2) 
        else:
            df_train = df_full_train

        documents = [sent_tokenize(doc) if task=="classification" else clean_tokenization_sent(doc, "text") for doc in list(df_train[source_column_name[task]])]
        documents_val = [sent_tokenize(doc) if task=="classification" else clean_tokenization_sent(doc, "text") for doc in list(df_val[source_column_name[task]])]
        if task=="classification":
            labels = list(df_train[label_column_name[task]])
            labels_val = list(df_val[label_column_name[task]])
        else: 
            labels = [clean_tokenization_sent(doc, "label") for doc in list(df_train[label_column_name[task]])]
            labels_val = [clean_tokenization_sent(doc, "label") for doc in list(df_val[label_column_name[task]])]

        ### es probable que los ides de articulos no sean unicos en diferentes particiones porque son el index del dataset loaded from HuggingFace
        article_identifiers = list(df_train[id_column_name[task]])  
        article_identifiers_val = list(df_val[id_column_name[task]])
    else:
        documents = [sent_tokenize(doc) if task=="classification" else clean_tokenization_sent(doc, "text") for doc in list(df_full_train[source_column_name[task]])]
        if task=="classification":  
            labels = list(df_full_train[label_column_name[task]])   
        else:
            labels = [clean_tokenization_sent(doc, "label") for doc in list(df_full_train[label_column_name[task]])] ## no debiese pasar con mis datasets
        article_identifiers = list(df_full_train[id_column_name[task]])
   
    documents_test = [sent_tokenize(doc) if task=="classification" else clean_tokenization_sent(doc, "text") for doc in list(df_test[source_column_name[task]])]
    if task=="classification":
        labels_test = list(df_test[label_column_name[task]])
    else:
        labels_test = [clean_tokenization_sent(doc, "label") for doc in list(df_test[label_column_name[task]])]
    article_identifiers_test = list(df_test[id_column_name[task]])

    if tokenizer_from_scratch:
        documents_ids, vocab_sent, invert_vocab_sent = documents_to_ids(documents) #train docs
        print ("Train dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
        if with_val:
            documents_ids_val, vocab_sent, invert_vocab_sent = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent, modifiable=True)
            print ("Train + Val dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
            dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task) 
            #print ("Val Dataset comprises:", len(dataset_val), "samples.")
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        documents_ids_test, vocab_sent, invert_vocab_sent = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent, modifiable=True)
        print ("Train + Val + Test dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")

    if not tokenizer_from_scratch:
        if task=="classification": ##change it for newly created models as I will sabve the dictionary on disk and load it from path
            try: 
                #print ("Loading sentence vocab from model checkpoint...")
                checkpoint = torch.load(path_ckpt)
                invert_vocab_sent = checkpoint['hyper_parameters']['invert_vocab_sent']
                #print ("Done")
            except:
                #print ("Loading sentence vocab from dataframe...")
                sent_dict_disk = pd.read_csv(path_ckpt+"vocab_sentences.csv")
                invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
                #print ("Done")
        else:
            #print ("Loading sentence vocab from dataframe")
            sent_dict_disk = pd.read_csv(path_ckpt+"vocab_sentences.csv")
            invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        
        vocab_sent = {v:k for k,v in invert_vocab_sent.items()}
        #print ("Loaded dictionary comprises", len(invert_vocab_sent.keys()), "sentences.")
        #print ("\nTrain...")
        documents_ids = documents_to_ids(documents, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
        if with_val:
            #print ("\nVal...")
            documents_ids_val = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
            dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val, task=task) 
            #print ("Val Dataset comprises:", len(dataset_val), "samples.")
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        #print ("\nTest...")
        documents_ids_test = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
        
    dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers, task=task) 
    #print ("Train Dataset comprises:", len(dataset_train), "samples.")
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False) ##True
    dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test, task=task) 
    #print ("Test Dataset comprises:", len(dataset_test), "samples.")
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    
    if with_val:
        return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent
    else: 
        return loader_train, loader_test, vocab_sent, invert_vocab_sent ## full_training
    

"""
def clean_tokenization(list_as_document, object):    
    if object=="text":
        tok_document = list_as_document.split("', '")
        tok_document[0] = tok_document[0][2:]
        tok_document[-1] = tok_document[-1][:-2]
        return tok_document
    else:
        tok_document = list_as_document.split(", ")
        tok_document[0] = tok_document[0][1:]
        tok_document[-1] = tok_document[-1][:-1]
        return [int(element) for element in tok_document]

def create_loaders(df_full_train, df_test, max_len, batch_size, with_val=True, tokenizer_from_scratch=True, path_ckpt='', df_val=None, task="classification"): ### when val partition doesn't exist in original dataset

    source_column_name = {"classification": 'article_text', "summarization": 'Cleaned_Article'}
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}
    id_column_name = {"classification": 'article_id', "summarization": 'Article_ID'}

    if with_val:         ### if val doesn't exist, split training data
        if df_val is None:
            df_train, df_val = train_test_split(df_full_train, test_size=0.2) 
        else:
            df_train = df_full_train

        documents = [sent_tokenize(doc) if task=="classification" else clean_tokenization(doc, "text") for doc in list(df_train[source_column_name[task]])]
        documents_val = [sent_tokenize(doc) if task=="classification" else clean_tokenization(doc, "text") for doc in list(df_val[source_column_name[task]])]
        if task=="classification":
            labels = list(df_train[label_column_name[task]])
            labels_val = list(df_val[label_column_name[task]])
        else: 
            labels = [clean_tokenization(doc, "label") for doc in list(df_train[label_column_name[task]])]
            labels_val = [clean_tokenization(doc, "label") for doc in list(df_val[label_column_name[task]])]

        ### es probable que los ides de articulos no sean unicos en diferentes particiones porque son el index del dataset loaded from HuggingFace
        article_identifiers = list(df_train[id_column_name[task]])  
        article_identifiers_val = list(df_val[id_column_name[task]])
    else:
        documents = [sent_tokenize(doc) if task=="classification" else clean_tokenization(doc, "text") for doc in list(df_full_train[source_column_name[task]])]
        if task=="classification":  
            labels = list(df_full_train[label_column_name[task]])   
        else:
            labels = [clean_tokenization(doc, "label") for doc in list(df_full_train[label_column_name[task]])] ## no debiese pasar con mis datasets
        article_identifiers = list(df_full_train[id_column_name[task]])
   
    documents_test = [sent_tokenize(doc) if task=="classification" else clean_tokenization(doc, "text") for doc in list(df_test[source_column_name[task]])]
    if task=="classification":
        labels_test = list(df_test[label_column_name[task]])
    else:
        labels_test = [clean_tokenization(doc, "label") for doc in list(df_test[label_column_name[task]])]
    article_identifiers_test = list(df_test[id_column_name[task]])

    if tokenizer_from_scratch:
        documents_ids, vocab_sent, invert_vocab_sent = documents_to_ids(documents)
        if with_val:
            documents_ids_val, vocab_sent, invert_vocab_sent = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
            dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val) 
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        documents_ids_test, vocab_sent, invert_vocab_sent = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)

    if not tokenizer_from_scratch:
        checkpoint = torch.load(path_ckpt)
        invert_vocab_sent = checkpoint['hyper_parameters']['invert_vocab_sent']
        vocab_sent= {v:k for k,v in invert_vocab_sent.items()}

        documents_ids = documents_to_ids(documents, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
        if with_val:
            documents_ids_val = documents_to_ids(documents_val, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
            dataset_val = DocumentDataset(documents_ids_val, labels_val, max_len, article_identifiers_val) 
            loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        documents_ids_test = documents_to_ids(documents_test, from_scratch=False, dict_text_ids=vocab_sent, invert_text_ids=invert_vocab_sent)
        

    dataset_train = DocumentDataset(documents_ids, labels, max_len, article_identifiers) 
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = DocumentDataset(documents_ids_test, labels_test, max_len, article_identifiers_test) 
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    
    if with_val:
        return loader_train, loader_val, loader_test, df_train, df_val, vocab_sent, invert_vocab_sent
    else: 
        return loader_train, loader_test, vocab_sent, invert_vocab_sent ## full_training
"""
    

