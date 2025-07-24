import torch
from sklearn.utils import class_weight
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize


def get_class_weights(df_, task="classification"):
    label_column_name = {"classification": 'article_label', "summarization": 'Calculated_Labels'}

    if task == "summarization":
        all_class_weights = torch.zeros(len(df_), 2)
        counts = {1: 0, 0: 0}

        for i, doc in enumerate(df_[label_column_name[task]]):
            labels = clean_tokenization_sent(doc, 'label')
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(torch.tensor(labels)),
                                                              y=torch.tensor(labels).numpy())
            new = Counter(labels)
            for k in new.keys():
                counts[k] += new[k]

            class_weights = torch.tensor(class_weights, dtype=torch.float)
            all_class_weights[i] = class_weights

        t_class_weights = all_class_weights.mean(axis=0)

    else:
        labels = df_[label_column_name[task]].tolist()
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(torch.tensor(labels)),
                                                          y=torch.tensor(labels).numpy())
        t_class_weights = torch.tensor(class_weights, dtype=torch.float)
        counts = Counter(labels)

    return t_class_weights, counts


def clean_tokenization_sent(list_as_document, object):
    if object == "text":
        tok_document = list_as_document.split(
            ' [St]')  # The token ' [St]' is used during our preprocessing to separate sentences.
        return tok_document
    else:
        tok_document = list_as_document.split(", ")
        tok_document[0] = tok_document[0][1:]
        tok_document[-1] = tok_document[-1][:-1]
        return [int(element) for element in tok_document]


def check_dataframe(df_, task='summarization'):
    if task == "summarization":
        to_remove = []
        for i in range(len(df_)):
            articulos = df_["Cleaned_Article"][i]
            lista_lab = df_["Calculated_Labels"][i]
            if len(clean_tokenization_sent(articulos, 'text')) != len(clean_tokenization_sent(lista_lab, 'label')):
                to_remove.append(df_.index[i])
    else:
        to_remove = []
        for i in range(len(df_)):
            articulos = df_["article_text"][i]
            if len(sent_tokenize(articulos)) < 2:
                to_remove.append(df_.index[i])

    return to_remove


def check_dimensions(dataset, max_allowed):
    print("Checking dataset dimensions -- Max allowed:", max_allowed)

    for i in range(len(dataset)):
        sample = dataset[i]
        if len(sample['documents_ids']) > max_allowed:
            print("More sentences than allowed in Dataset Object with id:", i)
            print("Length of doc:", len(sample['documents_ids']))

        if len(sample['labels']) > max_allowed:
            print("More labels than allowed in Dataset Object with id:", i)
            print("Length of labels:", len(sample['labels']))

        else:
            continue
    return


def documents_to_ids(documents, from_scratch=True, dict_text_ids=None, invert_text_ids=None, modifiable=False):
    # Our implementation assumes documents are represented as a list of sentences
    # Tokenize each document into sentences (each sentence will be a node in our final graphs)
    if from_scratch:
        dict_text_ids = {}
        documents_as_ids = []
        i = 1
        for doc in documents:
            doc_as_ids = []
            for sent in doc:
                if sent not in dict_text_ids:
                    dict_text_ids[sent] = i
                    i += 1
                doc_as_ids.append(dict_text_ids[sent])
            documents_as_ids.append(doc_as_ids)

        invert_text_ids = {v: k for k, v in dict_text_ids.items()}
        return documents_as_ids, dict_text_ids, invert_text_ids

    else:
        if modifiable:
            modified = False
            documents_as_ids = []
            i = max(invert_text_ids.keys()) + 1
            for doc in documents:
                doc_as_ids = []
                for sent in doc:
                    if sent not in dict_text_ids:
                        modified = True
                        dict_text_ids[sent] = i
                        invert_text_ids[i] = sent
                        i += 1
                    doc_as_ids.append(dict_text_ids[sent])
                documents_as_ids.append(doc_as_ids)
            if modified:
                return documents_as_ids, dict_text_ids, invert_text_ids
            else:
                return documents_as_ids, None, None
        else:
            documents_as_ids = []
            for doc in documents:
                doc_as_ids = []
                for sent in doc:
                    if sent not in dict_text_ids:
                        print("Error: sentence not in dictionary")
                        print("sentence:", sent)
                        doc_as_ids.append(0)
                    else:
                        doc_as_ids.append(dict_text_ids[sent])
                documents_as_ids.append(doc_as_ids)
            return documents_as_ids