# TODO: clean imports
import re, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import time
from tqdm import tqdm
import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import torch
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from torch_geometric.data import DataLoader

from gnn_model import GAT_NC_model
from graph_data_loaders import UnifiedAttentionGraphs_Sum

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from datasets import load_dataset
from functools import partial

from rouge_score import rouge_scorer
import pandas as pd
from operator import itemgetter

#from base_model import MultiHeadSelfAttention
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
from torchmetrics import F1Score
from torch import optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#from data_loaders import DocumentDataset
from torch_geometric.data import DataLoader, Dataset
from eval_models import retrieve_parameters, eval_results, filtering_matrices

from preprocess_data import load_data
#from summarization_utils import create_dataframe, clean_tokenization_sent
from base_model import MHASummarizer
from data_loaders import create_loaders, clean_tokenization_sent, check_dataframe, get_class_weights
from rouge_score import rouge_scorer

from graph_data_loaders import UnifiedAttentionGraphs_Sum
from gnn_model import partitions, GAT_NC_model



def predict_sentences_4Rouge(model, loader, path_invert_vocab, df_, R_scorer, maxf1=0.55, minf1=0.5, cpu_store=True):
    model.eval()
    device = model.device
    preds = []
    all_labels = []
    rougeLF1 = []
    rouge2F1 = []
    rouge1F1 = []

    sent_dict_disk = pd.read_csv(path_invert_vocab + "vocab_sentences.csv")
    invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}

    max_f1 = maxf1
    max_id_article = None
    min_f1 = minf1
    min_id_article = None
    isolated_count = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating ROUGE"):
            if data.has_isolated_nodes():
                isolated_count += 1
                continue

            try:
                out = model(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device),
                            data.batch.to(device))
            except:
                out = model(data.x.float().to(device), data.edge_index.to(device), None, data.batch.to(device))

            pred = out.argmax(dim=1)
            test_nodes = (pred == 1)

            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds += list(pred)
            all_labels.extend(data.y)

            # Construct summary from predicted nodes
            nodes_classified_as_1 = test_nodes.nonzero(as_tuple=True)[0]
            source_graph_id = data.edge_index[0]
            joint_wo_duplicates = ""
            seen_sentences = []

            for node in nodes_classified_as_1:
                try:
                    pos_i = (source_graph_id == node).nonzero(as_tuple=True)[0]
                    key_int = data.orig_edge_index[0][pos_i[0]].item()
                except:
                    continue
                if key_int not in seen_sentences:
                    seen_sentences.append(key_int)
                    key_matched2vocab = invert_vocab_sent[key_int]
                    joint_wo_duplicates += key_matched2vocab + " "

            # Get reference summary
            summary_matched = df_.where(df_['Article_ID'] == data['article_id'].item()).dropna()['Summary'].values[0]
            rouge_results = R_scorer.score(summary_matched, joint_wo_duplicates)

            rouge1F1.append(rouge_results['rouge1'].fmeasure)
            rouge2F1.append(rouge_results['rouge2'].fmeasure)
            rougeLF1.append(rouge_results['rougeL'].fmeasure)

            mean_f1 = np.mean([rouge1F1[-1], rouge2F1[-1], rougeLF1[-1]])
            if mean_f1 > max_f1:
                max_f1 = mean_f1
                max_id_article = data['article_id'].item()
            if mean_f1 < min_f1:
                min_f1 = mean_f1
                min_id_article = data['article_id'].item()

    print("\nArticle ID with best results:", max_id_article, "with F1-score:", max_f1)
    print("Article ID with worst results:", min_id_article, "with F1-score:", min_f1)
    print("#Graphs with isolated nodes:", isolated_count)
    print("Mean ROUGE 1-F1:", np.mean(rouge1F1), "Mean ROUGE 2-F1:", np.mean(rouge2F1), "Mean ROUGE L-F1:", np.mean(rougeLF1))

    return preds, all_labels, max_id_article, min_id_article, max_f1, min_f1, rouge1F1, rouge2F1, rougeLF1


# # implement visualize h in test set
# # TODO: move functions to util after debugging
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(6, 6))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def visualize_side_by_side(x, out_sample, color, title1="Input Features", title2="Output Embeddings"):
    x_np = x.detach().cpu().numpy()
    out_np = out_sample.detach().cpu().numpy()

    # Run t-SNE on both
    z1 = TSNE(n_components=2, perplexity=min(30, len(x_np) - 1)).fit_transform(x_np)
    z2 = TSNE(n_components=2, perplexity=min(30, len(out_np) - 1)).fit_transform(out_np)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, z, title in zip(axes, [z1, z2], [title1, title2]):
        ax.set_xticks([])
        ax.set_yticks([])
        scatter = ax.scatter(z[:, 0], z[:, 1], s=60, c=color, cmap="Set2")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def extract_from_file(file, column="cols", bins=100, max_col=2):
    df = pd.read_csv(file).head(max_col)
    seqs = []

    for row in df[column]:
        seq = ast.literal_eval(row)
        seqs.append(seq)

    return seqs

def extract_relative_ones(input_data, column=None, bins=300):
    positions = []

    if isinstance(input_data, str):  # it's a file path
        df = pd.read_csv(input_data)
        assert column is not None, "If input is a file, 'column' name must be specified"
        rows = df[column]
    else:  # assume it's already a list of lists
        rows = input_data

    for row in rows:
        try:
            if isinstance(row, str):
                seq = ast.literal_eval(row)
            else:
                seq = row
            seq_len = len(seq)
            for i, val in enumerate(seq):
                if val == 1:
                    rel_pos = i / seq_len
                    positions.append(rel_pos)
        except:
            continue

    hist, bin_edges = np.histogram(positions, bins=bins, range=(0, 1))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers * 100, hist

def smooth_counts(counts, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(counts, kernel, mode='same')

def plot_two_distributions(position1, position2=None):
    x1, raw1  = extract_relative_ones(position1)
    smooth1 = smooth_counts(raw1, window_size=5)

    plt.figure(figsize=(10, 4))
    plt.plot(x1, raw1, alpha=0.3, label="Raw File 1", drawstyle='steps-mid', color="blue")
    #plt.plot(x1, smooth1, label="Smoothed File 1", color="blue")


    if position2:
        x2, raw2 = extract_relative_ones(position2)
        smooth2 = smooth_counts(raw2, window_size=5)
        plt.plot(x2, raw2, alpha=0.3, label="Raw File 2", drawstyle='steps-mid', color="green")
        #plt.plot(x2, smooth2, label="Smoothed File 2", color="green")

    plt.xlabel("Relative Position (%)")
    plt.ylabel("Raw Count of 1s")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def predict_sentences(model, loader, cpu_store=True):
    model.eval()
    device = model.device
    preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            try:
                out = model(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device),
                            data.batch.to(device))
            except:
                out = model(data.x.float().to(device), data.edge_index.to(device), None, data.batch.to(device))

            pred = out.argmax(dim=1)

            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds.append(pred.tolist())
            all_labels.append(data.y.tolist())

    return preds, all_labels