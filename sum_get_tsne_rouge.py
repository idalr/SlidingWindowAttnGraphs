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

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"]='-1'

# TODO: move to util after debugging
## cut out unnecessary parts
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


# loading data
in_path = "/scratch2/rldallitsako/datasets/GovReport-Sum/"
data_train = None
labels_train = None
data_test = None
labels_test = None

path_root = "/scratch2/rldallitsako/datasets/AttnGraphs_GovReports/Extended_ReLu/max_unified"
filename_train="predict_train_documents.csv"
filename_val= "predict_val_documents.csv"
filename_test= "predict_test_documents.csv"

# define graphs
type_model = "GAT"
type_graph = "max"
flag_binary= False
multi_flag = True

# define model and train args
batch_size = 32
num_hidden = 256
num_classes = 2
num_layers = 2
lr = 1e-3
std = 0.5
num_print = 5
max_len = 1000
granularity= "local"
model_name= "Extended_ReLu"

df_train, df_val, df_test = load_data(in_path, data_train, labels_train, data_test, labels_test, with_val=True)

# ids2remove_train = check_dataframe(df_train)
# for id_remove in ids2remove_train:
#     df_train = df_train.drop(id_remove)
# df_train.reset_index(drop=True, inplace=True)
# print("Train shape:", df_train.shape)

#dataset_train = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_train, filter_type=type_graph, data_loader=None, mode="train", binarized=flag_binary, multi_layer_model=multi_flag)
#dataset_val = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_val, filter_type=type_graph, data_loader=None, mode="val", binarized=flag_binary, multi_layer_model=multi_flag)
dataset_test = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_test, filter_type=type_graph, data_loader=None, mode="test", binarized=flag_binary, multi_layer_model=multi_flag)

#create loader from existing sentence vocabulary
loader_train, loader_val, loader_test, _ , _ , _, _ = create_loaders(df_train, df_test, max_len, batch_size, df_val=df_val,
                                                                     task="summarization", tokenizer_from_scratch=False, path_ckpt=in_path)
my_class_weights, labels_counter = get_class_weights(df_train, task="summarization")
calculated_cw = my_class_weights
print("\nClass weights - from training partition:", calculated_cw)

path_checkpoint = "/scratch2/rldallitsako/datasets/HomoGraphs_GovReports/Extended_ReLuGAT/max_unified/"
GAT_ckpt= "GAT_3L_256U_max_unified_run2-OUT-epoch=49-Val_f1-ma=0.52.ckpt"  ###chequear best en results file (1 by 1)
### No guardé todos los params en checkpoint asi que inicializar modelo y cargar pesos

model = GAT_NC_model(dataset_test.num_node_features, num_hidden, num_classes, num_layers, lr, dropout=0.2, class_weights=calculated_cw, with_edge_attr=True)
checkpoint = torch.load(path_checkpoint+GAT_ckpt)
print ("Loading model from:", path_checkpoint+GAT_ckpt)
model.load_state_dict(checkpoint['state_dict'])

# then run evaluation on test_loader
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# example usage
batch = next(iter(test_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
_ = model(batch.x.float().to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device))

R_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
path_vocab = "/scratch/datasets/GovReport-Sum/"
preds, all_labels, max_id, min_id, max_f1, min_f1, rouge1, rouge2, rougeL = predict_sentences_4Rouge(
    model, test_loader, path_vocab, df_test, R_scorer, maxf1=0.55, minf1=0.49)

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


print("Visualizing embeddings from trained model...")

for i in range(num_print):
    data = dataset_test[i].to(device)
    with torch.no_grad():
        out_sample = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)

    # Choose either one
    # visualize(data.x, color=data.y)
    # visualize(out_sample, color=data.y)
    visualize_side_by_side(data.x, out_sample, color=data.y)