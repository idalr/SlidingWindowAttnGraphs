# TODO: clean imports
import argparse
import re, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import yaml
import random

from analyses.util_ana import visualize_tsne, compare_similariry, predict_sentences, plot_two_distributions

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

# TODO: also make it take config_gnn file
# TODO: merge summary_distribution into this file
# TODO: move all function to util in this folder (after debugging)
## cut out unnecessary parts
# TODO: merge rouge and bertscore into one function

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file, num_print,
             random=False, tsne=False, rouge_score=False, bert_score=False, sent_dist=False):
    ### Load configuration file and set parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    logger_name = config_file["logger_name"]
    type_model = config_file["type_model"]
    dataset_name = config_file["dataset_name"]
    path_vocab = config_file["load_data_paths"]["in_path"]
    model_name = config_file["model_name"]
    setting_file = config_file["setting_file"]  # Setting file from which to pick the best checkpoint

    flag_binary = config_file["binarized"]
    multi_flag = config_file["multi_layer"]
    unified_flag = config_file["unified_nodes"]
    save_flag = config_file["saving_file"]

    root_graph = config_file["data_paths"]["root_graph_dataset"]
    path_results = config_file["data_paths"]["results_folder"]
    path_logger = config_file["data_paths"]["path_logger"]
    path_vocab = config_file["load_data_paths"]["in_path"]

    num_classes = config_file["model_arch_args"]["num_classes"]
    lr = config_file["model_arch_args"]["lr"]
    dropout = config_file["model_arch_args"]["dropout"]
    dim_features = config_file["model_arch_args"]["dim_features"]
    n_layers = config_file["model_arch_args"]["n_layers"]
    num_runs = config_file["model_arch_args"]["num_runs"]

    model_name = config_file["model_name"]
    df_logger = pd.read_csv(path_logger + logger_name)
    path_checkpoint, model_score = retrieve_parameters(model_name, df_logger)
    file_to_save = model_name+"_"+str(model_score)[:5]
    type_graph = config_file["type_graph"]
    max_len = config_file["max_len"]
    path_models = path_logger+model_name+"/"

    if config_file["load_data_paths"]["with_val"] == True:
        df_train, _, df_test = load_data(**config_file["load_data_paths"])
    else:
        df_train, df_test = load_data(**config_file["load_data_paths"])

    if config_file["with_cw"] == True:
        my_class_weights, labels_counter = get_class_weights(df_train)
        calculated_cw = my_class_weights
        print("\nClass weights - from training partition:", my_class_weights)
        print("Class counter:", labels_counter)
    else:
        print("\n-- No class weights specificied --\n")
        calculated_cw = None

    filename_test = "post_predict_test_documents.csv"
    path_dataset = os.path.join(root_graph, "raw")
    path_filename_test = os.path.join(path_dataset, filename_test)
    if os.path.exists(path_filename_test):
        print("Test file Requirements already satified in ", root_graph + "/raw/")
    else:
        _, loader_test, _, _ = create_loaders(df_train, df_test, max_len, config_file["batch_size"],
                                                         with_val=False, tokenizer_from_scratch=False,
                                                         path_ckpt=path_vocab,
                                                         df_val=None, task="summarization")

        print("\nLoading", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
        model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
        print("Done")

        print("\nPredicting Test")
        _, _, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(
            loader_test, cpu_store=False, flag_file=True)
        post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
        post_predict_test_docs.to_csv(path_dataset + filename_test, index=False)
        for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test, all_doc_ids_test):
            post_predict_test_docs.loc[len(post_predict_test_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
            }
        post_predict_test_docs.to_csv(path_dataset + filename_test, index=False)
        print("Finished and saved in:", path_dataset + filename_test)

        dataset_test = UnifiedAttentionGraphs_Sum(root_graph, filename_test, filter_type=type_graph,
                                                  data_loader=loader_test, model_ckpt=path_checkpoint,
                                                  mode="test", binarized=flag_binary)

    ###gat_checkpoint = "/scratch2/rldallitsako/HomoGraphs_GovReports/Extended_ReLuGAT/max_unified/"
    gat_checkpoint = os.path.join(root_graph, model_name, type_graph + "_unified")
    gat_ckpt= "GAT_3L_256U_max_unified_run2-OUT-epoch=49-Val_f1-ma=0.52.ckpt"
    match = re.search(r'_(\d+)L_(\d+)U_', gat_ckpt)
    if match:
        num_layers = int(match.group(1))
        num_hidden = int(match.group(2))
    else:
        print("Pattern not found")

    # then run evaluation on test_loader
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gat_model = GAT_NC_model(dataset_test.num_node_features, num_hidden, num_classes, num_layers, lr,
                         dropout=0.2, class_weights=calculated_cw, with_edge_attr=True)
    print ("Loading model from:", gat_checkpoint+gat_ckpt)
    checkpoint = torch.load(gat_checkpoint+gat_ckpt, map_location=torch.device('cpu'))
    gat_model.load_state_dict(checkpoint['state_dict'])
    gat_model = gat_model.to(device)
    _ = gat_model(batch.x.float().to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device))

    if tsne:
        print("Visualizing embeddings from trained model...")

        if random:
            indices_test = list(range(dataset_test))
            sampled_indices_test = random.sample(indices_test, num_print)
        else:
            sampled_indices_test = range(num_print)

        for i in sampled_indices_test:
            data = dataset_test[i].to(device)
            with torch.no_grad():
                out_sample = gat_model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
            visualize_tsne(data.x, out_sample, color=data.y)

    if rouge_score:
        print("Analyzing Rouge scores...")
        preds, all_labels, max_id, min_id, max_f1, min_f1, rouge1, rouge2, rougeL = compare_similariry(
            gat_model, test_loader, path_vocab, df_test, scorer='rouge', maxf1=0.55, minf1=0.49)
        _, _= eval_results(torch.Tensor(preds), all_labels, num_classes,
                           f"Test - {num_layers}L-{model_name} {type_graph} Graphs")

        sns.boxplot(data=(rouge1, rouge2, rougeL), orient='v')
        sns.stripplot(data=(rouge1, rouge2, rougeL), marker="o", alpha=0.15, color="black", orient='v')
        plt.title(f"[{num_layers}L-{model_name}] Distribution of Rouge-1/-2/-L F1-scores", fontsize=12)
        plt.xticks(ticks=[0, 1, 2], labels=['Rouge-1', 'Rouge-2', 'Rouge-L'])
        plt.ylabel("Score", fontsize=10)
        plt.xlabel("Rouge Scorer", fontsize=10)
        plt.show()

    if bert_score:
        print("Analyzing BERTScore score...")
        preds, all_labels, max_id, min_id, max_f1, min_f1, P, R, F1 = compare_similariry(
            gat_model, test_loader, path_vocab, df_test, scorer='bertscore', maxf1=0.55, minf1=0.49)
        _, _= eval_results(torch.Tensor(preds), all_labels, num_classes,
                           f"Test - {num_layers}L-{model_name} {type_graph} Graphs")

        sns.boxplot(data=(P, R, F1), orient='v')
        sns.stripplot(data=(P, R, F1), marker="o", alpha=0.15, color="black", orient='v')
        plt.title(f"[{num_layers}L-{model_name}] Distribution of BERTScore Presicion/Recall/F1 scores", fontsize=12)
        plt.xticks(ticks=[0, 1, 2], labels=['Presicion', 'Recall', 'F1'])
        plt.ylabel("Score", fontsize=10)
        plt.xlabel("BERTScorer Scorer", fontsize=10)
        plt.show()

    if sent_dist:
        if rouge_score == False and bert_score == False:
            preds, all_labels = predict_sentences(gat_model, test_loader)

        # Plot sentence relative position distribution
        plot_two_distributions(preds, all_labels)

        # Plot sentence Heatmap
        oracle_matrix = np.array([doc + [0] * (max_len - len(doc)) for doc in all_labels])
        pred_matrix = np.array([doc + [0] * (max_len - len(doc)) for doc in preds])

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        sns.heatmap(oracle_matrix, cmap="Blues", cbar=False, ax=ax[0])
        ax[0].set_ylabel("Oracle")
        sns.heatmap(pred_matrix, cmap="Oranges", cbar=False, ax=ax[1])
        ax[1].set_ylabel("Predicted")
        ax[1].set_xlabel("Sentence Position")
        plt.suptitle("Sentence Selection Heatmap (per document)")
        plt.show()

        # Plot predicted summary sentence counts vs oracle summary sentence count
        num_preds_1 = [i.count(1) for i in preds]
        num_true_1 = [i.count(1) for i in all_labels]

        plt.figure(figsize=(8, 6))
        plt.scatter(num_true_1, num_preds_1, alpha=0.6, color='purple')
        plt.xlabel("Sentence counts of oracle summaries")
        plt.ylabel("Sentence counts of predicted summaries")
        plt.title("Comparing Sentence Counts in Summaries")
        plt.tight_layout()
        plt.show()

        # Plot predicted summary sentence counts vs document length
        doc_len = [len(i) for i in preds]

        plt.figure(figsize=(8, 6))
        plt.scatter(num_preds_1, doc_len, alpha=0.6, color='purple')
        plt.xlabel("Length of Document")
        plt.ylabel("Sentence counts of predicted summaries")
        plt.title("Comparing Predicted Summary Length to Document Length")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    arg_parser.add_argument(
        "--num_print",
        type=int,
        default=5,
        help="optional number to print",
    )
    arg_parser.add_argument(
        "--random",
        action="store_true",
        help="set this flag to enable random samples for num_print",
    )
    arg_parser.add_argument(
        "--tsne",
        action="store_true",
        help="plot t-SNE samples as num_print",
    )
    arg_parser.add_argument(
        "--rouge_score",
        action="store_true",
        help="analyze Rouge scores",
    )
    arg_parser.add_argument(
        "--bert_score",
        action="store_true",
        help="analyze BERTScore",
    )
    arg_parser.add_argument(
        "--sent_dist",
        action="store_true",
        help="analyze sentence distribution",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file, **vars(args))