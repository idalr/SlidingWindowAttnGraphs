import re, os
import matplotlib.pyplot as plt
import warnings

import pandas as pd
from nltk import sent_tokenize

from base_model import MHASummarizer
from eval_models import retrieve_parameters, eval_results, filtering_matrices
from graph_data_loaders import UnifiedAttentionGraphs_Sum

warnings.filterwarnings("ignore")

import time
import torch
from sklearn.manifold import TSNE

#from graph_data_loaders import AttentionGraphs_Sum
from preprocess_data import load_data
from data_loaders import create_loaders, clean_tokenization_sent, check_dataframe, get_class_weights

from gnn_model import partitions, GAT_model, GAT_NC_model

# define variables
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = 16
num_classes = 2

model_name = "Extended_NoTemp_w30"
path_models = "HomoGraphs_GovReports/Extended_NoTemp_w30/"
in_path = "datasets/GovReport-Sum/"
data_train = None
labels_train = None
data_test = None
labels_test = None

# load data
df_train, df_val, df_test = load_data(in_path, data_train, labels_train, data_test, labels_test, with_val=True)

print ("df_train", df_train.shape)
print ("df_val", df_val.shape)
print ("df_test", df_test.shape)

#################### minirun
df_train, df_val, df_test = df_train[:30], df_val[:30], df_test[:30]
####################

print("\nChecking train")
ids2remove_train = check_dataframe(df_train)
for id_remove in ids2remove_train:
    df_train = df_train.drop(id_remove)
df_train.reset_index(drop=True, inplace=True)
print("Train shape:", df_train.shape)

print("\nChecking val")
ids2remove_val = check_dataframe(df_val)
for id_remove in ids2remove_val:
    df_val = df_val.drop(id_remove)
df_val.reset_index(drop=True, inplace=True)
print("Val shape:", df_val.shape)

print("\nChecking test")
ids2remove_test = check_dataframe(df_test)
for id_remove in ids2remove_test:
    df_test = df_test.drop(id_remove)
df_test.reset_index(drop=True, inplace=True)
print("Test shape:", df_test.shape)

my_class_weights, labels_counter = get_class_weights(df_train, task="summarization")
print("\nClass weights - from training partition:", my_class_weights)
print("Class counter:", labels_counter)

### OBTAIN MAX SEQUENCE
sent_lengths = []
for i, doc in enumerate(df_train['Cleaned_Article']):
    sent_in_doc = clean_tokenization_sent(doc, 'text')

    if len(sent_in_doc) == 0:
        print("Empty doc in row-id:", i)
    sent_lengths.append(len(sent_in_doc))
    if len(sent_in_doc) >= 3000:
        print("Extremely long doc in row-id:", i, "with", len(sent_in_doc), "sentences.")

max_len = max(sent_lengths)  # Maximum number of sentences in a document
print("max number of sentences in train document:", max_len)

### OBTAIN MAX SEQUENCE
sent_lengths_val = []
for i, doc in enumerate(df_val['Cleaned_Article']):
    sent_in_doc = clean_tokenization_sent(doc, 'text')

    if len(sent_in_doc) == 0:
        print("Empty doc in row-id:", i)
    sent_lengths_val.append(len(sent_in_doc))
    if len(sent_in_doc) >= 3000:
        print("Extremely long doc in row-id:", i, "with", len(sent_in_doc), "sentences.")

max_len_val = max(sent_lengths_val)  # Maximum number of sentences in a document
print("max number of sentences in val document:", max_len_val)

### OBTAIN MAX SEQUENCE
sent_lengths_test = []
for i, doc in enumerate(df_test['Cleaned_Article']):
    sent_in_doc = clean_tokenization_sent(doc, 'text')

    if len(sent_in_doc) == 0:
        print("Empty doc in row-id:", i)
    sent_lengths_test.append(len(sent_in_doc))
    if len(sent_in_doc) >= 3000:
        print("Extremely long doc in row-id:", i, "with", len(sent_in_doc), "sentences.")

max_len_test = max(sent_lengths_test)  # Maximum number of sentences in a document
print("max number of sentences in test document:", max_len_test)

#create loader from existing sentence vocabulary
loader_train, loader_val, loader_test, _, _, _, invert_vocab_sent = create_loaders(df_train, df_test, max_len, batch_size, df_val=df_val,
                                                                     task="summarization", tokenizer_from_scratch=False, path_ckpt=in_path)
# del df_val, df_test
del invert_vocab_sent

# ############################################################### load 4 debug
# df_test, _, _ = load_data(in_path, data_test, labels_train, data_test, labels_test, with_val=True)
# df_test = df_test[:30]
max_len = 50
#
# print("\nChecking test")
# ids2remove_test = check_dataframe(df_test)
# for id_remove in ids2remove_test:
#     df_test = df_test.drop(id_remove)
# df_test.reset_index(drop=True, inplace=True)
# print("Test shape:", df_test.shape)
_, _, loader_test, _, _, _, _ = create_loaders(df_test, df_test, max_len, batch_size, df_val=df_test,
                                                                     task="summarization", tokenizer_from_scratch=False, path_ckpt=in_path)
#
# ### OBTAIN MAX SEQUENCE
# sent_lengths_test = []
# for i, doc in enumerate(df_test['Cleaned_Article']):
#     sent_in_doc = clean_tokenization_sent(doc, 'text')
#
#     if len(sent_in_doc) == 0:
#         print("Empty doc in row-id:", i)
#     sent_lengths_test.append(len(sent_in_doc))
#     if len(sent_in_doc) >= 3000:
#         print("Extremely long doc in row-id:", i, "with", len(sent_in_doc), "sentences.")
#
# max_len_test = max(sent_lengths_test)  # Maximum number of sentences in a document
# print("max number of sentences in test document:", max_len_test)
# ############################################################### load 4 debug

# load model and eval data
path_models = "HomoGraphs_GovReports/"
df_logger = pd.read_csv(path_models+"df_logger_cw.csv")

model_name= "Extended_NoTemp_w30" #"Extended_NoTemp"
path_checkpoint, model_score= retrieve_parameters(model_name, df_logger)

print ("\nLoading", model_name, "({0:.3f}".format(model_score),") from:", path_checkpoint)
model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
print ("Model temperature", model_lightning.temperature)

print("Start Predicting...")
preds_t, full_attn_weights_t, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(loader_train, cpu_store=False)
preds_v, full_attn_weights_v, all_labels_v, all_doc_ids_v, all_article_identifiers_v = model_lightning.predict(loader_val, cpu_store=False)
preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(loader_test, cpu_store=False)
print("Evaluating predictions...")
# acc_t, f1_all_t = eval_results(preds_t, all_labels_t, num_classes, "Train")
# acc_v, f1_all_v = eval_results(preds_v, all_labels_v, num_classes, "Val")
# acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")


# visualize
print("Visualize attentions...")

tolerance = 0.5
num_print = 1 #3
granularity= "local"
filtering=True
max_len = model_lightning.max_len
model_window = model_lightning.window
# max_len = 50

# compare valid_sent and max_len
sent_lengths = [min(i,max_len) for i in sent_lengths]
sent_lengths_val = [min(i,max_len) for i in sent_lengths_val]
sent_lengths_test = [min(i,max_len) for i in sent_lengths_test]

# Mean
filter_type = "mean"
_, _, _, deletions_t = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths, model_window, df_train, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
_, _, _, deletions_v = filtering_matrices(full_attn_weights_v, all_article_identifiers_v, sent_lengths_val, model_window, df_val, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
_, _, _, deletions_test = filtering_matrices(full_attn_weights_test, all_article_identifiers_test, sent_lengths_test, model_window, df_test, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)

# Max
filter_type = "max"
_, _, _, x_deletions_t = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths, model_window, df_train, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
_, _, _, x_deletions_v = filtering_matrices(full_attn_weights_v, all_article_identifiers_v, sent_lengths_val, model_window, df_val, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
_, _, _, x_deletions_test = filtering_matrices(full_attn_weights_test, all_article_identifiers_test, sent_lengths_test, model_window, df_test, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)

plt.hist(x=[deletions_t, deletions_v, deletions_test, x_deletions_t, x_deletions_v, x_deletions_test], bins=500, color=['blue', 'cyan', 'green', 'magenta', 'red', 'orange'])
plt.xlim(0, 7500)
plt.legend(["Train_mean", "Val_mean", "Test_mean", "Train_max", "Val_max", "Test_max"])
plt.title("Deletions for different local-filtering strategies - Model: "+model_name)
plt.show()

# implement visualize h
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


# set variables
type_model = "GAT"
type_graph = "max"
graph_construction = type_graph
model_name = "Extended_NoTemp_w30"
path_models = "HomoGraphs_GovReports/Extended_NoTemp_w30/"
project_name = model_name + "2" + type_model + "_" + type_graph

acc_tests = []
f1_tests = []
f1ma_tests = []
all_train_times = []
all_full_times = []

root_graph = "datasets/AttnGraphs_GovReports/"
path_root = root_graph + model_name + "/Attention/" + type_graph + '_unified'
filename_test = "predict_test_documents.csv"
path_checkpoint = 'HomoGraphs_GovReports/Extended_NoTemp_w30/Extended_NoTemp_w30-epoch=00-Val_f1-ma=0.08.ckpt'
# model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)

# build filename_test and construct graphs for dataset_test
_, _ = model_lightning.predict_to_file(loader_test, saving_file=True, filename=filename_test, path_root=path_root)
dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, type_graph, loader_test, degree=0.5, model_ckpt=path_checkpoint, mode="test", binarized=False)

print("Printing samples...")
for i in range(1):
    nl = 2
    dim = 64

    model = GAT_NC_model(384, dim, 2, nl, 0.001, dropout=0.1, class_weights=my_class_weights)

    #### first train sample before model fitting
    model.eval()
    data = dataset_test[i]
    out_sample = model(data.x, data.edge_index, data.edge_attr, data.batch)
    # visualize(data.x, color=data.y)
    # visualize(out_sample, color=data.y)

    visualize_side_by_side(data.x, out_sample, color=data.y)
