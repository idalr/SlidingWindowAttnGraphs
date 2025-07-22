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

in_path = "/scratch2/rldallitsako/datasets/GovReport-Sum/"
with_val = True
max_len = 1000

# load data
df_train = pd.read_csv(in_path+"Processed/df_train.csv")
df_val = pd.read_csv(in_path+"Processed/df_validation.csv")
df_test = pd.read_csv(in_path+"Processed/df_test.csv")

print ("df_train", df_train.shape)
print ("df_val", df_val.shape)
print ("df_test", df_test.shape)

#################### minirun
df_train, df_val, df_test = df_train[:5], df_val[:5], df_test[:5]
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

max_len_train = max(sent_lengths)  # Maximum number of sentences in a document
print("max number of sentences in train document:", max_len_train)

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
loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len, batch_size, df_val=df_val,
                                                                     task="summarization", tokenizer_from_scratch=False, path_ckpt=in_path)

# load model and eval data
path_models_logger = os.path.join("/scratch2/rldallitsako/HomoGraphs_GovReports/", "df_logger_cw.csv")
df_logger = pd.read_csv(path_models_logger)

model_name= "Extended_ReLu" #"Extended_NoTemp"
path_checkpoint, model_score= retrieve_parameters(model_name, df_logger)

print ("\nLoading", model_name, "({0:.3f}".format(model_score),") from:", path_checkpoint)
model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
print ("Model temperature", model_lightning.temperature)
model_window = model_lightning.window
print ("Model Window percent", model_window)
max_len = model_lightning.max_len
print ("Model Max length", model_lightning.max_len)

print("Start Predicting...")
_, full_attn_weights_t, _, _, all_article_identifiers_t = model_lightning.predict(loader_train, cpu_store=False)
_, full_attn_weights_v, _, _, all_article_identifiers_v = model_lightning.predict(loader_val, cpu_store=False)
_, full_attn_weights_test, _, _, all_article_identifiers_test = model_lightning.predict(loader_test, cpu_store=False)

# print("Evaluating predictions...")
# acc_t, f1_all_t = eval_results(preds_t, all_labels_t, num_classes, "Train")
# acc_v, f1_all_v = eval_results(preds_v, all_labels_v, num_classes, "Val")
# acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")


# visualize
print("Visualize attentions...")

tolerance = 0.5
num_print = 3
granularity= "local"
filtering=True

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

# Plotting deletions
# plt.hist(x=[deletions_t, deletions_v, deletions_test, x_deletions_t, x_deletions_v, x_deletions_test], bins=5000, color=['blue', 'lightblue', 'green', 'red', 'orange', 'yellow'])
# plt.xlim(0, 10000)
# plt.legend(["Mean - Train", "Mean - Val", "Mean - Test", "Max - Train", "Max - Val", "Max - Test"])
# plt.title("Deletions for different local-filtering strategies - Model: "+model_name)
# plt.show()
