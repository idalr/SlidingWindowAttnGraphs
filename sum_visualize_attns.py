import re, os
import matplotlib.pyplot as plt
import warnings

import pandas as pd
from nltk import sent_tokenize

from base_model import MHASummarizer
from eval_models import retrieve_parameters, eval_results, filtering_matrices

warnings.filterwarnings("ignore")

import time
import torch
from sklearn.manifold import TSNE

#from graph_data_loaders import AttentionGraphs_Sum
from preprocess_data import load_data
from data_loaders import create_loaders, clean_tokenization_sent, check_dataframe, get_class_weights

from gnn_model import partitions, GAT_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score

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

#loader_train,loader_val, loader_test, vocab_sent, invert_vocab_sent = create_loaders(df_train, df_test, max_len, batch_size, with_val=True,tokenizer_from_scratch=False, path_ckpt=in_path, df_val=df_val, task="summarization")
max_len = 50

#create loader from existing sentence vocabulary
loader_train, loader_val, loader_test, _, _, _, invert_vocab_sent = create_loaders(df_train, df_test, max_len, batch_size, df_val=df_val,
                                                                     task="summarization", tokenizer_from_scratch=False, path_ckpt=in_path)
del df_val, df_test
del invert_vocab_sent

# load model and eval data
path_models = "HomoGraphs_GovReports/"
df_logger = pd.read_csv(path_models+"df_logger_cw.csv")

model_name= "Extended_NoTemp_w30" #"Extended_NoTemp"
path_checkpoint, model_score, model_window = retrieve_parameters(model_name, df_logger)

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


# TODO: visualize
print("Visualize attentions...")

tolerance = 0.5
num_print = 3
granularity= "local"
filtering=True

# TODO: debug, because now filter is not working
# Mean
filter_type = "mean"
_, _, _, deletions_t = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths, model_window, df_train, print_samples=num_print,
                                                                                    degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
# _, _, _, deletions_v = filtering_matrices(full_attn_weights_v, all_article_identifiers_v, sent_lengths_val, model_window, df_val, print_samples=num_print,
#                                                                                     degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
# _, _, _, deletions_test = filtering_matrices(full_attn_weights_test, all_article_identifiers_test, sent_lengths_test, model_window, df_test, print_samples=num_print,
#                                                                                     degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
#
# plt.hist(x=[deletions_t, deletions_v, deletions_test], bins=50, color=['blue', 'red', 'orange'])
# plt.xlim(0, 7500)
# plt.legend(["Train", "Val", "Test"])
# plt.title("Deletions for mean local-filtering strategies - Model: "+model_name)
# plt.show()
#
# # Mean
# filter_type = "max"
# _, _, _, deletions_t = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths, model_window, df_train, print_samples=num_print,
#                                                                                     degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
# _, _, _, deletions_v = filtering_matrices(full_attn_weights_v, all_article_identifiers_v, sent_lengths_val, model_window, df_val, print_samples=num_print,
#                                                                                     degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
# _, _, _, deletions_test = filtering_matrices(full_attn_weights_test, all_article_identifiers_test, sent_lengths_test, model_window, df_test, print_samples=num_print,
#                                                                                     degree_std=tolerance, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)
#
# plt.hist(x=[deletions_t, deletions_v, deletions_test], bins=50, color=['blue', 'red', 'orange'])
# plt.xlim(0, 7500)
# plt.legend(["Train", "Val", "Test"])
# plt.title("Deletions for max local-filtering strategies - Model: "+model_name)
# plt.show()

# TODO: implement visualize h
# def visualize(h, color):
#     z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
#
#     plt.figure(figsize=(6, 6))
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     plt.show()

# from https://github.com/Buguemar/AttnGraphs/blob/main/Adjency%20Matrix%20Visual.ipynb
# from gnn_model import partitions
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchmetrics import F1Score
#
# type_model = "GAT"
# type_graph = "mean"
# graph_construction = type_graph
# model_name = "Extended_Anneal"
# path_models = "/scratch/mbugueno/HomoGraphs_GovReports/Extended_Anneal/"
# project_name = model_name + "2" + type_model + "_" + type_graph
#
# acc_tests = []
# f1_tests = []
# f1ma_tests = []
# all_train_times = []
# all_full_times = []
#
# for i in range(1):
#     nl = 2
#     dim = 64
#
#     model = GAT_model(dataset_train.num_node_features, dim, 2, nl, 0.001, dropout=0.1, class_weights=my_class_weights)
#
#     #### first train sample before model fitting
#     model.eval()
#     data = dataset_train[0]
#     out_sample = model(data.x, data.edge_index, data.edge_attr, data.batch)
#     visualize(out_sample, color=data.y)
#     model.train()
#     ####
#
#     early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True, patience=7, min_delta=0.001)
#     path_for_savings = path_models + type_model + "/" + graph_construction  # if config_file["baseline"] else path_models+type_model
#     checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1, dirpath=path_for_savings,
#                                           filename=type_model + "_" + str(nl) + "L_" + str(
#                                               dim) + "U_" + graph_construction + "_run" + str(
#                                               i) + "-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
#     wandb_logger = WandbLogger(
#         name=model_name + '2' + type_model + "_" + str(nl) + "L_" + str(dim) + "U_" + graph_construction + "_run" + str(
#             i), save_dir=path_for_savings, project=project_name)
#     trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=[early_stop_callback, checkpoint_callback],
#                          logger=wandb_logger, max_epochs=50, enable_progress_bar=False)
#
#     train_loader, val_loader, test_loader = partitions(dataset_train, dataset_test, dataset_val=dataset_val, bs=8)
#     print("partititons created")
#
#     starti = time.time()
#     trainer.fit(model, train_loader, val_loader)
#
#     print("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch)
#     train_time = time.time() - starti
#     print(f"Training time: {train_time:.2f} secs")
#
#     # load best checkpoint
#     print("Best model path:", trainer.checkpoint_callback.best_model_path)
#     checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
#     model.load_state_dict(checkpoint['state_dict'])
#
#     preds, trues = model.predict(test_loader, cpu_store=False)
#
#     acc = (torch.Tensor(trues) == preds).float().mean()
#     f1_score = F1Score(task='multiclass', num_classes=2, average='none')
#     f1_all = f1_score(preds.int(), torch.Tensor(trues).int())
#     print("Acc Test:", acc)
#     print("F1-macro Test:", f1_all.mean())
#     print("F1 for each class:", f1_all)
#
#     endi = time.time()
#     total_timei = endi - starti
#     print("Training + Testing time: " + str(total_timei))
#
#     acc_tests.append(acc.cpu().numpy())
#     f1_tests.append(f1_all.cpu().numpy())
#     f1ma_tests.append(f1_all.mean().cpu().numpy())
#     all_train_times.append(train_time)
#     all_full_times.append(total_timei)
#
#     #### first train sample after model fitting
#     model.eval()
#     out_sample = model(data.x, data.edge_index, data.edge_attr, data.batch)
#     visualize(out_sample, color=data.y)
#     model.train()
#     ####