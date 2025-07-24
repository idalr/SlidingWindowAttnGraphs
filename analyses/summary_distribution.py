import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import torch
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from torch_geometric.data import DataLoader

from analyses.util_ana import extract_relative_ones, smooth_counts, predict_sentences, extract_from_file, \
    plot_two_distributions
from gnn_model import GAT_NC_model
from graph_data_loaders import UnifiedAttentionGraphs_Sum


# def load_and_extract_col(csv_path, colname):
#     df = pd.read_csv(csv_path)
#     col_str = df[colname][0]
#     return ast.literal_eval(col_str)
#
# def compute_distribution(col, num_bins=100, return_proportion=False):
#     bin_counts = [0] * num_bins
#     bin_totals = [0] * num_bins
#     total_len = len(col)
#
#     for i, val in enumerate(col):
#         bin_index = int((i / total_len) * num_bins)
#         bin_index = min(bin_index, num_bins - 1)
#         if val == 1:
#             bin_counts[bin_index] += 1
#         bin_totals[bin_index] += 1
#
#     if return_proportion:
#         return [
#             bin_counts[i] / bin_totals[i] if bin_totals[i] > 0 else 0
#             for i in range(num_bins)
#         ]
#     else:
#         return bin_counts
#
# def load_all_distributions(csv_path, col_name, num_bins=100):
#     df = pd.read_csv(csv_path)
#     all_bins = []
#
#     for row in df[col_name]:
#         col = ast.literal_eval(row)
#         bins = compute_distribution(col, num_bins)
#         all_bins.append(bins)
#
#     # Sum over all rows to get aggregated counts
#     return np.sum(all_bins, axis=0)
#
# def smooth(values, window=10):
#     return np.convolve(values, np.ones(window) / window, mode='same')
#
# def load_and_aggregate(csv_path, col_name):
#     df = pd.read_csv(csv_path)
#     total_len = None
#     agg = None
#
#     for row in df[col_name]:
#         seq = ast.literal_eval(row)
#         if total_len is None:
#             total_len = len(seq)
#             agg = np.zeros(total_len)
#         agg += np.array(seq)
#
#     return agg
#
# def bin_and_smooth(sequence, num_bins=100, smooth_window=10):
#     binned = np.array_split(sequence, num_bins)
#     binned_counts = np.array([np.sum(b) for b in binned])
#     smoothed = uniform_filter1d(binned_counts, size=smooth_window)
#     return binned_counts, smoothed
#
# def get_plot(oracle_file, predicted_file, colname1, colname2):
#     # Load & process
#     col1 = load_and_extract_col(oracle_file, colname1)
#     col2 = load_and_extract_col(predicted_file, colname2)
#
#     dist1 = compute_distribution(col1)
#     dist2 = compute_distribution(col2)
#
#     smoothed1 = smooth(dist1)
#     smoothed2 = smooth(dist2)
#
#     # Plotting
#     x = np.linspace(0, 100, 100)
#     plt.figure(figsize=(12, 6))
#
#     #plt.plot(x, dist1, label="Oracle summaries", alpha=0.3, color='blue', drawstyle='steps-mid')
#     plt.plot(x, smoothed1, label="Smoothed oracle summaries", color='blue', linewidth=2)
#
#     #plt.plot(x, dist2, label="Predicted summaries", alpha=0.3, color='green', drawstyle='steps-mid')
#     plt.plot(x, smoothed2, label="Smoothed predicted summaries", color='green', linewidth=2)
#
#     plt.ylim(0, max(max(dist1), max(dist2)) * 1.1)
#     plt.xlabel("Relative Position (%)")
#     plt.ylabel("Number of summary sentences")
#     plt.title("Distribution of Summary Sentences by Relative Position (Oracle and Prediction)")
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #plt.grid(True)
#     plt.tight_layout()
#     plt.show()



# def plot_distribution(file, column="cols"):
#     x, raw = extract_relative_ones(file, column)
#     smoothed = smooth_counts(raw, window_size=5)
#
#     plt.figure(figsize=(10, 4))
#     plt.plot(x, raw, label="Raw", drawstyle='steps-mid', alpha=0.6)
#     plt.plot(x, smoothed, label="Smoothed (moving avg)", color='orange')
#
#     plt.xlabel("Relative Position (%)")
#     plt.ylabel("Count of 1s")
#     plt.title("Distribution of 1s by Relative Position")
#     plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()


# Load data to predict
num_hidden = 256
num_classes = 2
num_layers = 3
std = 0.5
lr = 0.001
type_graph = "max"
path_root = '../datasets/GovReport-Sum/Predicted'
filename_test = 'predict_test_document.csv'

dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, filter_type=type_graph, data_loader=loader_test, degree=std,model_ckpt=mha_checkpoint, mode="test",binarized=flag_binary)

gat_checkpoint = "/scratch2/rldallitsako/HomoGraphs_GovReports/Extended_ReLuGAT/max_unified/"
GAT_ckpt= "GAT_3L_256U_max_unified_run2-OUT-epoch=49-Val_f1-ma=0.52.ckpt"  ###chequear best en results file (1 by 1)
### No guardé todos los params en checkpoint asi que inicializar modelo y cargar pesos

model = GAT_NC_model(dataset_test.num_node_features, num_hidden, num_classes, num_layers, lr, dropout=0.2, class_weights=calculated_cw, with_edge_attr=True)
print ("Loading model from:", gat_checkpoint+GAT_ckpt)
checkpoint = torch.load(gat_checkpoint+GAT_ckpt, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# then run evaluation on test_loader
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
batch = next(iter(test_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
_ = model(batch.x.float().to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device))
preds, all_labels = predict_sentences(model, test_loader)

#(get_plot

oracle_pos = extract_from_file("../datasets/GovReport-Sum/Processed/df_test.csv", "Calculated_Labels", max_col=900)
predicted_pos = extract_from_file("../datasets/GovReport-Sum/Processed/df_validation.csv", "Calculated_Labels", max_col=900)

plot_two_distributions(oracle_pos, predicted_pos)
#plot_two_distributions("datasets/GovReport-Sum/Processed/df_test.csv", "datasets/GovReport-Sum/predict_test_documents.csv", "Calculated_Labels", "label")


import pandas as pd
import ast
import numpy as np
from bert_score import score

# df = pd.read_csv('datasets/GovReport-Sum/relux/test.csv')
#
# preds = []
# for row in df['preds']:
#     seq = ast.literal_eval(row)
#     preds.append(seq)
#
# labels = []
# for row in df['labels']:
#     seq = ast.literal_eval(row)
#     labels.append(seq)
#
# print(len(preds))

# find ratio 1/0 in each doc
# np.mean([i.count(1)/len(i) for i in preds])
# 0.28817335855733983
# np.mean([i.count(0)/len(i) for i in preds])
# 0.7118266414426602
# np.mean([i.count(1)/len(i) for i in labels])
# 0.07274365861093585
# np.mean([i.count(0)/len(i) for i in labels])
# 0.9272563413890642

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df1 = pd.read_csv('../datasets/GovReport-Sum/relux/relu_x_test_summary.csv')
df2 = pd.read_csv('../datasets/GovReport-Sum/Predicted/predict_test_documents.csv')
df3 = pd.read_csv('../datasets/GovReport-Sum/relux/test.csv')

# import seaborn as sns
# # Get distribution heatmap
# preds = df3['preds']
# labels = df3['labels']
#
# seq1 = [ast.literal_eval(i) for i in preds]
# seq2 = [ast.literal_eval(i) for i in labels]
# max_len = 1000
# oracle_matrix = np.array([doc + [0]*(max_len - len(doc)) for doc in seq2])
# pred_matrix = np.array([doc + [0]*(max_len - len(doc)) for doc in seq1])
#
# fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# sns.heatmap(oracle_matrix, cmap="Blues", cbar=False, ax=ax[0])
# ax[0].set_ylabel("Oracle")
# sns.heatmap(pred_matrix, cmap="Oranges", cbar=False, ax=ax[1])
# ax[1].set_ylabel("Predicted")
# ax[1].set_xlabel("Sentence Position")
# plt.suptitle("Sentence Selection Heatmap (per document)")
# plt.show()
#
#
# #
# # Get sentence counts per doc length
# preds = df1['predicted_nodes']
# labels = df2['label']
#
# seq1 = [list(map(int, i.strip('[]').split())) for i in preds]
# seq2 = [ast.literal_eval(i) for i in labels]
# ps = [len(i) for i in seq1]
# ls = [len(i) for i in seq2]
#
# plt.figure(figsize=(8, 6))
# plt.scatter(ls, ps, alpha=0.6, color='purple')
# plt.xlabel("Length of Document")
# plt.ylabel("Sentence counts of predicted summaries")
# plt.title("Comparing Predicted Summary Length to Document Length")
# #plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Get sentence counts
# preds = df1['predicted_nodes']
# labels = df2['label']
#
# seq1 = [list(map(int, i.strip('[]').split())) for i in preds]
# seq2 = [ast.literal_eval(i) for i in labels]
# ps = [len(i) for i in seq1]
# ls = [i.count(1) for i in seq2]
#
# plt.figure(figsize=(8, 6))
# plt.scatter(ls, ps, alpha=0.6, color='purple')
# plt.xlabel("Sentence counts of oracle summaries")
# plt.ylabel("Sentence counts of predicted summaries")
# plt.title("Comparing Sentence Counts in Summaries")
# #plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Get token counts
# df1['len_string1'] = df1['predicted_summary'].apply(lambda x: len(str(x).split()))
# df1['len_string2'] = df1['oracle_summary'].apply(lambda x: len(str(x).split()))
#
# plt.figure(figsize=(8, 6))
# plt.scatter(df1['len_string2'], df1['len_string1'], alpha=0.6, color='purple')
# plt.xlabel("Token Length of oracle summaries")
# plt.ylabel("Token Length of predicted summaries")
# plt.title("Comparing Token Lengths in Summaries")
# #plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# p_list = []
# r_list = []
# f_list = []
# for i,row in df.iterrows():
#     pred = row['predicted_summary']
#     gold = row['oracle_summary']
#
#     p, r, f1 = score([pred],[gold], lang="en", idf=True)
#     p_list.append(p)
#     r_list.append(r)
#     f_list.append(f1)
#
# print(np.mean(p_list))
#
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x=range(len(preds)), y=BF1, color='purple')
# plt.title("F1 Score to Document Length")
# plt.xlabel("Document Length")
# plt.ylabel("F1 Score")
# plt.tight_layout()
# plt.show()
#
# sns.boxplot(data=(BP, BR, BF1), orient='v')
# sns.stripplot(data=(BP, BR, BF1), marker="o", alpha=0.15, color="black", orient='v')
# plt.title("[1L-"+model_name+"] Distribution of P/R/F1 BERTScores", fontsize=12)
# plt.xticks(ticks=[0, 1, 2], labels=['Precision', 'Recall', 'F1'])
# plt.ylabel("Score", fontsize=10)
# plt.xlabel("BERTScore Scorer", fontsize=10)
# plt.show()
#
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x=[i.count(1) for i in preds], y=BF1, color='purple')
# plt.title("F1 Score to Predicted Summary Length")
# plt.xlabel("Length of predicted sentences")
# plt.ylabel("F1 Score")
# plt.tight_layout()
# plt.show()