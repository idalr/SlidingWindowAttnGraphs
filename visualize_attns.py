import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from preprocess_data import load_data
from base_model import MHAClassifier
from eval_models import retrieve_parameters, eval_results, filtering_matrices, sw_not_filtering_matrices, sw_filtering_matrices
from data_loaders import create_loaders

os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TOKENIZERS_PARALLELISM"] = "False"

in_path = "datasets/HyperNews/"

data_train = "articles-training-byarticle-20181122.xml"
labels_train = "ground-truth-training-byarticle-20181122.xml"
data_test = "articles-test-byarticle-20181207.xml"
labels_test = "ground-truth-test-byarticle-20181207.xml"

batch_size = 32
num_classes = 2


df_full_train, df_test = load_data(in_path, data_train, labels_train, data_test, labels_test)

print ("df_train", df_full_train.shape)
print ("df_test", df_test.shape)


sent_lengths=[]
for i, doc in enumerate(df_full_train['article_text']):
    sent_in_doc = sent_tokenize(doc)
    if len(sent_in_doc)==0:
        print ("Empty doc en:", i)
    sent_lengths.append(len(sent_in_doc))
max_len = max(sent_lengths)  # Maximum number of sentences in a document
print ("max number of sentences in document:", max_len)

sent_lengths_test=[]
for i, doc in enumerate(df_test['article_text']):
    sent_in_doc = sent_tokenize(doc)
    if len(sent_in_doc)==0:
        print ("Empty doc en:", i)
    sent_lengths_test.append(len(sent_in_doc))
max_len_test = max(sent_lengths_test)  # Maximum number of sentences in a document
print ("max number of sentences in document:", max_len_test)

# mini run
df_full_train = df_full_train.head(50)
df_test = df_test.head(50)
sent_lengths = sent_lengths[:50]
sent_lengths_test = sent_lengths_test[:50]

path_models = "/HomoGraphs_HND/"
df_logger = pd.read_csv(path_models+"df_logger_cw.csv")

model_name= "Extended_NoTemp_100" #"Extended_NoTemp"
path_checkpoint, model_score = retrieve_parameters(model_name, df_logger)
loader_train, loader_test, vocab_sent, invert_vocab_sent = create_loaders(df_full_train, df_test, max_len, batch_size, with_val=False,
                                                                          tokenizer_from_scratch=False, path_ckpt=in_path)
print ("\nLoading", model_name, "({0:.3f}".format(model_score),") from:", path_checkpoint)
model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
print ("Model temperature", model_lightning.temperature)

print("Start Predicting...")
preds_t, full_attn_weights_t, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(loader_train, cpu_store=False)
preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(loader_test, cpu_store=False)
print("Evaluating predictions...")
acc_t, f1_all_t = eval_results(preds_t, all_labels_t, num_classes, "Train")
acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")

# path_root="/AttnGraphs_HND/"+model_name+"/Attention/full"
# path_dataset = path_root+"/raw/"
#
# filename="post_predict_train_documents.csv"
# filename_test= "post_predict_test_documents.csv"
#
# print ("\nCreating file for PyG dataset in:", path_dataset)
# post_predict_train_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
# post_predict_train_docs.to_csv(path_dataset+"post_predict_train_documents.csv", index=False)
#
# for article_id, label, doc_as_ids in zip(all_article_identifiers_t, all_labels_t, all_doc_ids_t):
#     post_predict_train_docs.loc[len(post_predict_train_docs)] = {
#     "article_id": article_id.item(),
#     "label": label.item(),
#     "doc_as_ids": doc_as_ids.tolist()
#     }
# post_predict_train_docs.to_csv(path_dataset+"post_predict_train_documents.csv", index=False)
# print ("Finished and saved in:", path_dataset+"post_predict_train_documents.csv")
#
#
# post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
# post_predict_test_docs.to_csv(path_dataset+"post_predict_test_documents.csv", index=False)
#
# for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test, all_doc_ids_test):
#     post_predict_test_docs.loc[len(post_predict_test_docs)] = {
#     "article_id": article_id.item(),
#     "label": label.item(),
#     "doc_as_ids": doc_as_ids.tolist()
#     }
# post_predict_test_docs.to_csv(path_dataset+"post_predict_test_documents.csv", index=False)
# print ("Finished and saved in:", path_dataset+"post_predict_test_documents.csv")

print("Visualize attentions...")

std = 0.5
num_print = 5
granularity= "local"

# # not filtering
# filtered_matrices, total_nodes, total_edges = sw_not_filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths,
#                                                                             df_full_train, print_samples=num_print, degree_std=std)
#
# filtered_matrices_test, total_nodes_test, total_edges_test = sw_not_filtering_matrices(full_attn_weights_test, all_article_identifiers_test, sent_lengths_test,
#                                                                             df_test, print_samples=num_print, degree_std=std)

filtering=True
filter_type = "mean"
print(f'Filtering type: {filter_type}')

# Train
filtered_matrices, total_nodes, total_edges, deletions = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths,
                                                                            df_full_train, print_samples=num_print,
                                                                            degree_std=std, with_filtering=filtering,
                                                                            filtering_type=filter_type, granularity=granularity)

# Test
filtered_matrices_test, total_nodes_test, total_edges_test, deletions_test = filtering_matrices(full_attn_weights_test,
                                                                                                all_article_identifiers_test, sent_lengths_test,
                                                                                                df_test, print_samples=num_print,
                                                                                                degree_std=std, with_filtering=filtering, filtering_type=filter_type, granularity=granularity)

filter_type = "max"
print(f'Filtering type: {filter_type}')

# Train
max_filtered_matrices, max_total_nodes, max_total_edges, max_deletions = filtering_matrices(full_attn_weights_t, all_article_identifiers_t, sent_lengths,
                                                                            df_full_train, print_samples=num_print,
                                                                            degree_std=std, filtering_type=filter_type, granularity=granularity)

# Test
max_filtered_matrices_test, max_total_nodes_test, max_total_edges_test, max_deletions_test = filtering_matrices(full_attn_weights_test,
                                                                                                all_article_identifiers_test, sent_lengths_test,
                                                                                                df_test, print_samples=num_print,
                                                                                                degree_std=std, filtering_type=filter_type, granularity=granularity)


plt.hist(x=[deletions, deletions_test, max_deletions, max_deletions_test], bins=50, color=['blue', 'lightblue', 'red', 'orange'])
plt.xlim(0, 7500)
plt.legend(["Mean - Train", "Mean - Test", "Max - Train", "Max - Test"])
plt.title("Deletions for different local-filtering strategies - Model: "+model_name)
plt.show()
