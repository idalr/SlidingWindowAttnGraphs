import yaml
import argparse
from graph_data_loaders import UnifiedAttentionGraphs_Sum  # , AttentionGraphs_Sum
import torch
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from eval_models import retrieve_parameters, eval_results
from preprocess_data import load_data
from data_loaders import create_loaders, check_dataframe, get_class_weights
from base_model import MHASummarizer  # , MHASummarizer_extended

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
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
    path_models = path_logger+model_name+"/"

    if unified_flag == True:
        path_root = os.path.join(root_graph, model_name,type_graph + "_unified")  # root_graph+model_name+"/Attention/"+type_graph+ "_unified"
        project_name = model_name + "2" + type_model + "_" + type_graph + "_unified"
        file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph + "_unified")
    else:
        path_root = os.path.join(root_graph, model_name, type_graph)  # root_graph+model_name+"/Attention/"+type_graph
        project_name = model_name + "2" + type_model + "_" + type_graph
        file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph)

    ### Filename for graph datasets --- stored in path_root/raw/
    if config_file["load_data_paths"]["with_val"] == True:
        filename_val = "post_predict_val_documents.csv"
    filename_train = "post_predict_train_documents.csv"
    filename_test = "post_predict_test_documents.csv"

    try:
        print("Running", type_model, "on Attention-based document graphs.")
        print("Loading graphs from:", path_root)

        if unified_flag == True:
            dataset_train = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_train,
                                                         filter_type=type_graph, data_loader=None, mode="train",
                                                         binarized=flag_binary, multi_layer_model=multi_flag)
            if config_file["load_data_paths"]["with_val"] == True:
                dataset_val = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_val,
                                                           filter_type=type_graph, data_loader=None, mode="val",
                                                           binarized=flag_binary, multi_layer_model=multi_flag)
            dataset_test = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_test,
                                                        filter_type=type_graph, data_loader=None, mode="test",
                                                        binarized=flag_binary, multi_layer_model=multi_flag)
            print("Graphs with unified sentence nodes correctly loaded.")

        else:
            print(
                "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
            """Preliminary experiments show that the performance of UnifiedAttentionGraphs_Class is better than non-unified AttentionGraphs."""

        if config_file["load_data_paths"]["with_val"] == True:
            df_train, df_val, df_test = load_data(**config_file["load_data_paths"])
        else:
            df_train, _ = load_data(**config_file["load_data_paths"])

        if config_file["with_cw"] == True:
            my_class_weights, labels_counter = get_class_weights(df_train)
            calculated_cw = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            calculated_cw = None

    except:
        print("\nError loading dataset - No Graph Dataset found")
        print("\nCreating new dataset from pre-trained MHA model")
        print("Pre-trained model:", model_name)
        print("Type graph:", type_graph)
        print("Root graph:", path_root)
        print("Loading data...")

        if config_file["load_data_paths"]["with_val"] == True:
            df_train, df_val, df_test = load_data(**config_file["load_data_paths"])
        else:
            df_train, df_test = load_data(**config_file["load_data_paths"])

        ids2remove_train = check_dataframe(df_train, task='classification')
        for id_remove in ids2remove_train:
            df_train = df_train.drop(id_remove)
        df_train.reset_index(drop=True, inplace=True)
        print("Train shape:", df_train.shape)

        if config_file["load_data_paths"]["with_val"] == True:
            ids2remove_val = check_dataframe(df_val, task='classification')
            for id_remove in ids2remove_val:
                df_val = df_val.drop(id_remove)
            df_val.reset_index(drop=True, inplace=True)
            print("Val shape:", df_val.shape)

        ids2remove_test = check_dataframe(df_test, task='classification')
        for id_remove in ids2remove_test:
            df_test = df_test.drop(id_remove)
        df_test.reset_index(drop=True, inplace=True)
        print("Test shape:", df_test.shape)

        if config_file["with_cw"] == True:
            my_class_weights, labels_counter = get_class_weights(df_train)
            calculated_cw = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            calculated_cw = None

        try:
            max_len = int(config_file["max_len"])
            print("Max number of sentences given on config file.")
        except:
            print("Calculating max number of sentences...")
            sent_lengths = []
            for i, doc in enumerate(df_train['article_text']):
                sent_in_doc = sent_tokenize(doc)
                sent_lengths.append(len(sent_in_doc))
            max_len = max(sent_lengths)

        print("Max number of sentences allowed:", max_len)

        if config_file["load_data_paths"]["with_val"] == True:
            loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                            config_file["batch_size"],
                                                                            tokenizer_from_scratch=False,
                                                                            path_ckpt=path_vocab,
                                                                            df_val=df_val, task="summarization")
        else:
            loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, config_file["batch_size"],
                                                            with_val=False, tokenizer_from_scratch=False,
                                                            path_ckpt=path_vocab,
                                                            df_val=None, task="summarization")

        print("\nLoading", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
        model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
        print("Done")

        ### Model performance in validation and test partitions -- register results on file_results.txt
        if config_file["load_data_paths"]["with_val"] == True:
            preds_v, full_attn_weights_v, all_labels_v, all_doc_ids_v, all_article_identifiers_v = model_lightning.predict(
                loader_val, cpu_store=False)
            acc_v, f1_all_v = eval_results(preds_v, all_labels_v, num_classes, "Val")
            if unified_flag == True:
                del full_attn_weights_v

        preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(
            loader_test, cpu_store=False)
        acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")
        if unified_flag == True:
            del full_attn_weights_test

        if config_file["load_data_paths"]["with_val"] == True:
            filename_val = "post_predict_val_documents.csv"
        filename_train = "post_predict_train_documents.csv"
        filename_test = "post_predict_test_documents.csv"

    path_filename_train = os.path.join(path_root, "raw", filename_train)
    if os.path.exists(path_filename_train):
        print("File Requirements already satified in ", path_root + "/raw/")
    else:
        path_dataset = path_root + "/raw/"
        print("\nCreating files for PyG dataset in:", path_dataset)
        df_logger = pd.read_csv(path_logger + logger_name)

        ### Forward pass to get predictions from loaded MHA model
        print("Predicting Train")
        _, _, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(loader_train,
                                                                                               cpu_store=False,
                                                                                               flag_file=True)
        post_predict_train_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
        post_predict_train_docs.to_csv(path_dataset + filename_train, index=False)
        for article_id, label, doc_as_ids in zip(all_article_identifiers_t, all_labels_t, all_doc_ids_t):
            post_predict_train_docs.loc[len(post_predict_train_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
            }
        post_predict_train_docs.to_csv(path_dataset + filename_train, index=False)
        print("Finished and saved in:", path_dataset + filename_train)

        if config_file["load_data_paths"]["with_val"] == True:
            print("\nPredicting Val")
            _, _, all_labels_v, all_doc_ids_v, all_article_identifiers_v = model_lightning.predict(loader_val,
                                                                                                   cpu_store=False,
                                                                                                   flag_file=True)
            post_predict_val_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
            post_predict_val_docs.to_csv(path_dataset + filename_val, index=False)
            for article_id, label, doc_as_ids in zip(all_article_identifiers_v, all_labels_v, all_doc_ids_v):
                post_predict_val_docs.loc[len(post_predict_val_docs)] = {
                    "article_id": article_id.item(),
                    "label": label.item(),
                    "doc_as_ids": doc_as_ids.tolist()
                }
            post_predict_val_docs.to_csv(path_dataset + filename_val, index=False)
            print("Finished and saved in:", path_dataset + filename_val)

        print("\nPredicting Test")
        _, _, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(
            loader_test, cpu_store=False, flag_file=True)
        post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
        post_predict_test_docs.to_csv(path_dataset + filename_test, index=False)
        for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test,
                                                 all_doc_ids_test):
            post_predict_test_docs.loc[len(post_predict_test_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
            }
        post_predict_test_docs.to_csv(path_dataset + filename_test, index=False)
        print("Finished and saved in:", path_dataset + filename_test)

    if type_graph == "full":
        filter_type = None
    else:
        filter_type = type_graph

    with open(file_results + '.txt', 'a') as f:
        try:
            print("\nLoading pre-trained", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
            print("Loading 1L-MHASummarizer...")
            model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
            print("Done.")

            print("Model temperature", model_lightning.temperature)
            print("Evaluating VAL and TEST loaders...")
            max_len = config_file["max_len"]
            _, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                    config_file["batch_size"], df_val=df_val,
                                                                    task="summarization",
                                                                    tokenizer_from_scratch=False,
                                                                    path_ckpt=path_vocab)
            preds_v, _, all_labels_v, _, _ = model_lightning.predict(loader_val, cpu_store=False)
            preds_test, _, all_labels_test, _, _ = model_lightning.predict(loader_test, cpu_store=False)
            acc_v, f1_all_v = eval_results(preds_v, all_labels_v, num_classes, "Val")
            acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")

            print("================================================", file=f)
            print("Evaluation of pre-trained model:", model_name, file=f)
            print("================================================", file=f)
            print("[VAL] Acc:", acc_v, file=f)
            print("[VAL] F1 scores:", f1_all_v.mean(), file=f)
            print("[VAL] F1 scores:", f1_all_v, file=f)  # print train pre-trained model
            print("[TEST] Acc:", acc_test, file=f)
            print("[TEST] F1 scores:", f1_all_test.mean(), file=f)
            print("[TEST] F1 scores:", f1_all_test, file=f)  # print test pre-trained model
            print("Results saved on", file_results + '.txt')

        except:
            pass

        print("================================================", file=f)
        print("Attention Graph Creation/Loading Time:", filter_type, file=f)
        print("================================================", file=f)
        start_creation = time.time()
        if unified_flag == True:
            dataset_train = UnifiedAttentionGraphs_Sum(path_root, filename_train, filter_type, loader_train,
                                                       model_ckpt=path_checkpoint, mode="train",
                                                       binarized=flag_binary, multi_layer_model=multi_flag)
        else:
            print(
                "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
            print("Aborting process.")
            return
        creation_train = time.time() - start_creation
        print("Creation time for Train Graph dataset:", creation_train, file=f)
        print("Creation time for Train Graph dataset:", creation_train)

        # graphs val
        start_creation = time.time()
        if config_file["load_data_paths"]["with_val"] == True:
            start_creation = time.time()
            if unified_flag == True:
                dataset_val = UnifiedAttentionGraphs_Sum(path_root, filename_val, filter_type, loader_val,
                                                         model_ckpt=path_checkpoint, mode="val",
                                                         binarized=flag_binary, multi_layer_model=multi_flag)
        else:
            print(
                "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
            print("Aborting process.")
            return
        creation_val = time.time() - start_creation
        print("Creation time for Val Graph dataset:", creation_val, file=f)
        print("Creation time for Val Graph dataset:", creation_val)

        start_creation = time.time()
        if unified_flag == True:
            dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, filter_type, loader_test,
                                                       model_ckpt=path_checkpoint, mode="test",
                                                       binarized=flag_binary, multi_layer_model=multi_flag)
        else:
            print(
                "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
            print("Aborting process.")
            return
        creation_test = time.time() - start_creation
        print("Creation time for Test Graph dataset:", creation_test, file=f)
        print("Creation time for Test Graph dataset:", creation_test)
        print("================================================", file=f)
        f.close()


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
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file)