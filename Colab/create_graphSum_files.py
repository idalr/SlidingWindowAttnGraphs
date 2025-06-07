#############################################
# Colab version
import os
import sys

# Get the absolute path of the repository root
repo_root = os.path.abspath(os.path.join('/content', 'Attention-DocumentGraphs-Rin'))

# Add the root path to the search path
if repo_root not in sys.path:
    sys.path.append(repo_root)
#############################################
# TODO: optimize runtime in graph creation
#############################################

import yaml
import argparse
from Colab.graph_data_loaders import UnifiedAttentionGraphs_Sum  # , AttentionGraphs_Sum
import torch
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from eval_models import retrieve_parameters
from preprocess_data import load_data
from data_loaders import create_loaders, check_dataframe
from Colab.base_model import MHASummarizer
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    logger_name = config_file["logger_name"]  # "df_logger_cw.csv"
    model_name = config_file["model_name"]  # Extended_Anneal

    path_logger = config_file["data_paths"][
        "path_logger"]  # path_logger = "/scratch/mbugueno/HomoGraphs_GovReports/"
    root_graph = config_file["data_paths"]["root_graph_dataset"]  # "/scratch/datasets/AttnGraphs_GovReports/"
    folder_results = config_file["data_paths"]["results_folder"]  # "/home/mbugueno/AttGraphs/GNN_Results_Summarizer/"
    save_flag = config_file["saving_file"]
    unified_flag = config_file["unified_nodes"]
    flag_binary = config_file["binarized"]
    file_results = folder_results + "Doc_Level_Results"

    # create folder if not exist
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    path_df_logger = os.path.join(path_logger, logger_name)
    df_logger = pd.read_csv(path_df_logger)

    if flag_binary:
        path_root = os.path.join(root_graph, model_name, config_file["type_graph"] + '_binary')
    else:
        if unified_flag:
            path_root = os.path.join(root_graph, model_name, config_file["type_graph"] + "_unified")
        else:
            path_root = os.path.join(root_graph, model_name, config_file["type_graph"])

    filename_train = "predict_train_documents.csv"
    filename_val = "predict_val_documents.csv"
    filename_test = "predict_test_documents.csv"

    # import loader_data
    print("Loading data...")
    if config_file["load_data_paths"]["with_val"]:
        df_train, df_val, df_test = load_data(**config_file["load_data_paths"])

    #     ids2remove_val = check_dataframe(df_val)
    #     for id_remove in ids2remove_val:
    #         df_val = df_val.drop(id_remove)
    #     df_val.reset_index(drop=True, inplace=True)
    #     print ("Val shape:", df_val.shape)
    #
    # else:
    #     df_train, df_test = load_data(**config_file["load_data_paths"])
    #
    # ids2remove_train= check_dataframe(df_train)
    # for id_remove in ids2remove_train:
    #     df_train = df_train.drop(id_remove)
    # df_train.reset_index(drop=True, inplace=True)
    # print ("Train shape:", df_train.shape)
    #
    # ids2remove_test = check_dataframe(df_test)
    # for id_remove in ids2remove_test:
    #     df_test = df_test.drop(id_remove)
    # df_test.reset_index(drop=True, inplace=True)
    # print ("Test shape:", df_test.shape)

    max_len = config_file["max_len"]  # Maximum number of sentences in a document
    print("Max number of sentences allowed in document:", max_len)

    #################################minirun
    df_train, df_test, df_val = df_train[:20], df_test[:20], df_val[:20]
    #################################minirun

    if config_file["load_data_paths"]["with_val"]:
        loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len, 1, df_val=df_val,
                                                                           task="summarization",
                                                                           tokenizer_from_scratch=False,
                                                                           path_ckpt=config_file["load_data_paths"][
                                                                               "in_path"])
    else:
        loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, 1, with_val=False,
                                                         task="summarization", tokenizer_from_scratch=False,
                                                         path_ckpt=config_file["load_data_paths"]["in_path"])

    ####################
    path_checkpoint = '/content/drive/MyDrive/ADG/HomoGraphs_GovReports/Extended_NoTemp/Extended_NoTemp-epoch=11-Val_f1-ma=0.52.ckpt'
    model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
    # need to load invert_vocab_sent manually for model_lightning
    sent_dict_disk = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/GovReport-Sum/vocab_sentences.csv')
    invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
    model_lightning.invert_vocab_sent = invert_vocab_sent
    print("invert_vocab_sent Loaded.")
    ####################

    print("Model temperature", model_lightning.temperature)
    print("Done")

    path_filename_train = os.path.join(path_root, "raw", filename_train)
    if os.path.exists(path_filename_train):
        print("Requirements satisfied in:", path_root)
    else:  # if not os.path.exists(path_filename_train)::
        if save_flag:
            print("\nCreating required files for PyTorch Geometric Dataset Creation from pre-trained MHASummarizer...")
        else:
            print("\nObtaining predictions from pre-trained Single-layer MHASummarizer...")

        with open(file_results + '_' + model_name + '.txt', 'a') as f:
            if save_flag:
                print("\n\n================================================", file=f)
                print("Creating prediction files from pre-trained model:", model_name, file=f)
                print("Creating prediction files from pre-trained model:", model_name)
                print("================================================", file=f)
            else:
                print("\n\n================================================", file=f)
                print("Obtaining predictions from pre-trained model:", model_name, file=f)
                print("Obtaining predictions from pre-trained model:", model_name)
                print("================================================", file=f)
            start_creation = time.time()
            partition = "TRAIN"
            accs_tr, f1s_tr = model_lightning.predict_to_file(loader_train, saving_file=save_flag,
                                                              filename=filename_train,
                                                              path_root=path_root)  # /scratch/datasets/AttnGraphs_GovReports/Extended_Anneal/Attention/mean
            creation_time = time.time() - start_creation
            print("[" + partition + "] File creation time:", creation_time, file=f)
            print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item(), file=f)
            print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy(),
                  file=f)
            print("================================================", file=f)
            print("[" + partition + "] File creation time:", creation_time)
            print("[" + partition + "] Avg. Acc (doc-level):", torch.tensor(accs_tr).mean().item())
            print("[" + partition + "] Avg. F1-score (doc-level):", torch.stack(f1s_tr, dim=0).mean(dim=0).numpy())
            print("================================================")

            if config_file["load_data_paths"]["with_val"]:
                start_creation = time.time()
                partition = "VAL"
                accs_v, f1s_v = model_lightning.predict_to_file(loader_val, saving_file=save_flag,
                                                                filename=filename_val, path_root=path_root)
                creation_time = time.time() - start_creation
                print("[" + partition + "] File creation time:", creation_time, file=f)
                print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item(), file=f)
                print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy(), file=f)
                print("================================================", file=f)
                print("[" + partition + "] File creation time:", creation_time)
                print("[" + partition + "] Avg. Acc:", torch.tensor(accs_v).mean().item())
                print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_v, dim=0).mean(dim=0).numpy())
                print("================================================")
            else:
                pass

            start_creation = time.time()
            partition = "TEST"
            accs_t, f1s_t = model_lightning.predict_to_file(loader_test, saving_file=save_flag, filename=filename_test,
                                                            path_root=path_root)
            creation_time = time.time() - start_creation
            print("[" + partition + "] File creation time:", creation_time, file=f)
            print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item(), file=f)
            print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy(), file=f)
            print("================================================", file=f)
            print("[" + partition + "] File creation time:", creation_time)
            print("[" + partition + "] Avg. Acc:", torch.tensor(accs_t).mean().item())
            print("[" + partition + "] Avg. F1-score:", torch.stack(f1s_t, dim=0).mean(dim=0).numpy())
            print("================================================")
            f.close()

        print("Finished and saved in:", path_root)

    with open(file_results + '_' + model_name + '.txt', 'a') as f:

        if config_file["type_graph"] == "full":
            filter_type = None
        else:
            filter_type = config_file["type_graph"]

        tolerance = 0.5
        print("\n\n-----------------------------------------------", file=f)
        if flag_binary:
            print("Creating graphs with filter:", config_file["type_graph"], "-- binarized", file=f)
            print("Creating graphs with filter:", config_file["type_graph"], "-- binarized")
        else:
            print("Creating graphs with filter:", config_file["type_graph"], file=f)
            print("Creating graphs with filter:", config_file["type_graph"])
        print("-----------------------------------------------", file=f)
        start_creation = time.time()
        filename_train = "predict_train_documents.csv"
        dataset_train = UnifiedAttentionGraphs_Sum(path_root, filename_train, filter_type, loader_train,
                                                   degree=tolerance, model=model_lightning, mode="train",
                                                   binarized=flag_binary)  # , multi_layer_model=multi_flag)
        creation_train = time.time() - start_creation
        print("Creation time for Train Graph dataset:", creation_train, file=f)
        print("Creation time for Train Graph dataset:", creation_train)

        # graphs val
        start_creation = time.time()
        filename_val = "predict_val_documents.csv"
        dataset_val = UnifiedAttentionGraphs_Sum(path_root, filename_val, filter_type, loader_val, degree=tolerance,
                                                 model=model_lightning, mode="val",
                                                 binarized=flag_binary)  # , multi_layer_model=multi_flag)
        creation_val = time.time() - start_creation
        print("Creation time for Val Graph dataset:", creation_val, file=f)
        print("Creation time for Val Graph dataset:", creation_val)

        start_creation = time.time()
        filename_test = "predict_test_documents.csv"
        dataset_test = UnifiedAttentionGraphs_Sum(path_root, filename_test, filter_type, loader_test, degree=tolerance,
                                                  model=model_lightning, mode="test",
                                                  binarized=flag_binary)  # , multi_layer_model=multi_flag)
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

    mp.set_start_method("spawn", force=True)
    main_run(config_file, args.settings_file)