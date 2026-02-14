import yaml
import argparse
from graph_data_loaders import UnifiedAttentionGraphs_Class
import torch
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graphs.gnn_model import GAT_model, GCN_model, partitions
from base_model import MHAClassifier
from src.pipeline.eval import eval_results
from src.pipeline.connector import retrieve_parameters
from src.data.preprocess_data import load_data
from src.data.text_loaders import create_loaders
from src.data.utils import get_class_weights, check_dataframe

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file, model_path, model_score):
    # path args
    path_checkpoint = model_path
    score = model_score

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
    file_to_save = model_name + "_" + str(model_score)[:5]
    type_graph = config_file["type_graph"]
    path_models = path_logger + model_name + "/"

    if unified_flag == True:
        path_root = os.path.join(root_graph, model_name,
                                 type_graph + "_unified")  # root_graph+model_name+"/Attention/"+type_graph+ "_unified"
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

    # create folders if not exist
    if not os.path.exists(root_graph):
        os.makedirs(root_graph)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    if not os.path.exists(path_logger):
        os.makedirs(path_logger)

    try:
        print("Running", type_model, "on Attention-based document graphs.")
        print("Loading graphs from:", path_root)

        if unified_flag == True:
            dataset_train = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_train,
                                                         filter_type=type_graph, data_loader=None, mode="train",
                                                         binarized=flag_binary, multi_layer_model=multi_flag)
            if config_file["load_data_paths"]["with_val"] == True:
                dataset_val = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_val,
                                                           filter_type=type_graph, data_loader=None, mode="val",
                                                           binarized=flag_binary, multi_layer_model=multi_flag)
            dataset_test = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_test,
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
                                                                               df_val=df_val, task="classification",
                                                                               sent_tokenizer=True)
        else:
            loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, config_file["batch_size"],
                                                             with_val=False, tokenizer_from_scratch=False,
                                                             path_ckpt=path_vocab,
                                                             df_val=None, task="classification",
                                                             sent_tokenizer=True)

        print("\nLoading", model_name, "(", score, ") from:", path_checkpoint)
        vocab_sent_path = os.path.normpath(
            os.path.join(config_file['load_data_paths']['in_path'], 'vocab_sentences.csv'))
        sent_dict_disk = pd.read_csv(vocab_sent_path)
        invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
        model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
        model_lightning.invert_vocab_sent = invert_vocab_sent
        print("Done")

        ### Model performance in validation and test partitions -- register results on file_results.txt
        if config_file["load_data_paths"]["with_val"] == True:
            preds_v, _, all_labels_v, _, all_article_identifiers_v = model_lightning.ori_predict(
                loader_val, cpu_store=False)
            acc_v, f1_all_v = eval_results(preds_v, all_labels_v, num_classes, "Val")

        preds_test, _, all_labels_test, _, all_article_identifiers_test = model_lightning.ori_predict(
            loader_test, cpu_store=False)
        acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")

        if config_file["load_data_paths"]["with_val"] == True:
            filename_val = "post_predict_val_documents.csv"
        filename_train = "post_predict_train_documents.csv"
        filename_test = "post_predict_test_documents.csv"

        if not os.path.exists(path_root + "/raw/" + filename_train):
            path_dataset = path_root + "/raw/"
            print("\nCreating files for PyG dataset in:", path_dataset)
            if not os.path.exists(path_dataset):
                os.makedirs(path_dataset)
            df_logger = pd.read_csv(path_logger + logger_name)

            # TODO: need to beautify ori_predict

            ### Forward pass to get predictions from loaded MHA model
            print("Predicting Train")
            _, _, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.ori_predict(loader_train,
                                                                                                       cpu_store=False,
                                                                                                       flag_file=True,
                                                                                                       )
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
                _, _, all_labels_v, all_doc_ids_v, all_article_identifiers_v = model_lightning.ori_predict(loader_val,
                                                                                                           cpu_store=False,
                                                                                                           flag_file=True,
                                                                                                           )
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
            _, _, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.ori_predict(
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

        else:
            print("File Requirements already satisfied in ", path_root + "/raw/")

        if type_graph == "full":
            filter_type = None
        else:
            filter_type = type_graph

        with open(file_results + '.txt', 'a') as f:
            print("================================================", file=f)
            print("Evaluation of pre-trained model:", model_name, file=f)
            print("Model checkpoint:", path_checkpoint, file=f)
            print("================================================", file=f)
            if config_file["load_data_paths"]["with_val"] == True:
                print("[VAL] Acc:", acc_v.item(), file=f)
                print("[VAL] F1-macro:", f1_all_v.mean().item(), file=f)
                print("[VAL] F1-scores:", f1_all_v, file=f)
                print("------------------------", file=f)
            print("[TEST] Acc:", acc_test.item(), file=f)
            print("[TEST] F1-macro:", f1_all_test.mean().item(), file=f)
            print("[TEST] F1-scores:", f1_all_test, file=f)
            print("================================================", file=f)

            print("================================================", file=f)
            print("Attention Graph Creation/Loading Time:", filter_type, file=f)
            print("================================================", file=f)
            start_creation = time.time()
            if unified_flag == True:
                dataset_train = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_train,
                                                             filter_type=type_graph, data_loader=loader_train,
                                                             model=model_lightning, mode="train",
                                                             binarized=flag_binary, multi_layer_model=multi_flag)
            else:
                print(
                    "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
                print("Aborting process.")
                return
            creation_train = time.time() - start_creation
            print("[TRAIN] Dataset creation time: ", creation_train, file=f)
            print("[TRAIN] Dataset creation time: ", creation_train)

            if config_file["load_data_paths"]["with_val"] == True:
                start_creation = time.time()
                if unified_flag == True:
                    dataset_val = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_val,
                                                               filter_type=type_graph, data_loader=loader_val,
                                                               model=model_lightning, mode="val",
                                                               binarized=flag_binary, multi_layer_model=multi_flag)
                else:
                    print(
                        "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
                    print("Aborting process.")
                    return
                creation_val = time.time() - start_creation
                print("[VAL] Dataset creation time: ", creation_val, file=f)
                print("[VAL] Dataset creation time: ", creation_val)

            start_creation = time.time()
            if unified_flag == True:
                dataset_test = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_test,
                                                            filter_type=type_graph, data_loader=loader_test,
                                                            model=model_lightning, mode="test",
                                                            binarized=flag_binary, multi_layer_model=multi_flag)
            else:
                print(
                    "Graph creation is only supported for unified sentence nodes. Please set unified_nodes to True in the config file.")
                print("Aborting process.")
                return
            creation_test = time.time() - start_creation
            print("[TEST] Dataset creation time: ", creation_test, file=f)
            print("[TEST] Dataset creation time: ", creation_test)
            f.close()

    ### Run GNN models on graph datasets
    start = time.time()
    np.set_printoptions(precision=3)

    if unified_flag == True:
        graph_construction = type_graph + "_unified"
    else:
        graph_construction = type_graph

    with open(file_results + '.txt', 'a') as f:
        for nl in n_layers:
            for dim in dim_features:
                print("\n\n================================================")
                print("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim)
                print("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim, file=f)
                print("================================================")

                acc_tests = []
                f1_tests = []
                f1ma_tests = []
                all_train_times = []
                all_full_times = []

                for i in range(num_runs):
                    if type_model == "GAT":
                        model = GAT_model(dataset_train.num_node_features, dim, num_classes, nl, lr,
                                          dropout=dropout,
                                          class_weights=calculated_cw)
                    elif type_model == "GCN":
                        model = GCN_model(dataset_train.num_node_features, dim, num_classes, nl, lr,
                                          dropout=dropout,
                                          class_weights=calculated_cw)
                    else:
                        print("Type of GNN model not supported: No GNN was intended")
                        return

                    print("GNN Model Architecture:", model)

                    if config_file["load_data_paths"]["with_val"] == True:
                        train_loader, val_loader, test_loader = partitions(dataset_train, dataset_test,
                                                                           dataset_val=dataset_val,
                                                                           bs=config_file["batch_size"])
                    else:
                        train_loader, val_loader, test_loader = partitions(dataset_train, dataset_test,
                                                                           dataset_val=None,
                                                                           bs=config_file["batch_size"])
                    print("Data loaders created.")
                    print("Training on:", len(train_loader.dataset), "samples")
                    print("Validating on:", len(val_loader.dataset), "samples")

                    early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True,
                                                        **config_file["early_args"])
                    path_for_savings = path_models + type_model + "/" + graph_construction
                    checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                          dirpath=path_for_savings,
                                                          filename=type_model + "_" + str(nl) + "L_" + str(
                                                              dim) + "U_" + graph_construction + "_run" + str(
                                                              i) + "-OUT-{epoch:02d}-{Val_f1-ma:.2f}")

                    wandb_logger = WandbLogger(name=model_name + '2' + type_model + "_" + str(nl) + "L_" + str(
                        dim) + "U_" + graph_construction + "_run" + str(i), save_dir=path_for_savings,
                                               project=project_name)

                    trainer = pl.Trainer(accelerator='gpu', devices=1,
                                         callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger,
                                         **config_file["trainer_args"])

                    starti = time.time()
                    trainer.fit(model, train_loader, val_loader)

                    print("\n----------- Run #" + str(i) + "-----------\n", file=f)
                    print("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch, file=f)
                    train_time = time.time() - starti
                    print(f"Training time: {train_time:.2f} secs", file=f)

                    # load best checkpoint
                    print("Best model path:", trainer.checkpoint_callback.best_model_path, file=f)
                    print("Best model path:", trainer.checkpoint_callback.best_model_path)
                    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
                    model.load_state_dict(checkpoint['state_dict'])

                    preds, trues = model.predict(test_loader, cpu_store=False)
                    acc = (torch.Tensor(trues) == preds).float().mean()
                    f1_score = F1Score(task='multiclass', num_classes=num_classes, average='none')
                    f1_all = f1_score(preds.int(), torch.Tensor(trues).int())
                    print("Acc Test:", acc, file=f)
                    print("F1-macro Test:", f1_all.mean(), file=f)
                    print("F1 for each class:", f1_all, file=f)

                    endi = time.time()
                    total_timei = endi - starti
                    print("Training + Testing time: " + str(total_timei), file=f)
                    acc_tests.append(acc.cpu().numpy())
                    f1_tests.append(f1_all.cpu().numpy())
                    f1ma_tests.append(f1_all.mean().cpu().numpy())
                    all_train_times.append(train_time)
                    all_full_times.append(total_timei)

                print("\n************************************************", file=f)
                print("RESULTS FOR N_LAYERS:", nl, " HIDDEN DIM_FEATURES:", dim, file=f)
                print("Test Acc: %.3f" % np.mean(np.asarray(acc_tests)),
                      "-- std: %.3f" % np.std(np.asarray(acc_tests)),
                      file=f)
                print("Test F1-macro: %.3f" % np.mean(np.asarray(f1ma_tests)),
                      "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
                print("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
                print("Training time: %.2f" % np.mean(np.asarray(all_train_times)),
                      "-- std: %.2f" % np.std(np.asarray(all_train_times)), file=f)
                print("Total time: %.2f" % np.mean(np.asarray(all_full_times)),
                      "-- std: %.2f" % np.std(np.asarray(all_full_times)), file=f)
                print("************************************************\n\n", file=f)

        f.close()

    end = time.time()
    total_time = end - start
    print("\nRUNNING TIME FOR ALL THE EXPERIMENTS: " + str(total_time))


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
        "--model_path",
        required=True,
        type=str,
        help="add model abs path",
    )
    arg_parser.add_argument(
        "--model_score",
        required=True,
        type=str,
        help="add model score",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file, args.model_path, args.model_score)