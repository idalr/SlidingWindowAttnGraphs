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

import yaml
import argparse
from Colab.graph_data_loaders import AttentionGraphs, UnifiedAttentionGraphs_Class
import torch
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from nltk.tokenize import sent_tokenize
from src.graphs.gnn_model import GAT_model, partitions
from Colab.base_model import MHAClassifier
from src.pipeline.eval_models import retrieve_parameters, eval_results
from src.data.text_loaders import create_loaders, get_class_weights, check_dataframe

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    # import from yaml config
    logger_name = config_file["logger_name"]  # "df_logger_cw.csv"
    model_name = config_file["model_name"]  # Extended_Anneal
    type_model = config_file["type_model"]  # GAT or GCN
    type_graph = config_file["type_graph"]  # full, mean, max
    flag_binary = config_file["binarized"]  # True or False
    multi_flag = config_file["multi_layer"]  # True for MHASummarizer_extended or False for MHASummarizer
    unified_flag = config_file["unified_nodes"]  # True for graphs with unified nodes or False for standard graphs
    # model configs
    num_classes = config_file["model_arch_args"]["num_classes"]
    lr = config_file["model_arch_args"]["lr"]
    dropout = config_file["model_arch_args"]["dropout"]  # 0.2
    dim_features = config_file["model_arch_args"]["dim_features"]  # [128, 256]
    n_layers = config_file["model_arch_args"]["n_layers"]  # [1, 2]
    num_runs = config_file["model_arch_args"]["num_runs"]  # 5
    # path to store results
    root_graph = config_file["data_paths"]["root_graph_dataset"]  # /content/drive/MyDrive/ADG/HomoGraphs_HND/
    path_results = config_file["data_paths"]["results_folder"]  # GNN_Results/Classifier_Results/
    path_logger = config_file["data_paths"]["path_logger"]  # /content/drive/MyDrive/ADG/AttnGraphs_HND/

    # create folders if not exist
    # if not os.path.exists(root_graph):
    #    os.makedirs(root_graph)
    if not os.path.exists(path_results):
        os.makedirs(os.path.normpath(path_results))
    if not os.path.exists(path_logger):
        os.makedirs(os.path.normpath(path_logger))

    path_df_logger = os.path.normpath(os.path.join(path_logger, logger_name))
    df_logger = pd.read_csv(path_df_logger)

    # if model_name=="Extended_Sigmoid":
    #     path_checkpoint, model_score = retrieve_parameters(model_name, df_logger, require_best=False, retrieve_index=18)
    # else:
    ori_path_checkpoint, model_score, model_window = retrieve_parameters(model_name, df_logger)
    model_version = ori_path_checkpoint.partition(model_name + '\\')[2]
    path_checkpoint = os.path.join(path_logger, model_name, model_version)
    path_checkpoint = os.path.normpath(path_checkpoint)

    file_to_save = model_name + "_" + str(model_score)[:5]
    path_models = os.path.normpath(path_logger + model_name + "/")
    ##os.path.join(path_logger, model_name)
    if config_file["normalized"]:
        pass
        # path_root = os.path.join(root_graph, model_name, type_graph+"_norm")
        # project_name= model_name+"2"+type_model+"_"+type_graph+"_norm"
        # file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph+"_norm")
    else:
        if unified_flag == True:
            path_root = os.path.join(root_graph, model_name,
                                     type_graph + "_unified")  # root_graph+model_name+"/Attention/"+type_graph+ "_unified"
            path_root = os.path.normpath(path_root)
            project_name = model_name + "2" + type_model + "_" + type_graph + "_unified"
            file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph + "_unified")
            file_results = os.path.normpath(file_results)
        else:
            path_root = os.path.join(root_graph, model_name,
                                     type_graph)  # root_graph+model_name+"/Attention/"+type_graph
            path_root = os.path.normpath(path_root)
            project_name = model_name + "2" + type_model + "_" + type_graph
            file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph)
            file_results = os.path.normpath(file_results)

    filename = "post_predict_train_documents.csv"
    filename_test = "post_predict_test_documents.csv"

    try:

        # Cause error, in order to test if Exception also runs
        #error = AttentionGraphs(root=path_root, filename="error.csv", filter_type='', input_matrices=None,
        #                        path_invert_vocab_sent='')

        if unified_flag == True:
            dataset = UnifiedAttentionGraphs_Class(root=path_root, filename=filename,
                                                   filter_type=type_graph, data_loader=None,
                                                   model=None, window=model_window, mode="train",
                                                   binarized=flag_binary, multi_layer_model=multi_flag)
            dataset_test = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_test,
                                                        filter_type=type_graph, model=None, window=model_window, mode="test",
                                                        binarized=flag_binary, multi_layer_model=multi_flag)
        else:
            dataset = AttentionGraphs(root=path_root, filename=filename, filter_type="", input_matrices=None,
                                      path_invert_vocab_sent='', window='', degree=0.5, test=False)
            dataset_test = AttentionGraphs(root=path_root, filename=filename_test, filter_type="", input_matrices=None,
                                           path_invert_vocab_sent='', window='', degree=0.5, test=True)

        load_data_train_path = os.path.normpath(config_file["load_data_paths"]["in_path"] + "/Processed/df_train.csv")
        df_full_train = pd.read_csv(load_data_train_path)
        if config_file["with_cw"]:
            my_class_weights, labels_counter = get_class_weights(df_full_train)
            calculated_cw = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            calculated_cw = None

    except:
        print("Error loading dataset - No Graph Dataset found")
        print("\nCreating new dataset from pre-trained MHA model")
        print("Pre-trained model:", model_name)
        print("Type graph:", type_graph)

        print("Root graph:", path_root)

        print("Loading data...")
        load_data_train_path = os.path.normpath(config_file["load_data_paths"]["in_path"] + "/Processed/df_train.csv")
        df_full_train = pd.read_csv(load_data_train_path)
        load_data_test_path = os.path.normpath(config_file["load_data_paths"]["in_path"] + "/Processed/df_test.csv")
        df_test = pd.read_csv(load_data_test_path)

        sent_lengths = []
        for i, doc in enumerate(df_full_train['article_text']):
            sent_in_doc = sent_tokenize(doc)
            if len(sent_in_doc) == 0:
                print("Empty doc en:", i)
            sent_lengths.append(len(sent_in_doc))
        max_len = max(sent_lengths)  # Maximum number of sentences in a document

        ############################################################################# mini run
        #df_full_train = df_full_train.head(40)
        #df_test = df_test.head(40)
        ############################################################################# mini run

        # filter out data entries that have only one node
        ids2remove_train = check_dataframe(df_full_train, task='classification')
        for id_remove in ids2remove_train:
            df_full_train = df_full_train.drop(id_remove)
        df_full_train.reset_index(drop=True, inplace=True)
        print("Train shape:", df_full_train.shape)

        ids2remove_test = check_dataframe(df_test, task='classification')
        for id_remove in ids2remove_test:
            df_test = df_test.drop(id_remove)
        df_test.reset_index(drop=True, inplace=True)
        print("Test shape:", df_test.shape)

        loader_train, loader_test, _, invert_vocab_sent = create_loaders(df_full_train, df_test, max_len,
                                                                         config_file["batch_size"], with_val=False,
                                                                         task="classification",
                                                                         tokenizer_from_scratch=False,
                                                                         path_ckpt=config_file["load_data_paths"][
                                                                             "in_path"])

        if config_file["with_cw"]:
            my_class_weights, labels_counter = get_class_weights(df_full_train)
            calculated_cw = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            calculated_cw = None

        print("\nLoading", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
        # model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
        sent_dict_disk = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/HyperNews/vocab_sentences.csv')
        invert_vocab_sent = {k: v for k, v in zip(sent_dict_disk['Sentence_id'], sent_dict_disk['Sentence'])}
        model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
        model_lightning.invert_vocab_sent = invert_vocab_sent
        print("Done")

        preds_t, full_attn_weights_t, all_labels_t, all_doc_ids_t, all_article_identifiers_t = model_lightning.predict(
            loader_train, cpu_store=False)
        preds_test, full_attn_weights_test, all_labels_test, all_doc_ids_test, all_article_identifiers_test = model_lightning.predict(
            loader_test, cpu_store=False)

        print("Done with predicting")

        acc_t, f1_all_t = eval_results(preds_t, all_labels_t, num_classes, "Train")
        acc_test, f1_all_test = eval_results(preds_test, all_labels_test, num_classes, "Test")

        path_dataset = os.path.join(path_root, "raw")
        #if not os.path.exists(path_dataset):
        #  os.makedirs(os.path.normpath(path_dataset))
        print("\nCreating files for PyG dataset in:", path_dataset)
        path_dataset_file = os.path.normpath(os.path.join(path_dataset, filename))
        path_dataset_file_test = os.path.normpath(os.path.join(path_dataset, filename_test))

        post_predict_train_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
        post_predict_train_docs.to_csv(path_dataset_file, index=False)
        for article_id, label, doc_as_ids in zip(all_article_identifiers_t, all_labels_t, all_doc_ids_t):
            post_predict_train_docs.loc[len(post_predict_train_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
            }
        post_predict_train_docs.to_csv(path_dataset_file, index=False)
        print("Finished and saved in:", path_dataset_file)

        post_predict_test_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
        post_predict_test_docs.to_csv(path_dataset_file_test, index=False)
        for article_id, label, doc_as_ids in zip(all_article_identifiers_test, all_labels_test, all_doc_ids_test):
            post_predict_test_docs.loc[len(post_predict_test_docs)] = {
                "article_id": article_id.item(),
                "label": label.item(),
                "doc_as_ids": doc_as_ids.tolist()
            }
        post_predict_test_docs.to_csv(path_dataset_file_test, index=False)
        print("Finished and saved in:", path_dataset_file_test)

        #### create data.pt
        if type_graph == "full":
            filter_type = None
        else:
            filter_type = type_graph

        start_creation = time.time()
        if unified_flag:
            dataset = UnifiedAttentionGraphs_Class(root=path_root, filename=filename,
                                                   filter_type=filter_type, data_loader=loader_train,
                                                   model=model_lightning, window=model_window, mode="train",
                                                   binarized=flag_binary, multi_layer_model=multi_flag)
            creation_train = time.time() - start_creation
            start_creation = time.time()
            dataset_test = UnifiedAttentionGraphs_Class(root=path_root, filename=filename_test,
                                                        filter_type=filter_type, data_loader=loader_test,
                                                        model=model_lightning, window=model_window, mode="test",
                                                        binarized=flag_binary, multi_layer_model=multi_flag)
            creation_test = time.time() - start_creation
        else:
            dataset = AttentionGraphs(root=path_root, filename=filename, filter_type=filter_type,
                                      input_matrices=full_attn_weights_t,
                                      path_invert_vocab_sent=config_file["load_data_paths"]["in_path"],
                                      window=model_window, degree=0.5, normalized=config_file["normalized"], test=False)
            creation_train = time.time() - start_creation
            start_creation = time.time()
            dataset_test = AttentionGraphs(root=path_root, filename=filename_test, filter_type=filter_type,
                                           input_matrices=full_attn_weights_test,
                                           path_invert_vocab_sent=config_file["load_data_paths"]["in_path"],
                                           window=model_window, degree=0.5, normalized=config_file["normalized"],
                                           test=True)
            creation_test = time.time() - start_creation

        #### save creation time + base model results in file_results

        with open(file_results + '.txt', 'a') as f:
            print("================================================", file=f)
            print("Evaluation of pre-trained model:", model_name, file=f)
            print("================================================", file=f)
            print("[TRAIN] Acc:", acc_t.item(), file=f)
            print("[TRAIN] F1-macro:", f1_all_t.mean().item(), file=f)
            print("[TRAIN] F1-scores:", f1_all_t, file=f)
            print("------------------------", file=f)
            print("[TEST] Acc:", acc_test.item(), file=f)
            print("[TEST] F1-macro:", f1_all_test.mean().item(), file=f)
            print("[TEST] F1-scores:", f1_all_test, file=f)
            print("================================================", file=f)
            print("[TRAIN] Dataset creation time: ", creation_train, file=f)
            print("[TEST] Dataset creation time: ", creation_test, file=f)
            f.close()

    ### Run GNN models
    start = time.time()
    np.set_printoptions(precision=3)

    if unified_flag:
        type_graph = type_graph + '_unified'

    with open(file_results + '.txt', 'a') as f:

        for nl in n_layers:  # 1 or 2
            for dim in dim_features:  # 128 o 256
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
                        model = GAT_model(dataset.num_node_features, dim, num_classes, nl, lr, dropout=dropout,
                                          class_weights=calculated_cw)
                    #
                    # elif type_model=="GCN":
                    #     model = GCN_model(dataset.num_node_features, dim, num_classes, nl, lr, dropout=dropout, class_weights=calculated_cw)
                    #
                    # else:
                    #     print ("Type of GNN model not supported: No GNN was intended")
                    #     return

                    early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True,
                                                        **config_file["early_args"])

                    path_for_savings = os.path.join(path_models, type_model + type_graph)
                    path_for_savings = os.path.normpath(path_for_savings)

                    if config_file["normalized"]:
                        checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                              dirpath=path_for_savings,
                                                              filename=type_model + "_" + str(nl) + "L_" + str(
                                                                  dim) + "U_" + type_graph + "_norm_run" + str(
                                                                  i) + "-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
                        wandb_logger = WandbLogger(name=model_name + '2' + type_model + "_" + str(nl) + "L_" + str(
                            dim) + "U_" + type_graph + "_norm_run" + str(i), save_dir=path_for_savings + "_norm",
                                                   project=project_name)

                    else:
                        checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                              dirpath=path_for_savings,
                                                              filename=type_model + "_" + str(nl) + "L_" + str(
                                                                  dim) + "U_" + type_graph + "_run" + str(
                                                                  i) + "-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
                        wandb_logger = WandbLogger(name=model_name + '2' + type_model + "_" + str(nl) + "L_" + str(
                            dim) + "U_" + type_graph + "_run" + str(i), save_dir=path_for_savings, project=project_name)

                    trainer = pl.Trainer(accelerator='gpu', devices=1,
                                         callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger,
                                         **config_file["trainer_args"])

                    train_loader, val_loader, test_loader = partitions(dataset, dataset_test, dataset_val=None,
                                                                       bs=config_file["batch_size"])
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
                print("Test Acc: %.3f" % np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)),
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
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file)