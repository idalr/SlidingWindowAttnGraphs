import yaml
import argparse

from base_model import MHASummarizer
from gnn_model import GAT_NC_model, partitions
from graph_data_loaders import UnifiedAttentionGraphs_Sum, PartialGraphDataset  # , AttentionGraphs_Sum
import torch
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from eval_models import retrieve_parameters, eval_results
from preprocess_data import load_data
from data_loaders import create_loaders, check_dataframe, get_class_weights
from create_graphSum_files import main_run as create_graphsum

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import F1Score

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    logger_name = config_file["logger_name"]  # "df_logger_cw.csv"
    type_model = config_file["type_model"]  # GAT or GCN
    dataset_name = config_file["dataset_name"]
    path_vocab = config_file["load_data_paths"]["in_path"]
    flag_binary = config_file["binarized"]  # True or False
    multi_flag = config_file["multi_layer"]  # True for MHASummarizer_extended or False for MHASummarizer
    unified_flag = config_file["unified_nodes"]  # True for graphs with unified nodes or False for standard graphs
    MAX_TRAIN_INDEX = config_file.get('max_train_index', False) # If using partial dataset for training

    root_graph = config_file["data_paths"]["root_graph_dataset"]  # "/scratch/datasets/AttnGraphs_GovReports/"
    path_results = config_file["data_paths"]["results_folder"]  # "/home/mbugueno/GNN_Results/Summarizer_Results/Attention/"
    path_logger = config_file["data_paths"]["path_logger"]  # path_logger = "/scratch/mbugueno/HomoGraphs_GovReports/"

    num_classes = config_file["model_arch_args"]["num_classes"]
    lr = config_file["model_arch_args"]["lr"]
    dropout = config_file["model_arch_args"]["dropout"]  # 0.2
    dim_features = config_file["model_arch_args"]["dim_features"]  # [64, 128, 256]
    n_layers = config_file["model_arch_args"]["n_layers"]  # [1, 2, 3]
    num_runs = config_file["model_arch_args"]["num_runs"]  # 5

    model_name = config_file["model_name"]  # Extended_Anneal
    path_logger_name = os.path.join(path_logger, logger_name)
    df_logger = pd.read_csv(path_logger_name)

    # name setting file and model to retrieve from df_logger
    if "Extended_NoTemp" in model_name:
        setting_file = os.path.join("config","Summarizer", "ext_summarizer.yaml")
    else:
        if "Extended_Sigmoid" in model_name:
            model = "sigmoid"
        elif "Extended_Anneal" in model_name:
            model = "anneal"
        elif "Extended_ReLu" in model_name:
            model = "relu"
        elif model_name == "Extended_Step":
            model = "step"
        setting_file = os.path.join("config","Summarizer", "ext_summarizer_" + model + ".yaml")

    print("Matching to setting file:", setting_file)
    tempo = df_logger.where(df_logger['Setting'].str.contains(setting_file)).dropna()
    path_checkpoint, model_score = retrieve_parameters(model_name, tempo)

    file_to_save = model_name + "_" + str(model_score)[:5]
    type_graph = config_file["type_graph"]  # full, mean, max

    path_models = os.path.join(path_logger, model_name)

    if flag_binary == True:
        path_root = os.path.join(root_graph, model_name, type_graph + "_binary")
        project_name = model_name + "2" + type_model + "_" + type_graph + "_binary"
        ###file_results = path_results + dataset_name + "/" + file_to_save + "_2" + type_model + "_" + type_graph + "_binary"
        file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph + "_binary")
    else:
        if unified_flag == True:
            path_root = os.path.join(root_graph, model_name, type_graph + "_unified")
            project_name = model_name + "2" + type_model + "_" + type_graph + "_unified"
            ###file_results = path_results + dataset_name + "/" + file_to_save + "_2" + type_model + "_" + type_graph + "_unified"
            file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph + "_unified")
        else:
            path_root = os.path.join(root_graph, model_name, type_graph)
            project_name = model_name + "2" + type_model + "_" + type_graph
            ###file_results = path_results + dataset_name + "/" + file_to_save + "_2" + type_model + "_" + type_graph
            file_results = os.path.join(path_results, file_to_save + "_2" + type_model + "_" + type_graph)

    filename_train = "predict_train_documents.csv"
    filename_val = "predict_val_documents.csv"
    filename_test = "predict_test_documents.csv"

    print("Running", type_model, "on Attention-based document graphs.")
    print("Loading graphs from:", path_root)
    if unified_flag == True:
        if MAX_TRAIN_INDEX is not None:
            dataset_train = PartialGraphDataset(root=path_root, mode='train')
            dataset_val = PartialGraphDataset(root=path_root, mode='val')
            dataset_test = PartialGraphDataset(root=path_root, mode='test')
        else:
            dataset_train = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_train, filter_type=type_graph,
                                                   data_loader=None, mode="train",
                                                   binarized=flag_binary, multi_layer_model=multi_flag)
            dataset_val = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_val, filter_type=type_graph,
                                                 data_loader=None, mode="val",
                                                 binarized=flag_binary, multi_layer_model=multi_flag)
            dataset_test = UnifiedAttentionGraphs_Sum(root=path_root, filename=filename_test, filter_type=type_graph,
                                                  data_loader=None, mode="test",
                                                  binarized=flag_binary, multi_layer_model=multi_flag)
        print("Graphs with unified sentence nodes correctly loaded.")
    else:
        dataset_train = AttentionGraphs_Sum(root=path_root, filename=filename_train, filter_type=type_graph,
                                                    data_loader=None, mode="train", binarized=flag_binary,
                                                    multi_layer_model=multi_flag)
        dataset_val = AttentionGraphs_Sum(root=path_root, filename=filename_val, filter_type=type_graph,
                                                  data_loader=None, mode="val", binarized=flag_binary,
                                                  multi_layer_model=multi_flag)
        dataset_test = AttentionGraphs_Sum(root=path_root, filename=filename_test, filter_type=type_graph,
                                                   data_loader=None, mode="test", binarized=flag_binary,
                                                   multi_layer_model=multi_flag)
        print("Graphs correctly loaded.")

    ### Run GNN models
    start = time.time()
    np.set_printoptions(precision=3)


    if flag_binary == True:
        graph_construction = type_graph + "_binary"
    else:
        if unified_flag == True:
            graph_construction = type_graph + "_unified"
        else:
            graph_construction = type_graph

    with_edge_attr = True

    #### obtaining class weights
    df_train, df_val, df_test = load_data(**config_file["load_data_paths"])
    ids2remove_train = check_dataframe(df_train)
    for id_remove in ids2remove_train:
        df_train = df_train.drop(id_remove)
    df_train.reset_index(drop=True, inplace=True)
    if MAX_TRAIN_INDEX:
        df_train = df_train[:MAX_TRAIN_INDEX]
    print("Train shape:", df_train.shape)

    ids2remove_val = check_dataframe(df_val)
    for id_remove in ids2remove_val:
        df_val = df_val.drop(id_remove)
    df_val.reset_index(drop=True, inplace=True)
    print("Val shape:", df_val.shape)

    ids2remove_test = check_dataframe(df_test)
    for id_remove in ids2remove_test:
        df_test = df_test.drop(id_remove)
    df_test.reset_index(drop=True, inplace=True)
    print("Test shape:", df_test.shape)

    if config_file["with_cw"] == True:
        my_class_weights, labels_counter = get_class_weights(df_train, task="summarization")
        calculated_cw = my_class_weights
        print("\nClass weights - from training partition:", my_class_weights)
        print("Class counter:", labels_counter)
    else:
        print("\n-- No class weights specificied --\n")
        calculated_cw = None
    ####

    with open(file_results + '.txt', 'a') as f:
        try:
            print("\nLoading pre-trained", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
            print ("Loading 1L-MHASummarizer...")
            model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
            print("Done.")

            print("Model temperature", model_lightning.temperature)
            print("Evaluating VAL and TEST loaders...")
            max_len = config_file["max_len"]
            _, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                    config_file["batch_size"], df_val=df_val,
                                                                    task="summarization", tokenizer_from_scratch=False,
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

        for nl in n_layers:  # 1, 2, 3
            for dim in dim_features:  # 64, 128 o 256
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
                        model = GAT_NC_model(dataset_train.num_node_features, dim, num_classes, nl, lr, dropout=dropout,
                                             class_weights=calculated_cw, with_edge_attr=with_edge_attr)

                    # elif type_model=="GCN":
                    #    model = GCN_model(dataset.num_node_features, dim, num_classes,  nl, lr, dropout=dropout)

                    else:
                        print("Type of GNN model not supported: No GNN was intended")
                        return

                    early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True,
                                                        **config_file["early_args"])
                    path_for_savings = path_models + type_model + "/" + graph_construction
                    # [old] path_for_savings = path_models+"baselines_"+type_model if config_file["baseline"] else path_models+type_model

                    checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                          dirpath=path_for_savings,
                                                          filename=type_model + "_" + str(nl) + "L_" + str(
                                                              dim) + "U_" + graph_construction + "_run" + str(
                                                              i) + "-OUT-{epoch:02d}-{Val_f1-ma:.2f}")
                    if config_file["baseline"]:
                        wandb_logger = WandbLogger(name=type_model + "_" + str(nl) + "L_" + str(
                            dim) + "U_" + graph_construction + "_run" + str(i), save_dir=path_for_savings,
                                                   project=project_name)
                    else:
                        wandb_logger = WandbLogger(name=model_name + '2' + type_model + "_" + str(nl) + "L_" + str(
                            dim) + "U_" + graph_construction + "_run" + str(i), save_dir=path_for_savings,
                                                   project=project_name)

                    trainer = pl.Trainer(accelerator='gpu', devices=1,
                                         accumulate_grad_batches=config_file.get('accum_grad', 1), # 1, or specify
                                         callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger,
                                         **config_file["trainer_args"])

                    train_loader, val_loader, test_loader = partitions(dataset_train, dataset_test,
                                                                       dataset_val=dataset_val, bs=config_file[
                            "batch_size"])  ## try a small number 8-16
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

    create_graphsum(config_file, args.settings_file)
    main_run(config_file, args.settings_file)