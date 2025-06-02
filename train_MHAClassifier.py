import yaml
import argparse
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from torchmetrics import F1Score
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from nltk.tokenize import sent_tokenize

from preprocess_data import load_data
from base_model import MHAClassifier #, MHAClassifier_extended
from data_loaders import create_loaders, get_class_weights

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    path_models = config_file["path_models"]
    file_to_save = config_file["file_to_save"]
    model_params = config_file["model_arch_args"]
    logger_name = config_file["logger_name"]
    logger_file= config_file["logger_file"]

    if config_file["load_data_paths"]["with_val"] == True:
        df_train, df_val, df_test = load_data(**config_file["load_data_paths"])
    else:
        df_train, df_test = load_data(**config_file["load_data_paths"])

    print("df_train", df_train.shape)
    if config_file["load_data_paths"]["with_val"] == True:
        print("df_val", df_val.shape)
    print("df_test", df_test.shape)

    #df_train, df_val, df_test =df_train[:40], df_val[:40], df_test[:40]

    ### OBTAIN MAX SEQUENCE
    sent_lengths = []
    for i, doc in enumerate(df_train['article_text']):
        sent_in_doc = sent_tokenize(doc)
        if len(sent_in_doc) == 0:
            print("Empty doc en:", i)
        sent_lengths.append(len(sent_in_doc))
    max_len = max(sent_lengths)  # Maximum number of sentences in a document
    print("max number of sentences in document:", max_len)

    for exec_i in range(config_file["num_executions"]):
        print("\n=============================")
        print("Execution number:", exec_i)
        print("=============================")
        try:
            if config_file["load_data_paths"]["with_val"] == True:
                loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                                   config_file["batch_size"],
                                                                                   df_val=df_val, task="classification",
                                                                                   tokenizer_from_scratch=False,
                                                                                   path_ckpt=
                                                                                   config_file["load_data_paths"][
                                                                                       "in_path"])
            else:
                loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                                   config_file["batch_size"],
                                                                                   task="classification",
                                                                                   tokenizer_from_scratch=False,
                                                                                   path_ckpt=
                                                                                   config_file["load_data_paths"][
                                                                                       "in_path"])
        except:
            #### Create sentence dictionary-vocabulary
            print("Error: Vocabulary not found.\nCreating sentence vocabulary...")
            if config_file["load_data_paths"]["with_val"] == True:
                loader_train, loader_val, loader_test, _, _, _, invert_vocab_sent = create_loaders(df_train, df_test,
                                                                                                   max_len, config_file[
                                                                                                       "batch_size"],
                                                                                                   df_val=df_val,
                                                                                                   task="classification")
            else:
                loader_train, loader_val, loader_test, _, _, _, invert_vocab_sent = create_loaders(df_train, df_test,
                                                                                                   max_len, config_file[
                                                                                                       "batch_size"],
                                                                                                   task="classification")
            # save sentence dictionary-vocabulary on disk
            sent_dict = pd.DataFrame(
                data={'Sentence_id': list(invert_vocab_sent.keys()), 'Sentence': list(invert_vocab_sent.values())})
            sent_dict.to_csv(config_file["load_data_paths"]["in_path"] + "vocab_sentences.csv", index=False)

        if config_file["with_cw"]:
            my_class_weights, labels_counter = get_class_weights(df_train)
            model_params["class_weights"] = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            model_params["class_weights"] = None

        ## TRAINING SETUP
        model_params["max_len"]=max_len
        model_params["path_invert_vocab_sent"]= config_file["load_data_paths"]["in_path"]
        #if model_params["multi_layer"]:
        #    model_lightning = MHAClassifier_extended(**model_params)
        #else:
        model_lightning= MHAClassifier(**model_params)

        early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True, **config_file["early_args"])
        checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                              dirpath=path_models + logger_name,
                                              filename=logger_name + "-{epoch:02d}-{Val_f1-ma:.2f}")
        wandb_logger = WandbLogger(name=logger_name, save_dir=path_models, project=file_to_save)
        trainer = pl.Trainer(accelerator=config_file.get("device", "gpu"), devices=1,
                             callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger,
                             **config_file["trainer_args"])

        ###### TRAINING ######
        start_time = time.time()
        trainer.fit(model_lightning, loader_train, loader_val)
        stopped_on = trainer.callbacks[0].stopped_epoch
        print("\nTraining stopped on epoch:", stopped_on)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} secs")

        # load best checkpoint
        #if model_params["multi_layer"]:
        #    model_lightning = MHAClassifier_extended.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        #else:
        model_lightning = MHAClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            
        print ("Model loaded from:", trainer.checkpoint_callback.best_model_path)
        print ("Temperature of loaded model:", model_lightning.temperature)

        ###### TESTING ######
        preds, _, all_labels, _, _ = model_lightning.predict(loader_test, cpu_store=False)

        acc = (torch.Tensor(all_labels) == preds).float().mean()
        f1_score = F1Score(task='multiclass', num_classes=model_params["num_classes"], average=None)
        f1_all = f1_score(preds.int(), torch.Tensor(all_labels).int())
        print("Acc Test:", acc)
        print("F1-macro Test:", f1_all.mean())
        print("F1 for each class:", f1_all)
        total_time = time.time() - start_time
        print(f"Training + Testing time: {total_time:.2f} secs")
        print(f"Total running time: {time.time() - start_time:.2f} secs")

        path_models_logger_file = os.path.join(path_models, logger_file)

        if not os.path.exists(path_models_logger_file):
            df_logger = pd.DataFrame(columns=["Model", "Path", "Score", "Test score" , "Setting", "Stop epoch", "Temperature", "Window percent", "Training_time", "Total_time"])
            df_logger.to_csv(path_models_logger_file, index=False)
        else: #if exist
            df_logger = pd.read_csv(path_models_logger_file)
        df_logger.loc[len(df_logger)] = {
            "Model": logger_name,
            "Path": trainer.checkpoint_callback.best_model_path,
            "Score": trainer.checkpoint_callback.best_model_score.cpu().numpy(),
            "Test score": f1_all.mean().item(),
            "Setting": settings_file,
            "Stop epoch": stopped_on,
            "Temperature": model_lightning.temperature,
            "Window percent": model_lightning.window,
            "Training_time": train_time,
            "Total_time": total_time        
            }
        df_logger.to_csv(path_models_logger_file, index=False)
        print ("Finished and saved in:", path_models_logger_file, "\n")


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