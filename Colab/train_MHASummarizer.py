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
import os
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from torchmetrics import F1Score
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from preprocess_data import load_data
from base_model import MHASummarizer
from data_loaders import create_loaders, check_dataframe, get_class_weights

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def main_run(config_file, settings_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    path_models = config_file["path_models"]
    file_to_save = config_file["file_to_save"]
    model_params = config_file["model_arch_args"]
    logger_name = config_file["logger_name"]
    logger_file = config_file["logger_file"]

    df_train, df_val, df_test = load_data(**config_file["data_paths"])

    ids2remove_train = check_dataframe(df_train)
    for id_remove in ids2remove_train:
        df_train = df_train.drop(id_remove)
    df_train.reset_index(drop=True, inplace=True)
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

    max_len = config_file["max_len"]  # Maximum number of sentences in a document
    print("Max number of sentences allowed in document:", max_len)

    for exec_i in range(config_file["num_executions"]):
        print("\n=============================")
        print("Execution number:", exec_i)
        print("=============================")

        """
        #### Create sentence dictionary-vocabulary
        loader_train, loader_val, loader_test, _ , _ , vocab_sent, invert_vocab_sent = create_loaders(df_train, df_test, max_len, config_file["batch_size"], df_val=df_val, task="summarization")
        #save sentence dictionary-vocabulary on disk
        sent_dict = pd.DataFrame(data={'Sentence_id': list(invert_vocab_sent.keys()), 'Sentence': list(invert_vocab_sent.values())})
        sent_dict.to_csv(in_path+"vocab_sentences.csv", index=False)
        """

        loader_train, loader_val, loader_test, _, _, _, _ = create_loaders(df_train, df_test, max_len,
                                                                           config_file["batch_size"],
                                                                           df_val=df_val, task="summarization",
                                                                           tokenizer_from_scratch=False,
                                                                           path_ckpt=config_file["data_paths"][
                                                                               "in_path"])

        if config_file["with_cw"]:
            my_class_weights, labels_counter = get_class_weights(df_train, task="summarization")
            model_params["class_weights"] = my_class_weights
            print("\nClass weights - from training partition:", my_class_weights)
            print("Class counter:", labels_counter)
        else:
            print("\n-- No class weights specificied --\n")
            model_params["class_weights"] = None

        ## TRAINING SETUP
        early_stop_callback = EarlyStopping(monitor="Val_f1-ma", mode="max", verbose=True, **config_file["early_args"])
        wandb_logger = WandbLogger(name=logger_name, save_dir=path_models, project= file_to_save)

        # RESUME RUNNING
        if config_file['last_checkpoint']:
            checkpoint_file = os.path.join(config_file['path_models'],config_file['logger_name'], config_file['last_checkpoint'])
            last_epoch = config_file['last_epoch']+1
            print("Resuming from checkpoint:", checkpoint_file)
            model_lightning = MHASummarizer.load_from_checkpoint(checkpoint_file)
            checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                  dirpath=path_models + logger_name,
                                                  filename=config_file['last_checkpoint'] + "-{epoch:02d}-{Val_f1-ma:.2f}")
            trainer = pl.Trainer(accelerator=config_file.get("device", "gpu"), devices=1,
                                 callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger,
                                 max_epochs=config_file['trainer_args']['max_epochs']-last_epoch,
                                 enable_progress_bar=config_file['trainer_args']['enable_progress_bar'])


        ## NEW TRAINING SETUP
        else:
            model_params["max_len"] = max_len
            model_params["path_invert_vocab_sent"] = config_file["data_paths"]["in_path"]
            # if model_params["multi_layer"]:
            #    model_lightning = MHAClassifier_extended(**model_params)
            # else:
            model_lightning = MHASummarizer(**model_params)  # activation_attention="softmax"
            checkpoint_callback = ModelCheckpoint(monitor="Val_f1-ma", mode="max", save_top_k=1,
                                                  dirpath=path_models + logger_name,
                                                  filename=logger_name + "-{epoch:02d}-{Val_f1-ma:.2f}")
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
        model_lightning = MHASummarizer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        print("Model loaded from:", trainer.checkpoint_callback.best_model_path)
        print("Temperature of loaded model:", model_lightning.temperature)

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

        if not os.path.exists(path_models_logger_file):  # "df_logger.csv"
            df_logger = pd.DataFrame(
                columns=["Model", "Path", "Score", "Test score", "Setting", "Stop epoch", "Temperature",
                         "Window percent", "Training_time", "Total_time"])
            df_logger.to_csv(path_models_logger_file, index=False)
        else:  # if exist
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
        print("\nFinished and saved in:", path_models_logger_file, "\n")


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