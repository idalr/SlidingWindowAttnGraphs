import yaml
import argparse
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import sent_tokenize
from src.models.base_model import MHAClassifier, MHASummarizer
from src.pipeline.connector import retrieve_parameters
from src.data.graph_utils import filtering_matrix
from src.data.preprocess_data import load_data
from src.data.text_loaders import create_loaders
from src.data.utils import check_dataframe

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def main_run(config_file , settings_file, num_print, random):
    ### Load configuration file and set parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = config_file["cuda_visible_devices"]
    logger_name = config_file["logger_name"]
    dataset_name = config_file["dataset_name"]
    path_vocab = config_file["load_data_paths"]["in_path"]
    path_logger = config_file["data_paths"]["path_logger"]

    model_name = config_file["model_name"]
    df_logger = pd.read_csv(path_logger + logger_name)
    path_checkpoint, model_score = retrieve_parameters(model_name, df_logger)
    type_graph = config_file["type_graph"]

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

    sent_lengths = []
    for i, doc in enumerate(df_train['article_text']):
        sent_in_doc = sent_tokenize(doc)
        if len(sent_in_doc) == 0:
            print("Empty doc en:", i)
        sent_lengths.append(len(sent_in_doc))

    if config_file["load_data_paths"]["with_val"] == True:
        sent_lengths_val = []
        for i, doc in enumerate(df_val['article_text']):
            sent_in_doc = sent_tokenize(doc)
            if len(sent_in_doc) == 0:
                print("Empty doc en:", i)
            sent_lengths_val.append(len(sent_in_doc))

    sent_lengths_test = []
    for i, doc in enumerate(df_test['article_text']):
        sent_in_doc = sent_tokenize(doc)
        if len(sent_in_doc) == 0:
            print("Empty doc en:", i)
        sent_lengths_test.append(len(sent_in_doc))

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
                                                                            df_val=df_val, task="classification")
    else:
        loader_train, loader_test, _, _ = create_loaders(df_train, df_test, max_len, config_file["batch_size"],
                                                         with_val=False, tokenizer_from_scratch=False,
                                                         path_ckpt=path_vocab, df_val=None, task="classification")

    print("\nLoading", model_name, "({0:.3f}".format(model_score), ") from:", path_checkpoint)
    if dataset_name == "HND" or dataset_name == "BBC":
        model_lightning = MHAClassifier.load_from_checkpoint(path_checkpoint)
    elif dataset_name == "GR":
        model_lightning = MHASummarizer.load_from_checkpoint(path_checkpoint)
    model_window = model_lightning.window
    print("Done")

    ### Forward pass to get predictions from loaded MHA model
    print("Predicting Train")
    _, full_attn_weights_t, _, _, all_article_identifiers_t = model_lightning.predict(loader_train,
                                                                                               cpu_store=False,
                                                                                               flag_file=True)
    if config_file["load_data_paths"]["with_val"] == True:
        print("\nPredicting Val")
        _, full_attn_weights_v, _, _, all_article_identifiers_v = model_lightning.predict(loader_val,
                                                                                                   cpu_store=False,
                                                                                                   flag_file=True)
    print("\nPredicting Test")
    _, full_attn_weights_test, _, _, all_article_identifiers_test = model_lightning.predict(
        loader_test, cpu_store=False, flag_file=True)

    print("Visualize attentions...")
    if type_graph == 'full':
        filtering = False
        filter_type = 'mean'
    else:
        filtering = True
        filter_type = type_graph

    if random:
        indices_t = list(range(sent_lengths))
        sampled_indices_t = random.sample(indices_t, num_print)
        indices_v = list(range(sent_lengths_val))
        sampled_indices_v = random.sample(indices_v, num_print)
        indices_test = list(range(sent_lengths_test))
        sampled_indices_test = random.sample(indices_test, num_print)

    else:
        sampled_indices_t = range(num_print)
        sampled_indices_v = range(num_print)
        sampled_indices_test = range(num_print)

    for i,j,k in zip(sampled_indices_t, sampled_indices_v, sampled_indices_test):
        _ = filtering_matrix(full_attn_weights_t[i], all_article_identifiers_t[i], sent_lengths[i],
                                window=model_window, with_filtering=filtering, filtering_type=filter_type,
                                plotting=True)
        if config_file["load_data_paths"]["with_val"] == True:
            _ = filtering_matrix(full_attn_weights_v[j], all_article_identifiers_v[j], sent_lengths_val[j],
                                    window=model_window, with_filtering=filtering, filtering_type=filter_type,
                                    plotting=True)
        _ = filtering_matrix(full_attn_weights_test[k], all_article_identifiers_test[k], sent_lengths_test[k],
                                window=model_window, with_filtering=filtering, filtering_type=filter_type,
                                plotting=True)


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
        "--num_print",
        type=int,
        default=5,
        help="optional number to print",
    )
    arg_parser.add_argument(
        "--random",
        action="store_true",
        help="set this flag to enable random samples for num_print",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file, args.settings_file, num_print=args.num_print, random=args.random)