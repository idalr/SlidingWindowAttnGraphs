import re, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from datasets import load_dataset

warnings.filterwarnings("ignore")

import time
from tqdm import tqdm
import torch
import json

from nltk.tokenize import word_tokenize, sent_tokenize
from functools import partial
import multiprocessing as mp
from rouge_score import rouge_scorer

from utils_extractive import clean_dataframe, create_cleaned_objects_best_increment, calculate_rouge_per_row, \
    clean_content_from_file
from data_loaders import check_dataframe

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# arxiv: path_dataset="/scratch/datasets/arXiv-Sum/arxiv-dataset/", partition="train", path_save_cleaned_data="/scratch/datasets/arXiv-Sum/Processed/", with_RL= False, from_hf=True):

def load_and_create_extractive(path_dataset="ccdv/govreport-summarization", partition="train",
                               path_save_cleaned_data="/scratch/datasets/GovReport-Sum/Processed/", with_RL=False,
                               from_hf=True):
    if not os.path.isfile(path_dataset + "df_" + partition + "_aux.csv"):
        if from_hf == True:
            print("Loading dataset from Hugging Face...")
            flag_for_text_cleaning = True
            data_summ = load_dataset(path_dataset)  ## otros: ccdv/pubmed-summarization, ccdv/arxiv-summarization
            print("There are", len(data_summ['train']), "samples in the training set.")
            print("There are", len(data_summ['validation']), "samples in the validation set.")
            print("There are", len(data_summ['test']), "samples in the test set.")

            df_ = pd.DataFrame(data_summ[partition])
        else:
            ### create dataframe for summarization datasets loaded differently -- no HF Datasets
            print("Loading dataset from local files...")
            flag_for_text_cleaning = False
            with open(path_dataset + "train" + ".txt", 'r') as file:
                print("There are", len(file.readlines()), "train samples in the dataset")
            with open(path_dataset + "val" + ".txt", 'r') as file:
                print("There are", len(file.readlines()), "validation samples in the dataset")
            with open(path_dataset + "test" + ".txt", 'r') as file:
                print("There are", len(file.readlines()), "test samples in the dataset")

            empty_docs = []
            long_summaries = []
            empty_summaries = []
            with open(path_dataset + partition + ".txt", 'r') as file:
                total_lines = len(file.readlines())

            df_ = pd.DataFrame(columns=["article_id", "report", "summary"])

            with open(path_dataset + partition + ".txt", 'r') as file:
                for line in tqdm(file, total=total_lines):
                    content = json.loads(line)
                    article_id = content['article_id']
                    source_text = content['article_text']
                    abstract_text = content['abstract_text']
                    if len(source_text) == 1 and source_text[0] == "":
                        # print ("Empty doc in:", article_id)
                        empty_docs.append(article_id)
                        continue
                    elif (len(abstract_text) == 1 and abstract_text[0] == "") or (len(abstract_text) == 0):
                        # print ("Empty summary in:", article_id)
                        empty_summaries.append(article_id)
                        continue
                    elif len(abstract_text) > len(source_text):
                        # print ("Summary longer than source text in:", article_id)
                        long_summaries.append(article_id)
                        continue
                    else:
                        sent_in_doc, abstract_as_text = clean_content_from_file(source_text, abstract_text)
                        sent_in_doc_as_text = ' [St]'.join(sent_in_doc)
                        df_.loc[len(df_)] = {"article_id": article_id, "report": sent_in_doc_as_text,
                                             "summary": abstract_as_text}

            print("#Empty docs:", len(empty_docs), "\t#Empty summaries:", len(empty_summaries),
                  "\t#Summaries longer than source:", len(long_summaries))

        df_ = clean_dataframe(df_)  ### remove empty and duplicated samples
        df_.to_csv(path_dataset + "df_" + partition + "_aux.csv", index=False)

    else:
        print("Loading previously processed dataframe...")
        df_ = pd.read_csv(path_dataset + "df_" + partition + "_aux.csv")
        df_ = clean_dataframe(df_)
        print("\nDataframe previously processed -- #samples:", df_.shape)
        flag_for_text_cleaning = True if from_hf == True else False
        print("Flag for text cleaning:", flag_for_text_cleaning)

    clean_time = time.time()
    df_full = create_cleaned_objects_best_increment(df_, partition, with_RL=with_RL,
                                                    flag_cleaning=flag_for_text_cleaning)
    df_full.to_csv(path_save_cleaned_data + "df_" + partition + ".csv", index=False)
    print("[" + partition + "-BEST INCREMENT] Time taken for cleaning and saving the labeled data:",
          time.time() - clean_time)

    df_full = pd.read_csv(path_save_cleaned_data + "df_" + partition + ".csv")
    print("\nChecking preprocessed dataframe...")
    ids2remove = check_dataframe(df_full)  ### remove rows where #labels do not match #sentences
    if len(ids2remove) > 0:
        print("Removing", len(ids2remove), "samples from the dataframe...")
        for id_remove in ids2remove:
            df_full = df_full.drop(id_remove)
        df_full.reset_index(drop=True, inplace=True)
    else:
        print("No samples to remove from the dataframe.")
    print("Dataframe shape:", df_full.shape)

    results_ = calculate_rouge_per_row(df_full, partition, path_save_cleaned_data + "Oracle_Processed/",
                                       type_rouge="R1R2")
    # results_ = pd.read_csv(path_save_cleaned_data+"Oracle_Processed/df_test_rouge_R1R2_best_increment.csv")
    # print (results_.shape)

    print("[" + partition + "] ORACLE:\n", "Rouge1_F1:", results_["Rouge-1F1"].mean(), "\tRouge2_F1:",
          results_["Rouge-2F1"].mean(), "\tRougeL_F1:", results_["Rouge-LF1"].mean())

    return


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print(
            "Usage: python create_summarization_labels.py <dataset> <partition> <data_path> <bool-include_RL> <bool-read_from_HF>")
    else:
        dataset_path = sys.argv[1]
        partition = sys.argv[2]
        save_data_path = sys.argv[3]
        include_RL = sys.argv[4]
        flag_HF = sys.argv[5]
        load_and_create_extractive(dataset_path, partition, save_data_path, with_RL=include_RL, from_hf=flag_HF)
        print(f"Extractive Labels and ORACLE results computed for {dataset_path}, {partition} partition.")