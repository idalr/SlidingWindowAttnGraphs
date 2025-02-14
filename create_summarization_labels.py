import re, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import time 
from tqdm import tqdm
import torch

from nltk.tokenize import word_tokenize, sent_tokenize
from datasets import load_dataset
from functools import partial
import multiprocessing as mp
from rouge_score import rouge_scorer

from utils_extractive import clean_dataframe, create_cleaned_objects_best_increment, calculate_rouge_per_row
from data_loaders import check_dataframe

  
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"]='7'

def load_and_create_extractive(dataset="ccdv/govreport-summarization", partition="train", path_save_cleaned_data="/scratch/datasets/GovReport-Sum/Processed/", with_RL= False):

    data_summ = load_dataset(dataset) ## otros: ccdv/pubmed-summarization, ccdv/arxiv-summarization
    print ("There are", len(data_summ['train']), "samples in the training set.")
    print ("There are", len(data_summ['validation']), "samples in the validation set.")
    print ("There are", len(data_summ['test']), "samples in the test set.")

    df_ = pd.DataFrame(data_summ[partition])
    df_ = clean_dataframe(df_)


    clean_time = time.time()
    df_full = create_cleaned_objects_best_increment(df_, partition, with_RL=with_RL)
    df_full.to_csv(path_save_cleaned_data+"df_"+partition+".csv", index=False)
    print ("["+partition+"-BEST INCREMENT] Time taken for cleaning and saving the labeled data:", time.time()-clean_time)


    df_full = pd.read_csv(path_save_cleaned_data+"df_"+partition+".csv")
    print("\nChecking preprocessed dataframe...")  
    ids2remove = check_dataframe(df_full)
    if len(ids2remove) > 0:
        print ("Removing", len(ids2remove), "samples from the dataframe...")
        for id_remove in ids2remove:
            df_full = df_full.drop(id_remove)    
        df_full.reset_index(drop=True, inplace=True)
    else:
        print ("No samples to remove from the dataframe.")
    print ("Dataframe shape:", df_full.shape)

    results_= calculate_rouge_per_row(df_full, partition, path_save_cleaned_data+"Oracle_Processed/", type_rouge="R1R2")
    #results_ = pd.read_csv(path_save_cleaned_data+"Oracle_Processed/df_test_rouge_R1R2_best_increment.csv")
    #print (results_.shape)

    print ("["+partition+"] ORACLE:\n", "Rouge1_F1:", results_["Rouge-1F1"].mean(), "\tRouge2_F1:",results_["Rouge-2F1"].mean(), "\tRougeL_F1:", results_["Rouge-LF1"].mean())

    return 


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python create_summarization_labels.py <dataset> <partition> <data_path> <bool-include_RL>")
    else:
        dataset = sys.argv[1]
        partition = sys.argv[2]
        data_path = sys.argv[3]
        include_RL = sys.argv[4]
        load_and_create_extractive(dataset, partition, data_path, with_RL=include_RL)
        print(f"Extractive Labels and ORACLE results computed for {dataset}, {partition} partition.")