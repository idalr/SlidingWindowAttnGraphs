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
from functools import partial
import multiprocessing as mp
from rouge_score import rouge_scorer
from data_loaders import clean_tokenization_sent


def check_conditions_paper(source, summary):
    if len(word_tokenize(summary)) > 1000:
        return True
    if len(word_tokenize(source)) < 30:
        return True
    if len(sent_tokenize(source)) < 3:
        return True
    return False


def clean_dataframe(in_dataframe, top_k_sum=False, column_source="report", column_summary="summary"):
    ### remove duplicates
    print("There are", in_dataframe.shape[0], "samples in the original dataframe.")
    in_dataframe_duplicated = in_dataframe[in_dataframe.duplicated(subset=[column_source])]
    print("There are", in_dataframe_duplicated.shape[0], "duplicated documents in the partition.")
    print("Removing duplicates...")
    in_dataframe = in_dataframe.drop_duplicates(subset=[column_source])
    # df_train.reset_index(drop=True,inplace=True)  ### only for creating the Pt objects -- keeping original index

    ## remove empty documents and samples with summary longer than source text
    if top_k_sum:
        drop_sample = [True if (
                    (len(source) - len(summary) < 0) or len(source) == 0 or len(summary) == 0 or check_conditions_paper(
                source, summary)) else False for source, summary in
                       zip(in_dataframe[column_source], in_dataframe[column_summary])]
    else:
        drop_sample = [True if ((len(source) - len(summary) < 0) or len(source) == 0 or len(summary) == 0) else False
                       for source, summary in zip(in_dataframe[column_source], in_dataframe[column_summary])]
    print("#Samples with empty source, empty summary or summary longer than source: ", drop_sample.count(True))
    print("Removing empty documents...")
    print("Removing documents where summary is longer than source text...")
    in_dataframe = in_dataframe[[not i for i in drop_sample]]
    print("There are", in_dataframe.shape[0], "samples in the final dataframe.")
    return in_dataframe


### cleaning documents
def clean_source_document(document):
    # print (text,"\n")
    text = sent_tokenize(document)
    cleaned_text = []
    for sentence in text:
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        cleaned_text.append(sentence)
    return cleaned_text  # cleaned sentence list


"""
def clean_tokenization_sent(list_as_document, object):    
    if object=="text":
        tok_document = list_as_document.split(' [St]')
        return tok_document
    else:
        tok_document = list_as_document.split(", ")
        tok_document[0] = tok_document[0][1:]
        tok_document[-1] = tok_document[-1][:-1]
        return [int(element) for element in tok_document]
"""


def merge_short(document_as_sentences):
    merged_sentences = []
    holded = ""
    on_hold = False
    for sentence in document_as_sentences:
        if len(sentence) == 1:
            continue

        if (sentence not in ["» .", "'.'", '"...', '\\ .', 'ET).', '!!"', '* .']) and len(
                sentence) > 15 and not on_hold:
            merged_sentences.append(sentence)
        else:
            try:
                if not on_hold:
                    wts = word_tokenize(sentence)
                    if len(wts) == 2 and wts[0].isnumeric():
                        on_hold = True
                        holded = sentence
                        continue
                    else:
                        # print ("Merging sentence <", sentence, "> with previous sentence...")
                        # print ("Previous sentence:", merged_sentences[-1])
                        # print ("document_as_sentences:", document_as_sentences)
                        if len(merged_sentences) > 0:
                            merged_sentences[-1] += " " + sentence
                        if len(merged_sentences) == 0:
                            on_hold = True
                            holded = sentence
                else:
                    merged_sentences.append(holded + " " + sentence)
                    on_hold = False
            except:
                print("\nError merging sentence:", sentence, "-- On-Hold:", on_hold, "-- len merged:",
                      len(merged_sentences))
                print("Stopping the process...")
                break
    return merged_sentences


def average_rouge(summary, sentence, R_scorer):
    result = 0.0
    for type in R_scorer.rouge_types:
        result += R_scorer.score(summary, sentence)[type].fmeasure

    return result / len(R_scorer.rouge_types)


def clean_and_save_data(in_args, R_scorer):  ## clean specific document

    orig_index, source, summary = in_args
    cleaned_source = clean_source_document(source)
    merged_source = merge_short(cleaned_source)

    # compute Rouge1 F1score for all sentences in a document
    all_fscore12 = [average_rouge(summary, sentence, R_scorer) for sentence in merged_source]
    # dictionary of lists
    dict_calculated_scores = {'Sentence_id': range(len(merged_source)), 'Rouge-1/2': all_fscore12,
                              'Sentence': merged_source}
    df_calculated_scores = pd.DataFrame(dict_calculated_scores)
    df_calculated_scores.sort_values(by='Rouge-1/2', ascending=False, inplace=True)  # max to min

    max_score = 0.0
    num_WordsInSummary = len(word_tokenize(summary))
    num_WordsInSelected = 0
    selected = []

    for sentence_id, text, rouge in zip(df_calculated_scores.Sentence_id, df_calculated_scores.Sentence,
                                        df_calculated_scores["Rouge-1/2"]):
        if len(selected) == 0:
            selected.append(sentence_id)
            max_score = rouge
            num_WordsInSelected += len(word_tokenize(text))
        else:
            if num_WordsInSelected >= num_WordsInSummary:
                break
            else:
                cumulative = " ".join(pd.Series(merged_source)[selected]) + " " + text
                current_score = average_rouge(summary, cumulative,
                                              R_scorer)  # R_scorer.score(summary, cumulative)['rouge1'].fmeasure
                if current_score > max_score:
                    if num_WordsInSelected + len(word_tokenize(text)) <= num_WordsInSummary:
                        selected.append(sentence_id)
                        max_score = current_score
                        num_WordsInSelected += len(word_tokenize(text))
                    else:
                        break

    sentences_as_text = ' [St]'.join(merged_source)
    calculated_labels = [1 if i in selected else 0 for i in range(len(merged_source))]

    if len(calculated_labels) > len(merged_source):
        print("Error: HAY MAS LABELS QUE SENTENCES.")

    if len(calculated_labels) < len(merged_source):
        print("Error: HAY MENOS LABELS QUE SENTENCES.")

    return {
        "Article_ID": orig_index,  # doc_id
        "Cleaned_Article": sentences_as_text,  # merged_source,
        "Summary": summary,
        "Calculated_Labels": calculated_labels
    }


def create_cleaned_objects(df_input, partition):
    print("Processing " + partition + " data...")

    original_indexes = list(df_input.index)
    R_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])  # , use_stemmer=True

    with mp.get_context("fork").Pool(processes=len(os.sched_getaffinity(0))) as pool:
        res = pool.imap(partial(clean_and_save_data, **{"R_scorer": R_scorer}),
                        list(zip(original_indexes, df_input.report, df_input.summary)))

        full_df = {"Article_ID": [], "Cleaned_Article": [], "Summary": [], "Calculated_Labels": []}
        for i in tqdm(res, total=len(df_input)):
            full_df["Article_ID"].append(i["Article_ID"])
            full_df["Cleaned_Article"].append(i["Cleaned_Article"])
            full_df["Summary"].append(i["Summary"])
            full_df["Calculated_Labels"].append(i["Calculated_Labels"])

        return pd.DataFrame(full_df)


def clean_and_save_data_best_increment(in_args, R_scorer, flag_cleaning=True, max_len=1000):  ## clean specific document

    orig_index, source, summary = in_args

    cleaned_source = clean_source_document(source) if flag_cleaning == True else clean_tokenization_sent(source, "text")
    merged_source = merge_short(cleaned_source) if flag_cleaning == True else cleaned_source[:max_len]
    # print ("Processing input data for labeling...")
    # print ("#Documents:", len(merged_source))
    # print ("#Summaries:", len(summary))

    cumulative_sentences = ""
    num_WordsInSelected = 0
    selected = []
    highest_score = 0.0  # avg R1, R2
    num_WordsInSummary = len(word_tokenize(summary))  # limit
    s_id = -1

    while num_WordsInSelected < num_WordsInSummary:
        for ide in range(len(merged_source)):
            if ide not in selected:
                ide_score = average_rouge(summary, cumulative_sentences + " " + merged_source[ide], R_scorer)
                if ide_score > highest_score:
                    highest_score = ide_score
                    s_id = ide
        if s_id != -1:
            cumulative_sentences += " " + merged_source[s_id]
            num_WordsInSelected += len(word_tokenize(merged_source[s_id]))
            selected.append(s_id)
        else:
            break

            # print ("Selected sentences:", len(selected))
    sentences_as_text = ' [St]'.join(merged_source)
    calculated_labels = [1 if i in selected else 0 for i in range(len(merged_source))]

    if len(calculated_labels) > len(merged_source):
        print("Error: There are more calculated labels than available sentences.")

    if len(calculated_labels) < len(merged_source):
        print("Error: There are less calculated labels than available sentences.")

    return {
        "Article_ID": orig_index,  # doc_id
        "Cleaned_Article": sentences_as_text,  # merged_source,
        "Summary": summary,
        "Calculated_Labels": calculated_labels
    }


def create_cleaned_objects_best_increment(df_input, partition, with_RL=False, source_column="report",
                                          summary_column="summary", id_column="article_id", flag_cleaning=True):
    print("Processing " + partition + " data...")

    try:
        original_indexes = list(df_input[id_column])
    except:
        original_indexes = list(df_input.index)

    if with_RL == True:
        print("Using Average of Rouge-1/-2/-L")
        R_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    else:
        print("Using Average of Rouge-1/-2")
        R_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])  # , use_stemmer=True

    print("Flag for data cleaning:", flag_cleaning)
    with mp.get_context("fork").Pool(processes=len(os.sched_getaffinity(0))) as pool:
        res = pool.imap(
            partial(clean_and_save_data_best_increment, **{"R_scorer": R_scorer, "flag_cleaning": flag_cleaning}),
            list(zip(original_indexes, df_input[source_column], df_input[summary_column])))

        full_df = {"Article_ID": [], "Cleaned_Article": [], "Summary": [], "Calculated_Labels": []}
        for i in tqdm(res, total=len(df_input)):
            full_df["Article_ID"].append(i["Article_ID"])
            full_df["Cleaned_Article"].append(i["Cleaned_Article"])
            full_df["Summary"].append(i["Summary"])
            full_df["Calculated_Labels"].append(i["Calculated_Labels"])

        return pd.DataFrame(full_df)


def calculate_rouge_per_row(df_, partition, path_save_cleaned_data, type_rouge="R1R2"):
    if not os.path.exists(path_save_cleaned_data + "df_" + partition + "_rouge_" + type_rouge + ".csv"):
        df_rouge = pd.DataFrame(columns=["Article_ID", "Rouge-1P", "Rouge-1R", "Rouge-1F1",
                                         "Rouge-2P", "Rouge-2R", "Rouge-2F1", "Rouge-LP", "Rouge-LR", "Rouge-LF1"])
        df_rouge.to_csv(path_save_cleaned_data + "df_" + partition + "_rouge_" + type_rouge + ".csv", index=False)
    else:  # if exist
        df_rouge = pd.read_csv(path_save_cleaned_data + "df_" + partition + "_rouge_" + type_rouge + ".csv")

    R_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for i in range(len(df_)):  ### remove limit
        article_id = df_.Article_ID[i]
        summary_i = df_.Summary[i]
        sentences_i = np.asarray(clean_tokenization_sent(df_.Cleaned_Article[i], "text"))
        labels_i = np.asarray(clean_tokenization_sent(df_.Calculated_Labels[i], "label"), dtype=bool)
        labelled_i = sentences_i[labels_i]
        labelled_text_i = " ".join(labelled_i)
        rouge_results = R_scorer.score(summary_i, labelled_text_i)

        df_rouge.loc[len(df_rouge)] = {
            "Article_ID": article_id,
            "Rouge-1P": rouge_results['rouge1'].precision,
            "Rouge-1R": rouge_results['rouge1'].recall,
            "Rouge-1F1": rouge_results['rouge1'].fmeasure,
            "Rouge-2P": rouge_results['rouge2'].precision,
            "Rouge-2R": rouge_results['rouge2'].recall,
            "Rouge-2F1": rouge_results['rouge2'].fmeasure,
            "Rouge-LP": rouge_results['rougeL'].precision,
            "Rouge-LR": rouge_results['rougeL'].recall,
            "Rouge-LF1": rouge_results['rougeL'].fmeasure
        }

    df_rouge.to_csv(path_save_cleaned_data + "df_" + partition + "_rouge_" + type_rouge + ".csv", index=False)
    print("Finished and saved in:", path_save_cleaned_data + "df_" + partition + "_rouge_" + type_rouge + ".csv")

    return df_rouge


import ast
import json


def clean_content_from_file(source_text, abstract_text):
    sentences_in_doc = []
    final_sentences = []
    abstract_as_text = ""
    # print ("[Based on /n] #Sentences in article:", len(source_text))
    temp_to_join = ""
    for pre_sentence in source_text:
        if pre_sentence[-1] != ".":  ### cumulate till find puntuation mark
            temp_to_join += " " + pre_sentence

        else:  ### sentence ends in punctuation mark
            if temp_to_join != "":
                temp_to_join += " " + pre_sentence  ### merging
                sentences_in_doc.append(temp_to_join.strip())
                temp_to_join = ""
            else:
                sentences_in_doc.append(pre_sentence.strip())

    if temp_to_join != "":
        sentences_in_doc.append(temp_to_join.strip())
        temp_to_join = ""

    ### Cleaning sentences in list sentences_in_doc
    for pre_sentence in sentences_in_doc:
        pre_sentence_list = pre_sentence.split(" . ")
        temp_to_join = pre_sentence_list[0].strip()  ### first sub-sentence - base case
        if len(pre_sentence_list) != 1:
            for sent in pre_sentence_list[1:]:
                try:
                    if sent[0].isdigit() or len(word_tokenize(sent)) < 5:
                        temp_to_join += " . " + sent.strip()
                    else:
                        final_sentences.append((temp_to_join + " .").strip())
                        temp_to_join = sent.strip()
                except:
                    pass

        if temp_to_join != "":
            final_sentences.append(temp_to_join.strip())

    ### removing  <S> and </S> marks
    for pre_sentence in abstract_text:
        pre_sentence = pre_sentence.replace("<S> ", "")
        pre_sentence = pre_sentence.replace(" </S>", "")
        pre_sentence = pre_sentence.strip()
        abstract_as_text += " " + pre_sentence

    # print ("[After cleaning sentences] #Sentences in article:", len(final_sentences))

    return final_sentences, abstract_as_text