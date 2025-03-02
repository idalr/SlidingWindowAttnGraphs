import numpy as np
import pandas as pd
import torch
from torchmetrics import F1Score
import matplotlib.pyplot as plt
#from data_loaders import documents_to_ids, DocumentDataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
#from data_loaders import clean_tokenization_sent

def clean_tokenization_sent(list_as_document, object):    
    if object=="text":
        tok_document = list_as_document.split(' [St]')
        return tok_document
    else:
        tok_document = list_as_document.split(", ")
        tok_document[0] = tok_document[0][1:]
        tok_document[-1] = tok_document[-1][:-1]
        return [int(element) for element in tok_document]
    
def retrieve_parameters(model_name, df_logger, with_temperature=False, require_best=True, retrieve_index=None):
    subset_df = df_logger[df_logger['Model']==model_name]
    if require_best:
        retrieve_index = subset_df['Score'].idxmax()
        best_model_ckpt = subset_df.loc[retrieve_index]['Path']
    else:
        best_model_ckpt = subset_df.loc[retrieve_index]['Path']

    if with_temperature:
        return best_model_ckpt, subset_df.loc[retrieve_index]['Score'], subset_df.loc[retrieve_index]['Temperature']
    else:
        return best_model_ckpt, subset_df.loc[retrieve_index]['Score']
        

def eval_results(preds, all_labels, num_classes, partition, print_results=True):
    acc=(torch.Tensor(all_labels) ==preds).float().mean()   
    f1_score = F1Score(task='multiclass', num_classes=num_classes, average=None)
    f1_all = f1_score(preds.int(), torch.Tensor(all_labels).int())
    if print_results:
        print ("\nTrained Model Results on partition:", partition)
        print (partition + " Acc:", acc)
        print (partition + " F1-score macro:", f1_all.mean())
        print (partition + " F1-score for each class:", f1_all)

    return acc, f1_all

def get_window_mask(input_tensor, no_valid_sent, window=30):
    # for one 2-dim tensor
    # if no. valid sentences less than 5, (for at least window_size =2), then full MHA
    if no_valid_sent > 5:
        window_size = torch.ceil(torch.tensor(no_valid_sent)*window/100).clamp(min=2)
        idx = torch.arange(no_valid_sent)
        window_mask = (idx.view(1, -1) - idx.view(-1, 1)).abs() > torch.tensor(window_size).unsqueeze(-1).view(-1, 1)
        window_mask = window_mask.to(device=input_tensor.device)
        return window_mask
    else:
        return torch.zeros_like(input_tensor, dtype=torch.bool)


def get_threshold(input_tensor, degree_std= 1, unbiased=True, type="mean", mode="local"):

    if mode=="global":
        input_tensor = torch.reshape(input_tensor, (-1,))
        max, mean, std = torch.max(input_tensor), torch.mean(input_tensor), torch.std(input_tensor, unbiased)
    elif mode == "local":
        max, mean, std = torch.max(input_tensor, dim=1).values, torch.mean(input_tensor, dim=1), torch.std(input_tensor, dim=1)

    if type=="mean":
        min_value = mean - std*-degree_std  ### negative degree_std for less tolerance at filtering
    elif type=="max":
        min_value = max - std*degree_std
    else:
        #print ("No threshold was calculated.")
        return max, mean, std, None
        
    return max, mean, std, min_value

def get_sliding_window_threshold(input_tensor, window_mask, degree_std= 1, unbiased=True, type="mean", mode="local"):

    if mode == 'global':
        max_val, mean_val, std_val = torch.max(input_tensor[~window_mask]), torch.mean(input_tensor[~window_mask]), torch.std(input_tensor[~window_mask], unbiased=unbiased)
    elif mode == 'local':
        # # break down in local chunks
        # input_chunks = torch.chunk(input_tensor, chunks=input_tensor.shape[0], dim=-1)
        # window_chunks = torch.chunk(window_mask, chunks=input_tensor.shape[0], dim=-1)
        # loop over each chunk and get local values
        local_mean = []
        local_max = []
        local_std = []
        # for i,w in zip(input_chunks, window_chunks):
        #     local_mean.append(i[~w].mean())
        #     local_max.append(i[~w].max())
        #     local_std.append(i[~w].std())
        for i in range(input_tensor.size(0)):
            t = input_tensor[i]
            w = window_mask[i]
            local_mean.append(t[~w].mean())
            local_max.append(t[~w].max())
            local_std.append(t[~w].std())
        # stack up
        mean_val = torch.stack(local_mean)
        max_val = torch.stack(local_max)
        std_val = torch.stack(local_std)

    if type=="mean":
        min_val = mean_val - std_val*-degree_std  ### negative degree_std for less tolerance at filtering
    elif type=="max":
        min_val = max_val - std_val*degree_std
    else:
        #print ("No threshold was calculated.")
        return max_val, mean_val, std_val, None
    return max_val, mean_val, std_val, min_val


def not_filtering_matrices(full_attn_weights, all_article_identifiers, list_valid_sents, df=None, print_samples=0,
                           degree_std=0.5, color='viridis'):
    '''
    experimenting from the previous function `filtering_matrices` but without any filter
    :param full_attn_weights:
    :param all_article_identifiers:
    :param list_valid_sents:
    :param df: either None or a pd.DataFrame (cols = article_id, article_title, article_text, article_label)
    :return:
    '''

    print("\nGenerating Graphs Objects based on Full Attention Weights")

    # set up
    cropped_matrices = []
    total_edges = []
    total_nodes = []
    printed = 0
    index = 0

    for doc_att in full_attn_weights:
        cropped_matrix = doc_att[:list_valid_sents[index], :list_valid_sents[index]]
        cropped_matrices.append(cropped_matrix)

        total_edges.append(list_valid_sents[index] ** 2)
        total_nodes.append(list_valid_sents[index])

        window_mask = get_window_mask(cropped_matrix,list_valid_sents[index])
        max_v, mean, std, _ = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std, type=None)

        if printed < print_samples and len(cropped_matrix) >= 10:
            print("====================================")
            # get doc id
            ###try:
            ide_article = all_article_identifiers[index].item()
            # except: # is this for Summarization?
            #     ide_article = all_article_identifiers[index]

            print("\nDocument ID: ", ide_article, "-- #Sentences: ", list_valid_sents[index])

            # print text
            # try:
            #     print("Source text:\n", df[df['article_id'] == ide_article]['article_text'].values[0])
            # except: # for Summarization
            #     source_text = clean_tokenization_sent(df[df['Article_ID'] == ide_article]['Cleaned_Article'].values[0],
            #                                           "text")
            #     print("Source text:\n", source_text)
            #     if len(source_text)!=list_valid_sents[index]:
            #         print ("WARNING: Number of sentences in source text and attention weights do not match")
            #         print ("Stopping evaluation...")
            #         break

            f, axarr = plt.subplots(1, 2, figsize=(20, 4.5))
            axarr[0].matshow(cropped_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
            axarr[0].set_title('Full Attention Weights')

            # convert to array and only use values in sliding window
            cropped_array = torch.reshape(cropped_matrix, (-1,)).cpu().detach().numpy().flatten()
            window_array = torch.reshape(window_mask, (-1,)).cpu().detach().numpy().flatten()
            valid_array = cropped_array[~window_array]

            n, bins, patches = axarr[1].hist(valid_array, bins=25, color='b', histtype='bar', density=True)
            y = ((1 / (np.sqrt(2 * np.pi) * std.mean().cpu().numpy())) * np.exp(-0.5 * (1 / std.mean().cpu().numpy() * (bins - mean.mean().cpu().numpy())) ** 2))
            axarr[1].plot(bins, y, '--')
            axarr[1].set_title('Attention Weights Distribution')
            axarr[1].axvline(mean.mean().cpu().numpy(), color='g', linestyle='-', label='Mean', lw=3)
            axarr[1].axvline(mean.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='r',
                                 linestyle='--', label='Mean filter', lw=3)
            axarr[1].axvline(max_v.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='c',
                                 linestyle='--', label='Max filter', lw=3)
            axarr[1].legend()
            plt.show()

            printed += 1
        index += 1

    print("Done.")
    return cropped_matrices, total_nodes, total_edges


def and_filtering_matrices(full_attn_weights, all_article_identifiers, list_valid_sents, df=None, print_samples=0,
                       degree_std=0.5, with_filtering=True, filtering_type="mean", granularity="local", color='magma'):

    print("\nFiltering Attention Weights based on Max/Mean and Std Deviation")

    # set up
    filtered_matrices = []
    deletions = []
    cropped_matrices = []
    total_edges = []
    total_nodes = []
    printed = 0
    index = 0
    for doc_att in full_attn_weights:
        # TODO: fix bug cropped_matrix sometimes has too much attention
        cropped_matrix = doc_att[:list_valid_sents[index], :list_valid_sents[index]]
        cropped_matrices.append(cropped_matrix)

        window_mask = get_window_mask(cropped_matrix, list_valid_sents[index])

        # get alternative filtering
        if filtering_type == "mean":
            alternative = "max"
        else:
            alternative = "mean"

        # try:
        max_v, mean, std, threshold_min = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std,
                                                                type=filtering_type, mode=granularity)
        _, _, _, alter_threshold_min = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std,
                                                                    type=alternative, mode=granularity)
        # except: # not sure why it could break...
        #     print("Error in calculating threshold")
        #     print("Stopping evaluation...")
        #     print(cropped_matrix)
        #     print(list_valid_sents[index])
        #     print(doc_att)
        #     break

        if granularity == "local":
            filtered_matrix = torch.Tensor(cropped_matrix.size())
            other_filtered_matrix = torch.Tensor(cropped_matrix.size())
            for i in range(cropped_matrix.size(0)):
                filtered_matrix[i] = torch.where(cropped_matrix[i] < threshold_min[i], 0., cropped_matrix[i])
                other_filtered_matrix[i] = torch.where(cropped_matrix[i] < alter_threshold_min[i], 0., cropped_matrix[i])
            num_dropped = len(cropped_matrix) ** 2 - torch.count_nonzero(filtered_matrix).item()
        else: # "global"
            filtered_matrix = torch.where(cropped_matrix < threshold_min, 0., cropped_matrix.double())  # mean
            num_dropped = torch.count_nonzero(torch.reshape((cropped_matrix < threshold_min), (-1,)).int()).item()
            other_filtered_matrix = torch.where(cropped_matrix < alter_threshold_min, 0., cropped_matrix.double())

        deletions.append(num_dropped)
        total_edges.append(torch.count_nonzero(filtered_matrix).item())
        total_nodes.append(list_valid_sents[index])
        filtered_matrices.append(filtered_matrix)

        # printing
        if printed < print_samples and len(cropped_matrix) >= 10:
            print("====================================")
            # try:
            ide_article = all_article_identifiers[index].item()
            # except:
            #     ide_article = all_article_identifiers[index]

            print("\nDocument ID: ", ide_article, "-- #Sentences: ", list_valid_sents[index])
            # try:
            #     print("Source text:\n", df[df['article_id'] == ide_article]['article_text'].values[0])
            # except:
            #     source_text = clean_tokenization_sent(df[df['Article_ID'] == ide_article]['Cleaned_Article'].values[0],
            #                                           "text")
            #     print("Source text:\n", source_text)
            #     # if len(source_text)!=list_valid_sents[index]:
            #     #     print ("WARNING: Number of sentences in source text and attention weights do not match")
            #     #     print ("Stopping evaluation...")
            #     #     break

            print(f"After thresholding (mean:{mean.mean()}, - std: {std.mean()}")
            # print ("Removing values lower than",threshold_min.item(),"...")
            print(num_dropped, "entries were filtered out")

            # plotting
            f, axarr = plt.subplots(1, 4, figsize=(20, 4.5))
            axarr[0].matshow(cropped_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
            axarr[0].set_title('Full Attention Weights')

            if filtering_type == "mean":
                axarr[1].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
                axarr[2].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(),
                                     cmap=color)  ### just for comparison - no returns
            else:
                axarr[1].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
                axarr[2].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)

            axarr[1].set_title('Mean Filtering')
            axarr[2].set_title('Max Filtering')

            # convert to array and only use values in sliding window
            cropped_array = torch.reshape(cropped_matrix, (-1,)).cpu().detach().numpy().flatten()
            window_array = torch.reshape(window_mask, (-1,)).cpu().detach().numpy().flatten()
            valid_array = cropped_array[~window_array]

            n, bins, patches = axarr[3].hist(valid_array, bins=25, color='b', histtype='bar', density=True)
            y = ((1 / (np.sqrt(2 * np.pi) * std.mean().cpu().numpy())) * np.exp(
                    -0.5 * (1 / std.mean().cpu().numpy() * (bins - mean.mean().cpu().numpy())) ** 2))
            axarr[3].plot(bins, y, '--')
            axarr[3].set_title('Attention Weights Distribution')
            axarr[3].axvline(mean.mean().cpu().numpy(), color='g', linestyle='-', label='Mean', lw=3)
            axarr[3].axvline(mean.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='r',
                                 linestyle='--', label='Mean filter', lw=3)
            axarr[3].axvline(max_v.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='c',
                                 linestyle='--', label='Max filter', lw=3)
            axarr[3].legend()
            plt.show()

            printed += 1
        index += 1

    print("Done.")

    return filtered_matrices, total_nodes, total_edges, deletions


def filtering_matrices(full_attn_weights, all_article_identifiers, list_valid_sents, df=None, print_samples=0,
                       degree_std=0.5, with_filtering=True, filtering_type="mean", granularity="local", color=None):

    filtered_matrices=[]
    cropped_matrices=[]
    total_edges = []
    total_nodes = []

    if with_filtering:
        print ("\nFiltering Attention Weights based on Max/Mean and Std Deviation")
        deletions = []
        if color is None:
            color='magma' 
    else:   
        print ("\nGenerating Graphs Objects based on Full Attention Weights")
        if color is None:
            color='viridis' 
            
    printed=0

    index=0
    for doc_att in full_attn_weights:
        
        cropped_matrix = doc_att[:list_valid_sents[index],:list_valid_sents[index]]
        cropped_matrices.append(cropped_matrix)
             
        if with_filtering:
            try:
                max_v, mean, std, threshold_min = get_threshold(cropped_matrix, degree_std, type=filtering_type, mode=granularity)  
            except: 
                print ("Error in calculating threshold")
                print ("Stopping evaluation...")
                print (cropped_matrix)
                print (list_valid_sents[index])
                print (doc_att)
                break

            if granularity=="local":
                filtered_matrix = torch.Tensor(cropped_matrix.size())
                for i in range(cropped_matrix.size(0)):
                    filtered_matrix[i]= torch.where(cropped_matrix[i] < threshold_min[i], 0., cropped_matrix[i])
                num_dropped = len(cropped_matrix)**2 - torch.count_nonzero(filtered_matrix).item()
            else:
                filtered_matrix = torch.where(cropped_matrix < threshold_min, 0., cropped_matrix.double()) #mean
                num_dropped = torch.count_nonzero(torch.reshape((cropped_matrix < threshold_min), (-1,)).int()).item()

            deletions.append(num_dropped)
            total_edges.append(torch.count_nonzero(filtered_matrix).item())

            total_nodes.append(list_valid_sents[index])  

        else:
             
            total_edges.append(list_valid_sents[index]**2)
            total_nodes.append(list_valid_sents[index])
            
            filtered_matrix = cropped_matrix
            max_v, mean, std, _ = get_threshold(filtered_matrix, degree_std, type=None)  
            
        
        filtered_matrices.append(filtered_matrix)

        if printed<print_samples and len(cropped_matrix)>=10:
            print ("====================================")
            try:
                ide_article= all_article_identifiers[index].item()
            except:
                ide_article= all_article_identifiers[index]

            print ("\nDocument ID: ", ide_article, "-- #Sentences: ", list_valid_sents[index])
            try: 
                print ("Source text:\n", df[df['article_id']==ide_article]['article_text'].values[0])
            except:
                source_text = clean_tokenization_sent(df[df['Article_ID']==ide_article]['Cleaned_Article'].values[0],"text")
                print ("Source text:\n", source_text)
                # if len(source_text)!=list_valid_sents[index]:
                #     print ("WARNING: Number of sentences in source text and attention weights do not match")
                #     print ("Stopping evaluation...")
                #     break

            if with_filtering:
                print (f"After thresholding (mean:{mean.mean()}, - std: {std.mean()}")
                #print ("Removing values lower than",threshold_min.item(),"...")
                print (num_dropped, "entries were filtered out")
                if filtering_type=="mean":
                    alternative = "max"
                else:
                    alternative = "mean"

                _, _, _, other_threshold_min = get_threshold(cropped_matrix, degree_std, type=alternative, mode=granularity)  
                if granularity=="local":
                    other_filtered_matrix = torch.Tensor(cropped_matrix.size())
                    for i in range(cropped_matrix.size(0)):
                        other_filtered_matrix[i]= torch.where(cropped_matrix[i] < other_threshold_min[i], 0., cropped_matrix[i])
                else: #global
                    other_filtered_matrix = torch.where(cropped_matrix < other_threshold_min, 0., cropped_matrix.double())

                f, axarr = plt.subplots(1,4, figsize=(20, 4.5))
                axarr[0].matshow(cropped_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
                axarr[0].set_title('Full Attention Weights')
                if filtering_type=="mean":
                    axarr[1].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)  
                    axarr[2].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)    ### just for comparison - no returns     
                else:
                    axarr[1].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color) 
                    axarr[2].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)  
                          
                axarr[1].set_title('Mean Filtering')
                axarr[2].set_title('Max Filtering')
                    
                n, bins, patches = axarr[3].hist(torch.reshape(cropped_matrix, (-1,)).cpu().detach().numpy().flatten(), bins=25, color='b', histtype='bar', density=True)
                y = ((1 / (np.sqrt(2 * np.pi) * std.mean().cpu().numpy())) * np.exp(-0.5 * (1 / std.mean().cpu().numpy() * (bins - mean.mean().cpu().numpy()))**2))
                axarr[3].plot(bins, y, '--')
                axarr[3].set_title('Attention Weights Distribution')
                axarr[3].axvline(mean.mean().cpu().numpy(), color='g', linestyle='-', label='Mean', lw=3)
                axarr[3].axvline(mean.mean().cpu().numpy()- degree_std* std.mean().cpu().numpy(), color='r', linestyle='--', label='Mean filter', lw=3)
                axarr[3].axvline(max_v.mean().cpu().numpy()- degree_std* std.mean().cpu().numpy(), color='c', linestyle='--', label='Max filter', lw=3)
                axarr[3].legend()
                plt.show()

            else:
                f, axarr = plt.subplots(1,2, figsize=(20, 4.5))
                axarr[0].matshow(cropped_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
                axarr[0].set_title('Full Attention Weights')
                    
                n, bins, patches = axarr[1].hist(torch.reshape(cropped_matrix, (-1,)).cpu().detach().numpy().flatten(), bins=25, color='b', histtype='bar', density=True)
                y = ((1 / (np.sqrt(2 * np.pi) * std.mean().cpu().numpy())) * np.exp(-0.5 * (1 / std.mean().cpu().numpy() * (bins - mean.mean().cpu().numpy()))**2))
                axarr[1].plot(bins, y, '--')
                axarr[1].set_title('Attention Weights Distribution')
                axarr[1].axvline(mean.mean().cpu().numpy(), color='g', linestyle='-', label='Mean', lw=3)
                axarr[1].axvline(mean.mean().cpu().numpy()- degree_std* std.mean().cpu().numpy(), color='r', linestyle='--', label='Mean filter', lw=3)
                axarr[1].axvline(max_v.mean().cpu().numpy()- degree_std* std.mean().cpu().numpy(), color='c', linestyle='--', label='Max filter', lw=3)
                axarr[1].legend()
                plt.show()

            printed+=1
        index+=1
    
    print ("Done.")
    
    if with_filtering:
        return filtered_matrices, total_nodes, total_edges, deletions
    else:
        return filtered_matrices, total_nodes, total_edges

def filtering_matrix(doc_att, valid_sents, degree_std=0.5, with_filtering=True, filtering_type="mean", granularity="local"):
    # TODO: check if need to change because of sliding window
    ## modify this cropped_matrix
    cropped_matrix = doc_att[:valid_sents,:valid_sents]
            
    if with_filtering:
        max_v, mean, std, threshold_min = get_threshold(cropped_matrix, degree_std, type=filtering_type, mode=granularity)  

        if granularity=="local":
            filtered_matrix = torch.Tensor(cropped_matrix.size())
            for i in range(cropped_matrix.size(0)):
                filtered_matrix[i]= torch.where(cropped_matrix[i] < threshold_min[i], 0., cropped_matrix[i])
        else:
            filtered_matrix = torch.where(cropped_matrix < threshold_min, 0., cropped_matrix.double()) #mean

    else:        
        filtered_matrix = cropped_matrix            
    
    return filtered_matrix
