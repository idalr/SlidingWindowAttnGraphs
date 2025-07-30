import random
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data.graph_utils import get_window_mask, get_threshold, get_sliding_window_threshold
from src.data.utils import clean_tokenization_sent


def retrieve_parameters(model_name, df_logger, with_temperature=False, require_best=True, retrieve_index=None):
    # Retrieve the best model checkpoint and its score based on the model name from logger
    subset_df = df_logger[df_logger['Model' ]==model_name]
    if require_best:
        retrieve_index = subset_df['Score'].idxmax()
        best_model_ckpt = subset_df.loc[retrieve_index]['Path']
    else:
        best_model_ckpt = subset_df.loc[retrieve_index]['Path']

    if with_temperature:
        return best_model_ckpt, subset_df.loc[retrieve_index]['Score'], subset_df.loc[retrieve_index]['Temperature']
    else:
        return best_model_ckpt, subset_df.loc[retrieve_index]['Score']


def get_sample_indices(length, num_print, random_sampling=True):
    if random_sampling:
        return random.sample(range(length), num_print)
    else:
        return list(range(num_print))


def filtering_matrices(full_attn_weights, all_article_identifiers, list_valid_sents, window, df=None,
                       print_samples=0, degree_std=0.5, with_filtering=True, filtering_type="mean",
                       granularity="local", color=None, selected_indices=None):
    # filtering multiple matrices for plotting visualization
    # set up
    cropped_matrices = []
    total_edges = []
    total_nodes = []
    printed = 0
    index = 0

    if with_filtering:
        # additional set up
        deletions = []
        filtered_matrices = []

        # get alternative filtering
        if filtering_type == "mean":
            alternative = "max"
        elif filtering_type == 'max':
            alternative = "mean"
        print("\nFiltering Attention Weights based on Max/Mean and Std Deviation")

    else:  # no filtering
        print("\nGenerating Graphs Objects based on Full Attention Weights")

    # get valid cropped matrices
    indices = selected_indices if selected_indices is not None else range(len(full_attn_weights))
    for index in indices:
        doc_att = full_attn_weights[index]
        cropped_matrix = doc_att[:list_valid_sents[index], :list_valid_sents[index]]
        cropped_matrices.append(cropped_matrix)
        window_mask = get_window_mask(cropped_matrix, list_valid_sents[index], window)

        if window == 100:
            max_v, mean, std, threshold_min = get_threshold(cropped_matrix, degree_std, type=filtering_type)
        else:
            max_v, mean, std, threshold_min = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std,
                                                                           type=filtering_type)

        if with_filtering:
            _, _, _, other_threshold_min = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std,
                                                                        type=alternative, mode=granularity)
            if granularity == "local":
                filtered_matrix = torch.Tensor(cropped_matrix.size())
                other_filtered_matrix = torch.Tensor(cropped_matrix.size())
                for i in range(cropped_matrix.size(0)):
                    filtered_matrix[i] = torch.where(cropped_matrix[i] < threshold_min[i], 0., cropped_matrix[i])
                    other_filtered_matrix[i] = torch.where(cropped_matrix[i] < other_threshold_min[i], 0.,
                                                           cropped_matrix[i])
                num_dropped = len(cropped_matrix) ** 2 - torch.count_nonzero(filtered_matrix).item()
            else:  # "global"
                filtered_matrix = torch.where(cropped_matrix < threshold_min, 0., cropped_matrix.double())
                other_filtered_matrix = torch.where(cropped_matrix < other_threshold_min, 0., cropped_matrix.double())
                num_dropped = torch.count_nonzero(torch.reshape((cropped_matrix < threshold_min), (-1,)).int()).item()

            deletions.append(num_dropped)
            total_edges.append(torch.count_nonzero(filtered_matrix).item())
            total_nodes.append(list_valid_sents[index])
            filtered_matrices.append(filtered_matrix)

        else:  # no filtering
            total_edges.append(list_valid_sents[index] ** 2)
            total_nodes.append(list_valid_sents[index])

        if print_samples > 0:
            if printed < print_samples and len(cropped_matrix) >= 10:
                # a printing function
                print_filtering_matrix(df, all_article_identifiers[index], list_valid_sents[index])
                # a plotting function
                if with_filtering:
                    print(f"After thresholding (mean:{mean.mean()}, - std: {std.mean()}")
                    # print ("Removing values lower than",threshold_min.item(),"...")
                    print(num_dropped, "entries were filtered out")

                    # a plotting function
                    if color is None:
                        color = 'magma'
                    plot_filtering_matrix(cropped_matrix, mean, max_v, std, degree_std, color,
                                          filtering_type, filtered_matrix, other_filtered_matrix)
                else:
                    if color is None:
                        color = 'viridis'
                    plot_filtering_matrix(cropped_matrix, mean, max_v, std, degree_std, color)

                printed += 1

        index += 1

    if with_filtering:
        return filtered_matrices, total_nodes, total_edges, deletions
    else:
        return cropped_matrices, total_nodes, total_edges


def print_filtering_matrix(df, article_identifier, no_valid_sents):
    print("====================================")
    # get doc id
    try:
        print("\nDocument ID: ", article_identifier.item(), "-- #Sentences: ", no_valid_sents)
    except:  # for Summarization
        print("\nDocument ID: ", article_identifier, "-- #Sentences: ", no_valid_sents)

    # print text
    try:
        print("Source text:\n", df[df['article_id'] == article_identifier.item()]['article_text'].values[0])
    except:  # for Summarization
        source_text = clean_tokenization_sent(
            df[df['Article_ID'] == article_identifier.item()]['Cleaned_Article'].values[0], "text")
        print("Source text:\n", source_text)
        if len(source_text) != no_valid_sents:
            print(
                "WARNING: Number of sentences in source text and attention weights do not match")
            print("Stopping evaluation...")


def plot_filtering_matrix(cropped_matrix,
                         mean, max_v, std, degree_std, color, filtering_type=False,
                         filtered_matrix=None, other_filtered_matrix=None):

    f, axarr = plt.subplots(1, 2, figsize=(10, 4.5))

    # convert to array and only use values in sliding window
    cropped_array = torch.reshape(cropped_matrix, (-1,)).cpu().detach().numpy().flatten()
    n, bins, patches = axarr[1].hist(cropped_array, bins=25, color='b', histtype='bar', density=True)
    y = ((1 / (np.sqrt(2 * np.pi) * std.mean().cpu().numpy())) * np.exp(
        -0.5 * (1 / std.mean().cpu().numpy() * (bins - mean.mean().cpu().numpy())) ** 2))
    axarr[0].plot(bins, y, '--')
    axarr[0].set_title('Attention Weights Distribution')
    axarr[0].axvline(mean.mean().cpu().numpy(), color='g', linestyle='-', label='Mean', lw=3)
    axarr[0].axvline(mean.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='r',
                     linestyle='--', label='Mean filter', lw=3)
    axarr[0].axvline(max_v.mean().cpu().numpy() - degree_std * std.mean().cpu().numpy(), color='c',
                     linestyle='--', label='Max filter', lw=3)
    axarr[0].legend()

    axarr[1].matshow(cropped_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
    axarr[1].set_title('Full Attention Weights')
    plt.show()

    if filtering_type:
        f, axarr = plt.subplots(1, 2, figsize=(10, 4.5))
        if filtering_type == "mean":
            axarr[0].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
            axarr[1].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(),
                             cmap=color)  ### just for comparison - no returns
        else:
            axarr[0].matshow(other_filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)
            axarr[1].matshow(filtered_matrix.cpu().detach().numpy(), vmin=0, vmax=max_v.max(), cmap=color)

        axarr[0].set_title('Mean Filtering')
        axarr[1].set_title('Max Filtering')
        plt.show()