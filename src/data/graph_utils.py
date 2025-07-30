import torch


def solve_by_creating_edge(a, b, orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs,
                           processed_ids, dict_orig_to_ide_graph, new_weight):
    flag_inserted = False
    if a != b:
        if (a, b) not in all_edges:
            orig_target_list.append(b)
            orig_source_list.append(a)
            all_edges.append((a, b))
            flag_inserted = True

            if a in processed_ids:
                source_list.append(processed_ids.index(a))  # requires int from 0 to len
            else:
                source_list.append(dict_orig_to_ide_graph[a])  # requires int from 0 to len
            if b in processed_ids:
                target_list.append(processed_ids.index(b))  # requires int from 0 to len
            else:
                target_list.append(dict_orig_to_ide_graph[b])  # requires int from 0 to len
            if new_weight != None:
                edge_attrs.append(new_weight)
        else:
            if new_weight != None:  # check if the edge already exists and update the weight
                pos_edge = all_edges.index((a, b))
                if edge_attrs[pos_edge] < new_weight:
                    edge_attrs[pos_edge] = new_weight

    return orig_source_list, orig_target_list, all_edges, source_list, target_list, edge_attrs, flag_inserted


def get_threshold(input_tensor, degree_std=1, unbiased=True, type="mean", mode="local"):
    if mode == "global":
        input_tensor = torch.reshape(input_tensor, (-1,))
        max, mean, std = torch.max(input_tensor), torch.mean(input_tensor), torch.std(input_tensor, unbiased)
    elif mode == "local":
        max, mean, std = torch.max(input_tensor, dim=1).values, torch.mean(input_tensor, dim=1), torch.std(input_tensor,
                                                                                                           dim=1)

    if type == "mean":
        min_value = mean - std * -degree_std  # Negative degree_std for less tolerance at filtering

    elif type == "max":
        min_value = max - std * degree_std

    else:
        return max, mean, std, None

    return max, mean, std, min_value


def get_window_mask(input_tensor, no_valid_sent, window):
    # for one 2-dim tensor
    # if no. valid sentences less than 5, (for at least window_size =2), then full MHA
    if no_valid_sent > 5:  # cannot use pad value
        window_size = torch.ceil(torch.tensor(no_valid_sent * window / 100))
        idx = torch.arange(int(no_valid_sent))
        window_mask = (idx.view(1, -1) - idx.view(-1, 1)).abs() > window_size.view(-1, 1)
        if no_valid_sent > input_tensor.shape[0]:
            window_mask = window_mask[: input_tensor.shape[0], :input_tensor.shape[1]]
        window_mask = window_mask.to(device=input_tensor.device)
        return window_mask
    else:
        return torch.zeros_like(input_tensor, dtype=torch.bool)  # no mask when window is larger than the cropped_matrix


def get_sliding_window_threshold(input_tensor, window_mask, degree_std=1, unbiased=True, type="mean", mode="local"):
    # get threshold
    if mode == 'global':
        max_val, mean_val, std_val = torch.max(input_tensor[~window_mask]), torch.mean(
            input_tensor[~window_mask]), torch.std(input_tensor[~window_mask], unbiased=unbiased)
    elif mode == 'local':
        local_mean = []
        local_max = []
        local_std = []
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

    if type == "mean":
        min_val = mean_val - std_val * -degree_std
    elif type == "max":
        min_val = max_val - std_val * degree_std
    else:
        return max_val, mean_val, std_val, None
    return max_val, mean_val, std_val, min_val


def filtering_matrix(doc_att, valid_sents, window, degree_std=0.5, with_filtering=True, filtering_type="mean",
                     granularity="local"):
    # This function filters a single attention matrix based on a threshold calculated from the mean and standard deviation of the attention weights.
    cropped_matrix = doc_att[:valid_sents, :valid_sents]

    if filtering_type == "mean" or filtering_type == "max":
        if window == 100:
            max_v, mean, std, threshold_min = get_threshold(cropped_matrix, degree_std, type=filtering_type,
                                                            mode=granularity)
        else:  # if SWA and if filter applied
            window_mask = get_window_mask(cropped_matrix, valid_sents, window)
            max_v, mean, std, threshold_min = get_sliding_window_threshold(cropped_matrix, window_mask, degree_std,
                                                                           type=filtering_type, mode=granularity)

        if granularity == "local":
            filtered_matrix = torch.Tensor(cropped_matrix.size())
            for i in range(cropped_matrix.size(0)):
                filtered_matrix[i] = torch.where(cropped_matrix[i] < threshold_min[i], 0., cropped_matrix[i])
        else:
            filtered_matrix = torch.where(cropped_matrix < threshold_min, 0., cropped_matrix.double())  # mean

        return filtered_matrix

    else:
        return cropped_matrix
