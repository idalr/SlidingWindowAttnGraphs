import torch
import numpy as np
from torch import nn

def retrieve_from_dict(dict, list_ids):
    return [dict[id.item()] for id in list_ids]

def retrieve_from_dict_safe(dict_obj, list_ids, default="[UNK]"):
    """
    Map list of IDs to sentences safely.
    Missing IDs replaced by `default`.
    """
    result = []
    missing_count = 0
    for id in list_ids:
        key = id.item() if isinstance(id, torch.Tensor) else id
        if key in dict_obj:
            result.append(dict_obj[key])
        else:
            result.append(default)
            missing_count += 1
    if missing_count > 0:
        print(f"[WARNING] {missing_count}/{len(list_ids)} IDs not found in vocab. Replaced with default.")
    return result

def scaled_dot_product(q, k, v, mask=None, temperature=1, dropout=0.0, training=True, attention="softmax"):
    factor = 1 / np.sqrt(q.size(-1))
    attn_logits = q @ k.transpose(-2, -1) * factor

    if mask is not None:
        attn_logits += mask

    if attention == "softmax":
        attention = nn.functional.softmax(attn_logits / temperature, dim=-1)
    elif attention == "sigmoid":  # based on https://arxiv.org/pdf/2409.04431
        attention = nn.functional.sigmoid(attn_logits - torch.log(attn_logits.shape[-1]))
    elif attention == "relu":  # based on https://arxiv.org/pdf/2309.08586
        attention = nn.functional.relu(attn_logits) / attn_logits.shape[-1]

    attention = torch.dropout(attention, dropout, train=training)
    values = attention @ v

    return values, attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, temperature=1, dropout=0.0, activation_attention="softmax"):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 module wrt the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        self.dropout = dropout
        self.activation_attention = activation_attention
        self.qkv_proj = torch.nn.Linear(input_dim, 3 * embed_dim, bias=True)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self._init_params()

    def _init_params(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, src_key_padding_mask, matrix_mask, temperature=None):
        matrix_mask_extended = matrix_mask[:, None, :, :]
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads,
                          3 * self.head_dim)  # [Batch, SeqLen, 4, 3*96], considering embed_dim=384 and num_heads=4
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=matrix_mask_extended,
                                               temperature=self.temperature if temperature is None else temperature,
                                               dropout=self.dropout, training=self.training)

        attention = torch.mean(attention, dim=1)  # Average over heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o, attention


class SlidingWindowMultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, max_len, window_size, temperature=1, dropout=0.0, activation_attention="softmax"):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 module wrt the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.temperature = temperature
        self.dropout = dropout
        self.activation_attention = activation_attention
        self.qkv_proj = torch.nn.Linear(input_dim, 3 * embed_dim, bias=True)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self._init_params()

    def _init_params(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def create_sliding_window_mask(self, src_key_padding_mask, matrix_mask, seq_length, min_window_size=2, pad_value=-1e9):

        lens_valid_sent = (src_key_padding_mask == 0).sum(dim=1)
        # if valid_sent is longer than max_len, clip it to max_len
        clipped_lens = torch.clamp(lens_valid_sent, max=self.max_len)
        window_sizes = torch.ceil(clipped_lens*self.window_size/100).long().clamp(min=min_window_size)
        idx = torch.arange(seq_length, device=src_key_padding_mask.device)
        window_mask = (idx.view(1, -1, 1) - idx.view(1, 1, -1)).abs() > window_sizes.view(-1, 1, 1) # if masked window, then True
        padding_mask = (matrix_mask == pad_value) # if padded, then True
        merged_mask = window_mask | padding_mask # if at least one True then True
        matrix_mask[merged_mask] = pad_value # apply pad value

        return matrix_mask

    def forward(self, x, src_key_padding_mask, matrix_mask, temperature=None):
        batch_size, seq_length, _ = x.size()
        sliding_mask = self.create_sliding_window_mask(src_key_padding_mask, matrix_mask, seq_length)
        sliding_mask_extended = sliding_mask[:, None, :, :]
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)  # --- [Batch, SeqLen, 4, 3*96]
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=sliding_mask_extended,
                                               temperature=self.temperature if temperature is None else temperature,
                                               dropout=self.dropout, training=self.training)

        attention = torch.mean(attention, dim=1)  # Average over heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o, attention