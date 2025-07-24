import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

from src.graphs.core import GNN_LightingModel


class GAT_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, num_heads=4, dropout=False,
                 class_weights=[]):
        print("Creating GAT model")
        super(GAT_model, self).__init__()
        self.num_classes = out_features
        self.class_weights = class_weights
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_features, num_hidden_conv, heads=num_heads))
        for l in range(num_layers - 1):
            self.convs.append(GATConv(num_hidden_conv * num_heads, num_hidden_conv, heads=num_heads))

        self.lin = nn.Linear(num_hidden_conv * num_heads, out_features)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        print("Done")

    def on_save_checkpoint(self, checkpoint) -> None:
        """Objects to include in checkpoint file"""
        checkpoint["class_weights"] = self.class_weights

    def on_load_checkpoint(self, checkpoint) -> None:
        """Objects to retrieve from checkpoint file"""
        self.class_weights = checkpoint["class_weights"]

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr == None:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_attr.float())

            x = F.relu(x)
            if self.dropout != False:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


class GCN_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, dropout=False, class_weights=[]):
        print("Creating GCN model")
        super(GCN_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, num_hidden_conv))
        for l in range(num_layers - 1):
            self.convs.append(GCNConv(num_hidden_conv, num_hidden_conv))

        self.lin = nn.Linear(num_hidden_conv, out_features)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        print("Done")

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr == None:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_attr.float())

            emb = x
            x = F.relu(x)
            if self.dropout != False:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


def check_bs(bs, len_data):
    if len_data % bs == 1:
        return bs - 1
    else:
        return bs


def partitions(dataset_train, dataset_test, dataset_val=None, bs=32, trainp=0.8, valp=0.2):
    modified = False

    if np.round(trainp + valp) != 1.0:
        print("Partitions do not fit. Please specify partitions that sum 1.")
        return

    if dataset_val is None:
        dataset_train = dataset_train.shuffle()
        total = len(dataset_train)
        a = int(total * trainp)
        dataset_train_final = dataset_train[:a]
        dataset_val = dataset_train[a:]
        modified = True
    else:
        pass

    if modified == True:
        bs_tr = check_bs(bs, len(dataset_train_final))
    else:
        bs_tr = check_bs(bs, len(dataset_train))

    bs_val = check_bs(bs, len(dataset_val))
    bs_test = check_bs(bs, len(dataset_test))
    print("Batch sizes: Train:", bs_tr, "Validation:", bs_val, "Test:", bs_test)

    if modified == True:
        train_loader = DataLoader(dataset_train_final, batch_size=bs_tr, shuffle=True, num_workers=1)
    else:
        train_loader = DataLoader(dataset_train, batch_size=bs_tr, shuffle=True, num_workers=1)

    val_loader = DataLoader(dataset_val, batch_size=bs_val, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset_test, batch_size=bs_test, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader


class GAT_NC_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, num_heads=4, dropout=False,
                 class_weights=[], with_edge_attr=True):
        print("Creating GAT model")
        super(GAT_NC_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.with_edge_attr = with_edge_attr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_features, num_hidden_conv, heads=num_heads))
        for l in range(num_layers - 1):
            self.convs.append(GATConv(num_hidden_conv * num_heads, num_hidden_conv, heads=num_heads))
        self.lastConv = GATConv(num_hidden_conv * num_heads, out_features)  ### nn.Linear
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        print("Done")

    def forward(self, x, edge_index, edge_attr, batch):  #
        for i in range(self.num_layers):
            if self.with_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr.float())
            else:  # edge_attr==None or empty:
                x = self.convs[i](x, edge_index)

            x = F.relu(x)
            if self.dropout != False:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.with_edge_attr:
            x = self.lastConv(x, edge_index, edge_attr.float())
        else:
            x = self.lastConv(x, edge_index)

        return x