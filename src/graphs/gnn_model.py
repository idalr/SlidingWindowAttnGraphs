import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

from torchmetrics import F1Score
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch_geometric
import torch.optim as optim
import torch.nn.functional as F
#from torch_geometric.data import DataLoader #Dataset, Data,
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader


class GNN_LightingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 

        for k, v in loss.items():
            self.log("Train_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1])) 
        return loss["loss"]
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 

        for k, v in loss.items():
            self.log("Val_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1]))  

        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) #1e-3
        return optimizer
    
    def forward_performance(self, data):        
        try: 
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.batch.to(self.device))  
        except: 
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None, data.batch.to(self.device))  
        
        loss = self.criterion(out, data.y.long())
        pred = out.argmax(dim=1) 
        

        acc=(pred == data.y).sum()/len(data.y)
        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)
        f1_ma=f1_score(pred, data.y)

        return {"loss": loss, "f1-ma":f1_ma, 'acc':acc} 
           
    
    def predict(self, loader, cpu_store=True):
        self.eval()
        preds=[]
        all_labels = []

        with torch.no_grad():
            for data in loader:  
                try: 
                    out = self(data.x.float().to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.batch.to(self.device))  
                except:
                    out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None, data.batch.to(self.device))  
                    
                pred = out.argmax(dim=1) 
                if cpu_store:
                    pred = pred.detach().cpu().numpy() 
                preds+=list(pred) 
                all_labels.extend(data.y)

            if not cpu_store:
                preds = torch.Tensor(preds)

        return preds, all_labels
    
    
class GAT_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, num_heads=4, dropout=False, class_weights=[]):
        print ("Creating GAT model")
        super(GAT_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()        
        self.convs.append(GATConv(in_features, num_hidden_conv, heads=num_heads))
        for l in range(num_layers-1):
            self.convs.append(GATConv(num_hidden_conv * num_heads, num_hidden_conv, heads=num_heads))

        self.lin = nn.Linear(num_hidden_conv * num_heads, out_features)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights) 
        else: 
            self.criterion = nn.CrossEntropyLoss()

        print ("Done")
        
        
    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr==None: 
                x = self.convs[i](x, edge_index)
            else: 
                x = self.convs[i](x, edge_index, edge_attr.float())
                        
            x = F.relu(x)  
            if self.dropout!=False:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x

    
class GCN_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, dropout=False, class_weights=[]):
        print ("Creating GCN model")
        super(GCN_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()        
        self.convs.append(GCNConv(in_features, num_hidden_conv))
        for l in range(num_layers-1):
            self.convs.append(GCNConv(num_hidden_conv, num_hidden_conv))

        self.lin = nn.Linear(num_hidden_conv, out_features)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights) 
        else: 
            self.criterion = nn.CrossEntropyLoss()
        print ("Done")
        
    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr==None: 
                x = self.convs[i](x, edge_index)
            else: 
                x = self.convs[i](x, edge_index, edge_attr.float())
                
            emb = x
            x = F.relu(x)
            if self.dropout!=False:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x    


def check_bs(bs, len_data):
    if len_data % bs == 1:
        return bs-1
    else:
        return bs
    
def partitions(dataset, dataset_test, dataset_val= None, bs=32, trainp=0.8, valp=0.2):
    if np.round(trainp+valp)!=1.0:
        print ("Partitions don't fit. Please specify partitions that sum 1.")
        return 
    
    if dataset_val is None:   
        dataset = dataset.shuffle()
        total=len(dataset)
        a=int(total*trainp)
        dataset_train = dataset[:a]
        dataset_val = dataset[a:]
    else:
        dataset_train = dataset
    
    bs_tr = check_bs(bs, len(dataset_train))
    bs_val = check_bs(bs, len(dataset_val))
    bs_test = check_bs(bs, len(dataset_test))

    train_loader = DataLoader(dataset_train, batch_size=bs_tr, shuffle=True, num_workers=1) 
    val_loader = DataLoader(dataset_val, batch_size=bs_val, shuffle=False, num_workers=1) 
    test_loader = DataLoader(dataset_test, batch_size=bs_test, shuffle=False, num_workers=1)
            
    return train_loader, val_loader, test_loader

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(6,6))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


class GAT_NC_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, num_heads=4, dropout=False, class_weights=[], with_edge_attr=True):
        print ("Creating GAT model")
        super(GAT_NC_model, self).__init__()
        ##torch.manual_seed(1234567)
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.with_edge_attr = with_edge_attr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()        
        self.convs.append(GATConv(in_features, num_hidden_conv, heads=num_heads))
        for l in range(num_layers-1):
            self.convs.append(GATConv(num_hidden_conv * num_heads, num_hidden_conv, heads=num_heads))
        self.lastConv = GATConv(num_hidden_conv * num_heads, out_features) ### nn.Linear
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights) 
        else: 
            self.criterion = nn.CrossEntropyLoss()
        print ("Done")        
        
    def forward(self, x, edge_index, edge_attr, batch): #
        for i in range(self.num_layers):
            if self.with_edge_attr: 
                x = self.convs[i](x, edge_index, edge_attr.float())
            else: #edge_attr==None or empty: 
                x = self.convs[i](x, edge_index)              

            x = F.relu(x)  
            if self.dropout!=False:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.with_edge_attr:         
            x = self.lastConv(x, edge_index, edge_attr.float())
        else: 
            x = self.lastConv(x, edge_index)    
        
        return x
