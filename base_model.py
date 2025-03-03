import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torchmetrics import F1Score
import pandas as pd
from torch import optim
import pytorch_lightning as pl
import numpy  as np
from sentence_transformers import SentenceTransformer
from eval_models import eval_results

def scaled_dot_product(q, k, v, mask=None, temperature=1, dropout=0.0, training=True, attention="softmax"):

    factor = 1/np.sqrt(q.size(-1))
    attn_logits = q @ k.transpose(-2, -1) * factor
    
    if mask is not None:
        attn_logits += mask

    #attention = torch.softmax(attn_logits/temperature, dim=-1)
    if attention == "softmax":
        attention = nn.functional.softmax(attn_logits/temperature, dim=-1)
    elif attention == "sigmoid": #based on https://arxiv.org/pdf/2409.04431
        attention = nn.functional.sigmoid(attn_logits - torch.log(attn_logits.shape[-1]))
    elif attention == "relu": #based on https://arxiv.org/pdf/2309.08586
        attention = nn.functional.relu(attn_logits)/attn_logits.shape[-1]

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
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim, bias=True)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self._init_params()

    def _init_params(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, src_key_padding_mask, matrix_mask, temperature=None):
        matrix_mask_extended = matrix_mask[:,None,:,:]
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)      #--- [Batch, SeqLen, 4, 3*96]
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=matrix_mask_extended, temperature=self.temperature if temperature is None else temperature, dropout=self.dropout, training = self.training)
        
        attention = torch.mean(attention, dim=1) # Average over heads        
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o, attention


class SlidingWindowMultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, window_size=30, temperature=1, dropout=0.0, activation_attention="softmax"):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 module wrt the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
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
        window_sizes = torch.ceil(lens_valid_sent*self.window_size/100).long().clamp(min=min_window_size)
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
    

class Classifier_Lighting(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Train_" + k, v, prog_bar=True, batch_size=len(batch['documents_ids']))
        return loss["loss"]
        
    def validation_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Val_" + k, v, prog_bar=True, batch_size=len(batch['documents_ids']))  
        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward_performance(self, data):        
        out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))  
        
        loss = self.criterion(out, data['labels'])
        pred = out.argmax(dim=1) 
        acc=(pred == data['labels']).sum()/len(data['labels'])
        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)

        f1_ma=f1_score(pred, data['labels'])
        return {"loss": loss, "f1-ma":f1_ma, 'acc':acc} 

    def predict(self, test_loader, cpu_store=True):
        self.eval()
        #correct_test = 0
        #total_test = 0
        preds=[]
        full_attn_weights=[]
        all_labels = []
        all_doc_ids = []
        all_article_identifiers = []
        #all_batch_metrics={'acc':[], 'f1_ma':[]}
        with torch.no_grad():
            for data in test_loader:             
                out, att_w = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))   
                full_attn_weights.extend(att_w)
                all_doc_ids.extend(data['documents_ids'])
                pred = out.argmax(dim=1) 
                if cpu_store:
                    pred = pred.detach().cpu().numpy() 
                preds+=list(pred) 
                all_labels.extend(data['labels'])
                all_article_identifiers.extend(data['article_id'])

            if not cpu_store:
                preds = torch.Tensor(preds)
        
        return preds, full_attn_weights, all_labels, all_doc_ids, all_article_identifiers


class MHAClassifier(Classifier_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim,  max_len, lr, 
                 intermediate=False, num_heads=4, dropout=False, class_weights=[],
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent = "", activation_attention="softmax",
                 window=30):
        #print ("Creating Classifier Model")
        super(MHAClassifier, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
            
        #print ("Attention method:", activation_attention)
        # sliding window MHA
        if window:
            self.attention = SlidingWindowMultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(),
                                                                 embed_dim, num_heads, window_size=30, temperature=1,
                                                                 dropout=attn_dropout, activation_attention=activation_attention)
        # full MHA
        else:
            self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim,
                                                    num_heads, temperature=1, dropout=attn_dropout,
                                                    activation_attention=activation_attention)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        if not intermediate:
            self.fc = nn.Linear(self.embed_dim, num_classes)
        else: 
            self.fc1 = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second linear layer

        self.dropout = dropout #nn.Dropout(p=dropout)
        self.max_len = max_len
        self.lr = lr
        self.intermediate=intermediate   ###cambiar a sequential con num layers
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights) 
        else: 
            self.criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.temperature = 1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0

        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #print ("Done.")
        
        self.save_hyperparameters(ignore=["invert_vocab_sent"]) ### for loading later
    
    def training_step(self, batch, batch_idx):
        return_val = super(MHAClassifier, self).training_step(batch, batch_idx)
        if self.temperature_scheduler=="anneal_decrease":
            #step should be a very low value, 1e-4 or ( 1e-3 or 1e-5)
            self.temperature = max(0.1, np.exp(- self.temperature_step*self.global_iter) )
        self.global_iter+=1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHAClassifier, self).on_train_epoch_end()  
        if self.temperature_scheduler=="step_decrease":
            self.temperature = self.temperature - self.temperature_step 
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler=="step_decrease" or self.temperature_scheduler=="anneal_decrease":
            print ("Temperature at the end of epoch:", self.temperature)
        
        return return_val
        
    ### nuevos para checkpoint -- only works for MHAClassifiers trained with the last changes 
    def on_save_checkpoint(self, checkpoint) -> None:
        """Objects to include in checkpoint file"""
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        """Objects to retrieve from checkpoint file"""
        self.temperature= checkpoint["temperature_end_of_epoch"]


    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        #### TODO: HACER CARGA EMBEDDINGS PRE-TRAINED     
  
        x_emb=[]        
        for doc in doc_ids: 
            source= self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc!= 0]))
            complement= np.zeros((len(doc[doc== 0]), self.embed_dim))
            temp_emb= np.concatenate((source, complement)) #np.array
            x_emb.append(temp_emb)

        x_emb=torch.tensor(x_emb)

        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)


        if self.dropout!=False:
            attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        #masked pooling -- TODO: class token pooling
        attn_output[matrix_mask[:,0,:] != 0] = torch.nan
        attn_output = torch.nanmean(attn_output, dim=1)  # Mean pooling over the sequence length  #### que pasa con el masking de las no sentencias? para calcular loss  -- necesito recuperar el max length del documento 
        
        if not self.intermediate:
            logits = self.fc(attn_output)
        else: 
            attn_output = torch.relu(self.fc1(attn_output))
            logits = self.fc2(attn_output)
        return logits, attn_weights    


########
########
class MHAClassifier_extended(Classifier_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim,  max_len, lr, 
                 intermediate=False, num_heads=4, multi_layer=False, class_weights=[], 
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent = "", activation_attention="softmax"):
        #print ("Creating Classifier Model")
        super(MHAClassifier_extended, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
            
        #print ("Attention method:", activation_attention)
        self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim, num_heads, temperature=1, dropout=attn_dropout, activation_attention=activation_attention)
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        if intermediate:
            self.pre_linear = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.multi_attention = MultiHeadSelfAttention(hidden_dim, hidden_dim, num_heads, temperature=1, dropout=attn_dropout, activation_attention=activation_attention)
            
        self.head = nn.Linear(hidden_dim if intermediate else self.embed_dim, num_classes)  
        self.max_len = max_len
        self.lr = lr
        self.intermediate=intermediate   ###cambiar a sequential con num layers
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights) 
        else: 
            self.criterion = nn.CrossEntropyLoss()
        self.multi_layer = multi_layer  
        self.temperature = 1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0
        

        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #print ("Done.")
        
        self.save_hyperparameters(ignore=["invert_vocab_sent"]) ### for loading later
    
    def training_step(self, batch, batch_idx):
        return_val = super(MHAClassifier_extended, self).training_step(batch, batch_idx)
        if self.temperature_scheduler=="anneal_decrease":
            #step should be a very low value, 1e-4 or ( 1e-3 or 1e-5)
            self.temperature = max(0.1, np.exp(- self.temperature_step*self.global_iter) )
        self.global_iter+=1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHAClassifier_extended, self).on_train_epoch_end()  
        if self.temperature_scheduler=="step_decrease":
            self.temperature = self.temperature - self.temperature_step 
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler=="step_decrease" or self.temperature_scheduler=="anneal_decrease":
            print ("Temperature at the end of epoch:", self.temperature)
        
        return return_val
        
    ### nuevos para checkpoint -- only works for MHAClassifiers trained with the last changes 
    def on_save_checkpoint(self, checkpoint) -> None:
        """Objects to include in checkpoint file"""
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        """Objects to retrieve from checkpoint file"""
        self.temperature= checkpoint["temperature_end_of_epoch"]


    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        #### TODO: HACER CARGA EMBEDDINGS PRE-TRAINED     
  
        x_emb=[]        
        for doc in doc_ids: 
            source= self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc!= 0]))
            complement= np.zeros((len(doc[doc== 0]), self.embed_dim))
            temp_emb= np.concatenate((source, complement)) #np.array
            x_emb.append(temp_emb)

        x_emb=torch.tensor(x_emb)

        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)

        if self.intermediate:
            attn_output = torch.relu(self.pre_linear(attn_output))

        if self.multi_layer:
            attn_output, attn_weights = self.multi_attention(attn_output.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)

        #masked pooling -- TODO: class token pooling
        attn_output[matrix_mask[:,0,:] != 0] = torch.nan
        attn_output = torch.nanmean(attn_output, dim=1)  # Mean pooling over the sequence length  #### que pasa con el masking de las no sentencias? para calcular loss  -- necesito recuperar el max length del documento 
        
        logits = self.head(attn_output)
        return logits, attn_weights    
########
########


def retrieve_from_dict(dict, list_ids):
    return [dict[id.item()] for id in list_ids]


class Summarizer_Lighting(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Train_" + k, v, prog_bar=True, batch_size=len(batch['documents_ids']))
        return loss["loss"]
        
    def validation_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Val_" + k, v, prog_bar=True, batch_size=len(batch['documents_ids']))  
        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward_performance(self, data):    
        out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))  
        pred = out.view(-1, 2)
        labels_ = data['labels'].view(-1)
        mask_not_ignore = (labels_ != -1)
        pred = pred[mask_not_ignore]
        labels_ = labels_[mask_not_ignore]

        loss = self.criterion(pred, labels_)
        pred_cat = pred.argmax(dim=1) 

        acc=(pred_cat == labels_).sum()/len(labels_)

        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)
        f1_score_none = F1Score(task='multiclass', num_classes=self.num_classes, average=None).to(self.device)
        f1_ma=f1_score(pred_cat, labels_)
        
        return {"loss": loss, "f1-ma":f1_ma, 'acc':acc} 

    def predict(self, test_loader, cpu_store=True):
        self.eval()
        preds=[]
        full_attn_weights=[]
        all_labels = []
        all_doc_ids = []
        all_article_identifiers = []
        with torch.no_grad():
            for data in test_loader:             
                out, att_w = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))   
                full_attn_weights.extend(att_w)
                all_doc_ids.extend(data['documents_ids'])
                pred = out.view(-1, 2)
                labels_ = data['labels'].view(-1)
                
                mask_not_ignore = (labels_ != -1)
                pred = pred[mask_not_ignore]
                labels_ = labels_[mask_not_ignore]      
                pred = pred.argmax(dim=1) 

                if cpu_store:
                    pred = pred.detach().cpu().numpy() 
                preds+=list(pred) 
                all_labels.extend(labels_)
                all_article_identifiers.extend(data['article_id'])

            if not cpu_store:
                preds = torch.Tensor(preds)
        
        return preds, full_attn_weights, all_labels, all_doc_ids, all_article_identifiers

    def predict_to_file(self, test_loader, cpu_store=True, saving_file=False, filename="", path_root=""):
        #### run only with batch size 1
        self.eval()
        all_accs = []
        all_f1_scores = []
        with torch.no_grad():
            if saving_file:
                path_dataset = path_root+"/raw/"
                print ("\nCreating files for PyG dataset in:", path_dataset)
                predicting_docs = pd.DataFrame(columns=["article_id", "label", "doc_as_ids"])
                predicting_docs.to_csv(path_dataset+filename, index=False)

            for data in tqdm(test_loader, total=len(test_loader)):             
                out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))   

                pred = out.view(-1, 2)
                labels_ = data['labels'].view(-1)                
                mask_not_ignore = (labels_ != -1)
                pred = pred[mask_not_ignore]
                labels_ = labels_[mask_not_ignore]      
                pred = pred.argmax(dim=1) 

                if cpu_store:
                    pred = pred.detach().cpu().numpy() 

                ### eval results of batch 
                acc, f1_score = eval_results(torch.Tensor(pred), labels_, 2, None, print_results=False)
                all_accs.append(acc)
                all_f1_scores.append(f1_score)
                
                label = labels_ ##
                doc_as_ids = data['documents_ids'][0] ##
                article_id= data['article_id'] ##
                
                if saving_file:
                    try: 
                        predicting_docs.loc[len(predicting_docs)] = {
                            "article_id": article_id.item(),
                            "label": label.tolist(), 
                            "doc_as_ids": doc_as_ids[mask_not_ignore].tolist()
                            }
                        predicting_docs.to_csv(path_dataset+filename, index=False)
                    except:
                        print ("Error in saving file during model prediction")
                        break

        return all_accs, all_f1_scores #preds, full_attn_weights, all_labels, all_doc_ids, all_article_identifiers


    def predict_single(self, batch_single, cpu_store=True):
        self.eval()
        preds=[]    
        with torch.no_grad():                
            out, att_w = self(batch_single['documents_ids'].to(self.device), batch_single['src_key_padding_mask'].to(self.device), batch_single['matrix_mask'].to(self.device))   
            pred = out.view(-1, 2)
            labels_ = batch_single['labels'].view(-1)
            
            mask_not_ignore = (labels_ != -1)
            pred = pred[mask_not_ignore]  
            pred = pred.argmax(dim=1) 

            if cpu_store:
                pred = pred.detach().cpu().numpy() 
            preds+=list(pred) 

        if not cpu_store:
            preds = torch.Tensor(preds)
        
        return preds, att_w
    

class MHASummarizer_extended(Summarizer_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim,  max_len, lr, 
                 intermediate=False, num_heads=4, multi_layer=False, class_weights = [], #criterion = nn.CrossEntropyLoss() 
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent = '', activation_attention="softmax"):
        #print ("Creating Summarizer Model...")
        super(MHASummarizer_extended, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
            
        #print ("Attention method:", activation_attention)    
        self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim, num_heads, temperature=1, dropout=attn_dropout, activation_attention=activation_attention)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        if intermediate:
            self.pre_linear = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.multi_attention = MultiHeadSelfAttention(hidden_dim, hidden_dim, num_heads, temperature=1, dropout=attn_dropout, activation_attention=activation_attention)

        self.head = nn.Linear(hidden_dim if intermediate else self.embed_dim, num_classes)  

        self.max_len = max_len
        self.lr = lr
        self.intermediate=intermediate  
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)  #criterion
        #self.dropout = dropout
        self.multi_layer = multi_layer
        self.temperature = 1 #torch.nn.parameter.Parameter(torch.Tensor(1), requires_grad=False) #1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0
        
        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #print ("Done.")

        self.save_hyperparameters(ignore=["invert_vocab_sent"]) ### for loading later
    
    def training_step(self, batch, batch_idx):
        return_val = super(MHASummarizer_extended, self).training_step(batch, batch_idx)
        if self.temperature_scheduler=="anneal_decrease":
            #step should be a very low value, 1e-4 or ( 1e-3 or 1e-5)
            self.temperature = max(0.1, np.exp(- self.temperature_step*self.global_iter) )
        self.global_iter+=1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHASummarizer_extended, self).on_train_epoch_end()  
        if self.temperature_scheduler=="step_decrease":
            self.temperature = self.temperature - self.temperature_step
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler=="step_decrease" or self.temperature_scheduler=="anneal_decrease":
            print ("Temperature at the end of epoch:", self.temperature)
        
        return return_val
    
    def on_save_checkpoint(self, checkpoint) -> None:
        """Objects to include in checkpoint file"""
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        """Objects to retrieve from checkpoint file"""
        self.temperature= checkpoint["temperature_end_of_epoch"]
        
    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        x_emb=[]        
        for doc in doc_ids: 
            source= self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc!= 0]))
            complement= np.zeros((len(doc[doc== 0]), self.embed_dim))
            temp_emb= np.concatenate((source, complement)) 
            x_emb.append(temp_emb)
        
        x_emb=torch.tensor(x_emb)
        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)
      
        #if only one layer:
        if self.intermediate:
            attn_output = torch.relu(self.pre_linear(attn_output))

        if self.multi_layer:
            attn_output, attn_weights = self.multi_attention(attn_output.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)
        
        logits = self.head(attn_output)
        return logits, attn_weights    
    
    ##########################


class MHASummarizer(Summarizer_Lighting):
    def __init__(self, embed_dim, num_classes, hidden_dim,  max_len, lr, 
                 intermediate=False, num_heads=4, dropout=False, class_weights = [], #criterion = nn.CrossEntropyLoss() 
                 temperature_scheduler=None, temperature_step=None, attn_dropout=0.0,
                 path_invert_vocab_sent = '', activation_attention="softmax"):
        #print ("Creating Summarizer Model...")
        super(MHASummarizer, self).__init__()
        self.sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        for name, param in self.sent_model.named_parameters():
            param.requires_grad = False
            
        #print ("Attention method:", activation_attention)    
        self.attention = MultiHeadSelfAttention(self.sent_model.get_sentence_embedding_dimension(), embed_dim, num_heads, temperature=1, dropout=attn_dropout, activation_attention=activation_attention)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if not intermediate:
            self.fc = nn.Linear(self.embed_dim, num_classes)
        else: 
            self.fc1 = nn.Linear(self.embed_dim, hidden_dim)  # First linear layer
            self.fc2 = nn.Linear(hidden_dim, num_classes)  # Second linear layer

        self.dropout = dropout #nn.Dropout(p=dropout)
        self.max_len = max_len
        self.lr = lr
        self.intermediate=intermediate  
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)  #criterion
        self.dropout = dropout
        self.temperature = 1 #torch.nn.parameter.Parameter(torch.Tensor(1), requires_grad=False) #1
        self.temperature_scheduler = temperature_scheduler
        self.temperature_step = temperature_step
        self.global_iter = 0
        
        sent_dict_disk = pd.read_csv(path_invert_vocab_sent+"vocab_sentences.csv")
        self.invert_vocab_sent = {k:v for k,v in zip(sent_dict_disk['Sentence_id'],sent_dict_disk['Sentence'])}
        #print ("Done.")

        self.save_hyperparameters(ignore=["invert_vocab_sent"]) ### for loading later
    
    def training_step(self, batch, batch_idx):
        return_val = super(MHASummarizer, self).training_step(batch, batch_idx)
        if self.temperature_scheduler=="anneal_decrease":
            #step should be a very low value, 1e-4 or ( 1e-3 or 1e-5)
            self.temperature = max(0.1, np.exp(- self.temperature_step*self.global_iter) )
        self.global_iter+=1

        return return_val

    def on_train_epoch_end(self):
        return_val = super(MHASummarizer, self).on_train_epoch_end()  
        if self.temperature_scheduler=="step_decrease":
            self.temperature = self.temperature - self.temperature_step
        self.temperature = max(self.temperature, 0.1)
        if self.temperature_scheduler=="step_decrease" or self.temperature_scheduler=="anneal_decrease":
            print ("Temperature at the end of epoch:", self.temperature)
        
        return return_val
    
    def on_save_checkpoint(self, checkpoint) -> None:
        #Objects to include in checkpoint file
        checkpoint["temperature_end_of_epoch"] = self.temperature

    def on_load_checkpoint(self, checkpoint) -> None:
        #Objects to retrieve from checkpoint file
        self.temperature= checkpoint["temperature_end_of_epoch"]
        
    def forward(self, doc_ids, src_key_padding_mask, matrix_mask):
        #### TODO: HACER CARGA EMBEDDINGS PRE-TRAINED     
        x_emb=[]        
        for doc in doc_ids: 
            source= self.sent_model.encode(retrieve_from_dict(self.invert_vocab_sent, doc[doc!= 0]))
            complement= np.zeros((len(doc[doc== 0]), self.embed_dim))
            temp_emb= np.concatenate((source, complement)) #np.array
            x_emb.append(temp_emb)
        
        x_emb=torch.tensor(x_emb)
        attn_output, attn_weights = self.attention(x_emb.float().to(self.device), src_key_padding_mask, matrix_mask, temperature=self.temperature)

        if self.dropout!=False:
            attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
               

        if not self.intermediate:
            logits = self.fc(attn_output)
        #with 2 layers:
        else: 
            attn_output = torch.relu(self.fc1(attn_output))
            logits = self.fc2(attn_output)

        return logits, attn_weights  
    
    ##########################