import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import F1Score
from torch import optim
import pytorch_lightning as pl
from tqdm import tqdm

from src.pipeline.eval import eval_results


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
        out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device),
                      data['matrix_mask'].to(self.device))

        loss = self.criterion(out, data['labels'])
        pred = out.argmax(dim=1)
        acc = (pred == data['labels']).sum() / len(data['labels'])
        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)

        f1_ma = f1_score(pred, data['labels'])
        return {"loss": loss, "f1-ma": f1_ma, 'acc': acc}

    def predict(self, test_loader, cpu_store=True, flag_file=False):
        self.eval()
        preds = []
        full_attn_weights = []
        all_labels = []
        all_doc_ids = []
        all_article_identifiers = []

        with torch.no_grad():
            for data in test_loader:
                if flag_file == False:
                    out, att_w = self(data['documents_ids'].to(self.device),
                                      data['src_key_padding_mask'].to(self.device), data['matrix_mask'].to(self.device))
                    full_attn_weights.extend(att_w)
                    pred = out.argmax(dim=1)

                all_doc_ids.extend(data['documents_ids'])

                if flag_file == False:
                    if cpu_store:
                        pred = pred.detach().cpu().numpy()
                    preds += list(pred)

                all_labels.extend(data['labels'])
                all_article_identifiers.extend(data['article_id'])

            if not cpu_store:
                preds = torch.Tensor(preds)

        return preds, full_attn_weights, all_labels, all_doc_ids, all_article_identifiers

    def predict_minimal(
            self,
            test_loader,
            cpu_store=True,
            return_attn=False
    ):
        self.eval()
        preds = []
        all_labels = []
        all_article_identifiers = []
        full_attn_weights = [] if return_attn else None # OPTIONAL: only if needed
        with torch.no_grad():
            for data in test_loader:
                out, att_w = self(
                    data['documents_ids'].to(self.device),
                    data['src_key_padding_mask'].to(self.device),
                    data['matrix_mask'].to(self.device),
                )
                pred = out.argmax(dim=1)

                if cpu_store:
                    preds.append(pred.cpu())
                else:
                    preds.append(pred)

                if return_attn:
                    full_attn_weights.append(att_w.cpu())
                all_labels.extend(data['labels'])
                all_article_identifiers.extend(data['article_id'])

                del out, att_w, pred
                torch.cuda.empty_cache()
        preds = torch.cat(preds)
        return preds, full_attn_weights, all_labels, all_article_identifiers

    def predict_single(self, batch_single, cpu_store=True):
        self.eval()
        preds = []
        with torch.no_grad():
            out, att_w = self(batch_single['documents_ids'].to(self.device),
                              batch_single['src_key_padding_mask'].to(self.device),
                              batch_single['matrix_mask'].to(self.device))
            pred = out.argmax(dim=1)

            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds += list(pred)

        if not cpu_store:
            preds = torch.Tensor(preds)

        return preds, att_w


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
        out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device),
                      data['matrix_mask'].to(self.device))
        pred = out.view(-1, 2)
        labels_ = data['labels'].view(-1)
        mask_not_ignore = (labels_ != -1)
        pred = pred[mask_not_ignore]
        labels_ = labels_[mask_not_ignore]

        loss = self.criterion(pred, labels_)
        pred_cat = pred.argmax(dim=1)

        acc = (pred_cat == labels_).sum() / len(labels_)

        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)
        f1_score_none = F1Score(task='multiclass', num_classes=self.num_classes, average=None).to(self.device)
        f1_ma = f1_score(pred_cat, labels_)

        return {"loss": loss, "f1-ma": f1_ma, 'acc': acc}

    def predict(self, test_loader, cpu_store=True):
        self.eval()
        preds = []
        full_attn_weights = []
        all_labels = []
        all_doc_ids = []
        all_article_identifiers = []
        with torch.no_grad():
            for data in test_loader:
                out, att_w = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device),
                                  data['matrix_mask'].to(self.device))
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
                preds += list(pred)
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
        predicting_docs = []

        if saving_file:
            path_dataset = path_root + "/raw/"
            if not os.path.exists(path_dataset):
                os.makedirs(path_dataset)
            print("\nCreating files for PyG dataset in:", path_dataset)

        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):
                out, _ = self(data['documents_ids'].to(self.device), data['src_key_padding_mask'].to(self.device),
                              data['matrix_mask'].to(self.device))
                for ide, out_sample in enumerate(out):

                    pred = out_sample.view(-1, 2)
                    labels_ = data['labels'][ide].view(-1)
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

                    label = labels_
                    doc_as_ids = data['documents_ids'][ide]
                    article_id = data['article_id'][ide]

                    if saving_file:
                        try:
                            predicting_doc = {
                                "article_id": article_id.item() if type(article_id) == torch.Tensor else article_id,
                                "label": label.tolist(),
                                "doc_as_ids": doc_as_ids[mask_not_ignore].tolist()
                            }
                            predicting_docs.append(predicting_doc)
                        except:
                            print("Error in saving file during model prediction")
                            break

            pd.DataFrame(predicting_docs).to_csv(path_dataset + filename, index=False)

        return all_accs, all_f1_scores

    def predict_single(self, batch_single, cpu_store=True):
        self.eval()
        preds = []
        with torch.no_grad():
            out, att_w = self(batch_single['documents_ids'].to(self.device),
                              batch_single['src_key_padding_mask'].to(self.device),
                              batch_single['matrix_mask'].to(self.device))
            pred = out.view(-1, 2)
            labels_ = batch_single['labels'].view(-1)

            mask_not_ignore = (labels_ != -1)
            pred = pred[mask_not_ignore]
            pred = pred.argmax(dim=1)

            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds += list(pred)

        if not cpu_store:
            preds = torch.Tensor(preds)

        return preds, att_w
