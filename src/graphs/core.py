from torchmetrics import F1Score
import torch
import pytorch_lightning as pl
import torch.optim as optim


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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward_performance(self, data):
        try:
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device),
                       data.batch.to(self.device))
        except:
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None,
                       data.batch.to(self.device))

        loss = self.criterion(out, data.y.long())
        pred = out.argmax(dim=1)
        acc = (pred == data.y).sum() / len(data.y)
        f1_score = F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.device)
        f1_ma = f1_score(pred, data.y)

        return {"loss": loss, "f1-ma": f1_ma, 'acc': acc}

    def predict(self, loader, cpu_store=True):
        self.eval()
        preds = []
        all_labels = []

        with torch.no_grad():
            for data in loader:
                try:
                    out = self(data.x.float().to(self.device), data.edge_index.to(self.device),
                               data.edge_attr.to(self.device), data.batch.to(self.device))
                except:
                    out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None,
                               data.batch.to(self.device))

                pred = out.argmax(dim=1)
                if cpu_store:
                    pred = pred.detach().cpu().numpy()
                preds += list(pred)
                all_labels.extend(data.y)

            if not cpu_store:
                preds = torch.Tensor(preds)

        return preds, all_labels