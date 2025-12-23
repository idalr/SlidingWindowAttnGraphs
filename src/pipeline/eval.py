import torch
from torchmetrics import F1Score


def eval_results(preds, all_labels, num_classes, partition, print_results=True):
    labels = torch.as_tensor(all_labels, device=preds.device)
    acc = (labels == preds).float().mean()
    f1_score = F1Score(task='multiclass', num_classes=num_classes, average=None).to(preds.device)
    f1_all = f1_score(preds.int(), labels.int())
    if print_results:
        print("\nTrained Model Results on partition:", partition)
        print(partition + " Acc:", acc)
        print(partition + " F1-score macro:", f1_all.mean())
        print(partition + " F1-score for each class:", f1_all)

    return acc, f1_all
