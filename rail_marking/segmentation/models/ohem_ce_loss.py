#!/usr/bin/env python
import torch
import torch.nn as nn


__all__ = ["OHEMCELoss"]


class OHEMCELoss(nn.Module):
    def __init__(self, thresh, weighted_values=None, ignore_lb=255):
        super(OHEMCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb

        if weighted_values is not None:
            weighted_values = torch.FloatTensor(weighted_values)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weighted_values = weighted_values.to(device)

        self.criteria = nn.CrossEntropyLoss(weight=weighted_values, ignore_index=ignore_lb, reduction="none")

    def forward(self, logits, labels):
        assert logits.device == labels.device

        device = logits.device
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels.long()).view(-1)
        loss_hard = loss[loss > self.thresh.to(device)]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
