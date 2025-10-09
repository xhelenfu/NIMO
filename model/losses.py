import torch
import torch.nn.functional as F


class ResistanceScoreLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(ResistanceScoreLoss, self).__init__()
        self.reduction = reduction

    def forward(self, labels, attention):
        attention_probabilities = torch.sigmoid(attention)

        labels = labels.float()

        loss = F.binary_cross_entropy(
            attention_probabilities, labels, reduction=self.reduction
        )

        return loss
