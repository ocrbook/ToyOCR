import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred, target):
    """Warper of mse loss."""
    return F.mse_loss(pred, target, reduction="none")


class MSELoss(nn.Module):
    """MSELoss

    Args:
        reduction (str,optional): The method that reduces the loss to a scalar.Options are "none","mean" and "sum"
        loss_weight(float,optional):The weight of the loss.Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
        :param pred(torch.Tensor): The prediction result.
        :param target(torch.Tensor): THe gt
        :param weight(torch.Tensor): The Weight of the loss for each prediciton.Default is None.
        :param avg_factor(int,optional): Average factor that is used to average the loss.Defaulf is none.
        :return:
            torch.Tensor: The calculated loss
        """

        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction, avg_factor=avg_factor)

        return loss
    
    

class BalanceL1Loss(nn.Module):
    def __init__(self, negative_ratio=3.):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, gt, mask):
        '''
        Args:
            pred: (N, 1, H, W).
            gt: (N, H, W).
            mask: (N, H, W).
        '''
        loss = torch.abs(pred[:, 0] - gt)
        positive = loss * mask
        negative = loss * (1 - mask)
        positive_count = int(mask.sum())
        negative_count = min(
                int((1 - mask).sum()),
                int(positive_count * self.negative_ratio))
        negative_loss, _ = torch.topk(negative.view(-1), negative_count)
        negative_loss = negative_loss.sum() / negative_count
        positive_loss = positive.sum() / positive_count
        return positive_loss + negative_loss,dict(l1_loss=positive_loss, nge_l1_loss=negative_loss)
