import torch
import torch.nn as nn

from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """ Smooth L1 loss.

    :param pred(torch.Tensor): The prediction.
    :param target(torch.Tensor): The learning target of the predition
    :param beta(float,optional): The threshold in the piecewise function.
            Defaults to 1.0.
    :return:
            torch.Tneosr:Calculated loss
    """

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    return loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss

    :param pred(torch.Tensor): The prediction.
    :param target(torch.Tensor): The learning target of the prediction.
    :return:
        torch.Tensor: Calculated loss
    """

    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


class SmoothL1loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta(float,optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction(str,optional): The method to reduce the loss.
            Options are "none","mean" and "sum".Defaults to "mean"
        loss_weight(float,optional): The weight of loss.

    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function

        :param pred(torch.Tensor): The prediction
        :param target(torch.Tensor): The gt
        :param weight(torch.Tensor,optional): The weight of loss for each prediction.Default is None
        :param avg_factor(int,optional): Average factor that is used to average the loss.
        :param reduction_override(str,optional): The reduction method used to override the original reduction method of
                loss.
        :param kwargs:
        :return:

        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, weight, beta=self.beta, reduction=reduction,
                                                      avg_factor=avg_factor, **kwargs)
