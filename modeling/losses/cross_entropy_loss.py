import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None):
    """Calculate the CrossEntropy loss.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.
    :return:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()

    if inds.numel() > 0:
        bin_label_weights = None

    else:
        bin_label_weights = label_weights.view(-1, 0).expand(label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None):
    """Calculate the binary CrossEntropy loss.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.

    :return:
        torch.Tensor: The calculated loss

    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction="none")

    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None):
    """Calculate the  CrossEntropy loss for masks.

    :param pred(torch.Tensor): The prediction with shape (N,C),C is the number of the classes.
    :param label(torch.Tensor): The gt
    :param weight(torch.Tensor): Sample-wise loss weight.
    :param reduction(torch.Tensor): The method used to reduce the loss.
    :param avg_factor(torch.Tensor): Average factor that is used to average the loss.Defaults to None
    :param class_weight(torch.Tensor): The weight for each class.

    :return:
        torch.Tensor: The calculated loss
    """
    assert reduction == 'mean' and avg_factor is None

    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0):
        """Cross EntropyLoss


        :param use_sigmoid: (bool,optional) whether the prediction uses sigmoid
        :param use_mask:  (bool,optional) whether the prediction uses mask
        :param reduction:  (str,optional) Defaults to 'mean'.
        :param class_weight: (list[float],optional) Weight of each class.
        :param loss_weight:  (float,optional) weight of the loss.Defaults to 1.0


        """

        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy

        if self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        :param cls_score: (torch.Tensor)
        :param label:
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :param kwargs:
        :return:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight,
                                                         reduction=reduction, avg_factor=avg_factor, **kwargs)

        return loss_cls
