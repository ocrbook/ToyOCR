import functools
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """ Reduce loss as specified.

    :param loss(Tensor): Elementwise loss tensor.
    :param reduction(str): Options are "none", "mean" and "sum".
    :return:
        Tensor: Reduced loss tensor.
    """

    reduction_enum = F._Reduction.get_enum(reduction)

    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduction loss.

    :param loss(Tensor): ELement-wise loss.
    :param weight:  Element-wise weights
    :param reduction:  Same as pytorch
    :param avg_factor:  Average factor when computing the mean of losses.
    :return:
        Tensor: Processed loss values.
    """

    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)

    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor

        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


def weighted_loss(loss_func):
    """
    :param loss_func:
    :return:
    @weighted_loss
    def l1_loss(pred,target):
         return (pred-target).abs()

    pred=torch.Tensor([0,2,3])
    target=torch.Tensor([1,1,1])
    l1_loss(pred,target)
    tensor(1.3333)
    l1_loss(pred,target,weight)
    tensor(1.)

    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss

    return wrapper
