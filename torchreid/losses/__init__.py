from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss, CrossEntropyLoss_PerImg
from .cross_entropy_loss_negative import CrossEntropyLoss_Neg
from .hard_mine_triplet_loss import TripletLoss
from .hct_loss import HctLoss


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss