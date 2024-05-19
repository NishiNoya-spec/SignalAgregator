import torch
import torch.nn.functional as F

from exploration.models.deep_learning.constants import (
    MAX_VALUES,
    MIN_VALUES,
    STEP,
)


class ModifiedMSELoss(torch.nn.Module):

    def __init__(
        self, threshold, empty_count_left, empty_count_right,
        important_edge_area_m, step
    ):
        super(ModifiedMSELoss, self).__init__()
        self.mse_criterion = torch.nn.MSELoss()
        self.dice_criterion = DiceLoss()
        self.threshold_1 = (
            (-threshold - MIN_VALUES['LNK100_Torsion'])
            / MAX_VALUES['LNK100_Torsion']
        )
        self.threshold_2 = (
            (threshold - MIN_VALUES['LNK100_Torsion'])
            / MAX_VALUES['LNK100_Torsion']
        )
        self.empty_count_left = empty_count_left
        self.empty_count_right = empty_count_right
        self.important_edge_area = int(important_edge_area_m / STEP)
        self.step = step

    def forward(self, predictions, targets, epoch):
        target = targets[:, :, self.empty_count_left:-self.empty_count_right]
        prediction = predictions[:, :,
                                 self.empty_count_left:-self.empty_count_right]

        target_left_edge, target_right_edge, target_middle = (
            self._get_parts(target)
        )
        prediction_left_edge, prediction_right_edge, prediction_middle = (
            self._get_parts(prediction)
        )

        left_mse_loss = self.mse_criterion(
            prediction_left_edge,
            target_left_edge,
        )

        return left_mse_loss

    def _get_defect_data(self, data):
        return (data <= self.threshold_1) | (data >= self.threshold_2)

    def _get_parts(self, data):
        left_edge = data[:, :, :self.important_edge_area]
        right_edge = data[:, :, -self.important_edge_area:]
        middle = data[:, :, self.important_edge_area:-self.important_edge_area]
        return left_edge, right_edge, middle


class DiceLoss(torch.nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class BCEDiceLoss(torch.nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target, bce_coef=0.01, dice_coef=0.99):
        return bce_coef * self.bce(pred, target
                                   ) + dice_coef * self.dice(pred, target)


# Пример использования
# loss = hausdorff_loss(outputs, labels)
def hausdorff_loss(y_pred, y_true):
    dist_matrix = torch.cdist(y_pred, y_true)
    max_dist = torch.max(
        torch.max(torch.min(dist_matrix, dim=0)[0]),
        torch.max(torch.min(dist_matrix, dim=1)[0])
    )
    loss = F.relu(max_dist)
    return loss


# Пример использования
# quantile = 0.5  # Пример квантиля (медиана)
# loss = quantile_loss(outputs, labels, quantile)
def quantile_loss(y_pred, y_true, quantile):
    error = y_true - y_pred
    loss = torch.max(quantile * error, (quantile - 1) * error)
    return loss.mean()
