from __future__ import print_function, division
import torch.nn.functional as F
# 计算损失函数和处理预测图像的函数
# 这些函数的作用是帮助在图像分割任务中评估模型性能和处理预测结果。
# dice_loss 和 calc_loss 函数用于计算损失值，而 threshold_predictions_v 和 threshold_predictions_p 函数用于对预测图像进行二值化处理。
def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
# 该函数计算 Dice Loss，用于评估预测图像与目标图像之间的相似度。
# prediction 和 target 是输入的预测图像和目标图像。
# smooth 是一个平滑项，用于防止分母为零。
# i_flat 和 t_flat 将输入的预测图像和目标图像展平成一维向量。
# intersection 计算预测图像和目标图像之间的交集。
# 返回值是 1 减去 Dice 系数，得到的值越小，预测结果与目标图像越相似。

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss
# 该函数计算总体损失，包括二值交叉熵（BCE）损失和 Dice Loss。
# bce_weight 参数控制 BCE 损失和 Dice Loss 之间的权重平衡，默认为 0.5。
# 使用 F.binary_cross_entropy_with_logits 计算 BCE 损失。
# 使用 F.sigmoid 将预测图像转换为概率。
# 使用 dice_loss 函数计算 Dice Loss。
# 最终损失是 BCE 损失和 Dice Loss 的加权和。

def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds
# 该函数对预测图像进行阈值处理。
# thr 是阈值，默认为 150。
# 创建 thresholded_preds 的副本。
# 将小于阈值 thr 的所有像素值设置为 0，大于等于阈值的所有像素值设置为 255。
# 返回阈值处理后的图像。

def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds
# 该函数也对预测图像进行阈值处理。
# thr 是阈值，默认为 0.01。
# 创建 thresholded_preds 的副本。
# 将小于阈值 thr 的所有像素值设置为 0，大于等于阈值的所有像素值设置为 1。
# 返回阈值处理后的图像。