import numpy as np
from scipy import spatial
# 图像分割任务的评估指标，包括 Dice 系数、准确率和数值评分

def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum
# 该函数用于计算两个二值图像之间的 Dice 系数，这是一个衡量图像相似度的指标。
# im1 和 im2 是输入的两个图像。
# 将输入图像转换为布尔类型（np.bool），然后确保两个图像的形状相同。
# 对图像进行二值化处理（像素值大于 0.5 为 True，否则为 False）。
# 计算两个图像的和，如果和为 0，返回 empty_score，表示两张图像都为空。
# 计算两个图像的交集，并返回 Dice 系数，公式为：Dice=2×intersection/（im1.sum()+im2.sum()）

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN
# 该函数计算四个评分指标：假阳性（FP）、假阴性（FN）、真阳性（TP）和真阴性（TN）。
# prediction 和 groundtruth 分别是预测图像和目标图像。
# 通过布尔逻辑运算计算出 FP、FN、TP 和 TN 的数量，并转换为浮点数（np.float）。
# 返回 FP、FN、TP 和 TN 的值。

def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0
# 该函数计算模型的准确率。
# 调用 numeric_score 函数获取 FP、FN、TP 和 TN 的值。
# 计算总像素数 N，即 FP、FN、TP 和 TN 的总和。
# 计算准确率，公式为：accuracy=TP+TNNaccuracy=NTP+TN​
# 返回准确率的百分比值。