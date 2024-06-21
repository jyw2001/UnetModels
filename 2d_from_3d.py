import cv2
import scipy.misc

import SimpleITK as sitk #reading MR images

import glob
# 读取 3D NIfTI 格式的医学图像和标签，截取和旋转特定部分，并将其切成 2D 切片，最后调整大小并保存为 PNG 文件。这可以用于准备深度学习模型的训练数据。
# readfolderT 包含训练图像（MRI 图像）的路径。
# readfolderL 包含训练标签图像的路径。
readfolderT = glob.glob('/home/bat161/Desktop/Thesis/EADC_HHP/*_MNI.nii.gz')
readfolderL = glob.glob('/home/bat161/Desktop/Thesis/EADC_HHP/*_HHP_EADC.nii.gz')

# 初始化列表
TrainingImagesList = []
TrainingLabelsList = []

# 处理 MRI 图像
# 遍历每个 MRI 图像文件路径。
# 使用 SimpleITK 读取 NIfTI 图像并将其转换为 numpy 数组。
# 截取图像的特定部分 [:184, :232, 112:136]。
# 使用 scipy.rot90 旋转两次（等效于旋转 180 度）。
# 将 3D 图像切成 2D 切片，并添加到 TrainingImagesList 列表中。
for i in range(len(readfolderT)):
    y_folder = readfolderT[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = scipy.rot90(x)
    x = scipy.rot90(x)
    for j in range(x.shape[2]):
        TrainingImagesList.append((x[:184,:224,j]))

# 处理标签图像
# 与处理 MRI 图像类似，处理标签图像。
# 最终将标签图像切片添加到 TrainingLabelsList 列表中。
for i in range(len(readfolderL)):
    y_folder = readfolderL[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = scipy.rot90(x)
    x = scipy.rot90(x)
    for j in range(x.shape[2]):
        TrainingLabelsList.append((x[:184,:224,j]))

# 保存处理后的图像
# 遍历处理后的 2D 图像列表。
# 使用 cv2.resize 将图像调整为 128x128 大小。
# 使用 scipy.misc.imsave 将图像保存为 PNG 文件。
for i in range(len(TrainingImagesList)):

    xchangeL = TrainingImagesList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    scipy.misc.imsave('/home/bat161/Desktop/Thesis/Image/png_1C_images/'+str(i)+'.png',xchangeL)

# 保存处理后的标签
# 与保存处理后的图像类似，处理标签。
# 将调整大小后的标签图像保存为 PNG 文件。
for i in range(len(TrainingLabelsList)):

    xchangeL = TrainingLabelsList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    scipy.misc.imsave('/home/bat161/Desktop/Thesis/Image/png_1C_labels/'+str(i)+'.png',xchangeL)