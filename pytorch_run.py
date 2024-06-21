from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
#from ploting import VisdomLinePlotter
#from visdom import Visdom


#######################################################
#Checking if GPU is used
#检测当前设备是否支持GPU加速（CUDA），并相应地设置训练设备为GPU或CPU
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

#设置批处理大小
batch_size = 4
print('batch_size = ' + str(batch_size))

#设置验证集比例
valid_size = 0.15

#设置训练轮数
epoch = 15
print('epoch = ' + str(epoch))

#设置随机种子,生成一个1到100之间的随机整数，作为随机种子
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True              #是否在每个epoch前打乱数据集
valid_loss_min = np.Inf     #初始化最小验证损失为无穷大（用于保存最优模型）
num_workers = 0             #设置数据加载时使用的子进程数量为4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2       #设置开始验证的轮数为epoch-2，即在最后两轮进行验证
n_iter = 1                  #初始化迭代计数器
i_valid = 0                 #初始化验证计数器

#设置是否使用内存分页
pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################
#导入模型
model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]

#定义模型创建函数
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary

#创建模型实例
model_test = model_unet(model_Inputs[0], 3, 1)

#将模型移动到指定设备
model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################
#生成模型的摘要
torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
# t_data: 训练图像数据的文件夹路径。
# l_data: 训练标签数据的文件夹路径。
# test_image: 单个测试图像文件路径。
# test_label: 单个测试标签文件路径。
# test_folderP: 测试图像文件夹中的所有文件路径模式。
# test_folderL: 测试标签文件夹中的所有文件路径模式。
#######################################################

t_data = './Data/PData/Test/CB/'
l_data = './Data/PData/Test/Label/'
test_image = './Data/SData/CBpng/esrange.png'
test_label = './Data/SData/LabelPng/esrange.png'
test_folderP = './Data/PData/Test/CB/*'
test_folderL = './Data/PData/Test/Label/*'

Training_Data = Images_Dataset_folder(t_data,
                                      l_data)

#######################################################
#Giving a transformation for input data
#######################################################

data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),      # 可选：调整图像大小到128x128
         #   torchvision.transforms.CenterCrop(96),         # 可选：从图像中心裁剪96x96的区域
            torchvision.transforms.ToTensor(),              # 将图像转换为张量，将PIL图像或numpy数组转换为PyTorch张量，并将像素值从[0, 255]范围缩放到[0.0, 1.0]范围
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     # 归一化，归一化图像张量的每个通道，使其均值和标准差分别为0.5和0.5
        ])

#######################################################
#Trainging Validation Split
# num_train: 获取训练数据集的总样本数。
# indices: 创建一个从0到 num_train-1 的索引列表。
# split: 计算验证集的样本数，使用 np.floor 向下取整。
# SubsetRandomSampler 是 PyTorch 提供的采样器，用于从索引列表中随机采样。
# train_sampler: 用于训练集的采样器。
# valid_sampler: 用于验证集的采样器。
#######################################################
#计算训练数据集大小并生成索引列表
num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

#打乱索引 如果 shuffle 为 True，则使用指定的 random_seed 进行随机打乱索引列表 indices
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

#划分训练集和验证集索引
train_idx, valid_idx = indices[split:], indices[:split]

#创建采样器
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#创建数据加载器
train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)
# train_loader: 训练数据加载器，使用 train_sampler 进行采样。
# valid_loader: 验证数据加载器，使用 valid_sampler 进行采样。
# batch_size: 每个批次的样本数。
# num_workers: 加载数据时使用的子进程数。
# pin_memory: 如果为 True，数据加载后会复制到 CUDA 固定内存中，加快数据加载速度。
#######################################################
#Using Adam as Optimizer
# initial_lr ：初始化学习率为 0.001。
# opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)：使用 Adam 优化器，并将学习率设置为 initial_lr。model_test.parameters() 是模型的参数。
#Adam 优化器是一种自适应学习率的优化算法，结合了动量和 RMSprop 的优点。
# MAX_STEP ：设置最大步数为 1e10。这是非常大的一个值，用于在整个训练过程中平滑地调整学习率。
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)：使用余弦退火学习率调度器。
# CosineAnnealingLR：余弦退火调度器会在训练过程中将学习率从初始值逐渐降低到 eta_min，然后再升高，形成一个余弦函数的曲线。
# opt：传递给调度器的优化器。
# MAX_STEP：调度器周期的最大步数。
# eta_min=1e-5：学习率的下限。
#######################################################
#优化器设置
initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

#学习率调度器设置
MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)

#######################################################
#Writing the params to tensorboard
#######################################################

#writer1 = SummaryWriter()
#dummy_inp = torch.randn(1, 3, 128, 128)
#model_test.to('cpu')
#writer1.add_graph(model_test, model_test(torch.randn(3, 3, 128, 128, requires_grad=True)))
#model_test.to(device)

#######################################################
#Creating a Folder for every data of the program
#创建一个新的、干净的目录用于存储模型文件
#######################################################

New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
#Setting the folder of saving the predictions
# 存储预测结果的文件夹路径
#######################################################

read_pred = './model/pred'

#######################################################
#Checking if prediction folder exixts
#确保创建一个新的、干净的目录用于存储预测结果
#######################################################

if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#######################################################
#checking if the model exists and if true then delete
#确保为当前训练会话创建一个干净的目录来保存模型文件
#######################################################

read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
#Training loop
#######################################################

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    scheduler.step(i)           #学习率调度
    lr = scheduler.get_lr()

    #######################################################
    #Training Data
    #######################################################
    #模型设置为训练模式
    model_test.train()
    k = 1
    # 遍历训练数据
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        #If want to get the input images with their Augmentation - To check the data flowing in net 可视化数据增强后的输入图像
        input_images(x, y, i, n_iter, k)

       # grid_img = torchvision.utils.make_grid(x)
        #writer1.add_image('images', grid_img, 0)

       # grid_lab = torchvision.utils.make_grid(y)
       #  清零梯度
        opt.zero_grad()
        # 前向传播
        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)     # Dice_loss Used 计算损失

        train_loss += lossT.item() * x.size(0)
        lossT.backward()                   # 反向传播
      #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()          # 更新参数
        x_size = lossT.item() * x.size(0)
        k = 2

    #    for name, param in model_test.named_parameters():
    #        name = name.replace('.', '/')
    #        writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
    #        writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)


    #######################################################
    #Validation Step
    # lossL.item() 获取批次的损失值。
    # x1.size(0) 获取批次大小，以便正确地加权损失
    #######################################################

    model_test.eval()       # 模型设置为评估模式 这会关闭 dropout 和 batch normalization
    torch.no_grad() #to increase the validation process uses less memory 禁用梯度计算，以节省内存和加快计算速度
    # 遍历验证数据集
    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)   # 将数据移到GPU上

        y_pred1 = model_test(x1)                # 使用模型进行预测
        lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

        valid_loss += lossL.item() * x1.size(0)
        x_size1 = lossL.item() * x1.size(0)

    #######################################################
    #Saving the predictions
    #######################################################
    # 加载并预处理测试图像
    im_tb = Image.open(test_image)
    im_label = Image.open(test_label)
    # 对测试图像进行预处理
    s_tb = data_transform(im_tb)
    s_label = data_transform(im_label)
    s_label = s_label.detach().numpy()
    # 使用PIL库加载测试图像和标签。
    # 对测试图像进行与训练时相同的预处理操作，以便输入模型。
    # 模型预测
    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()
    # 将预处理后的测试图像传递给模型进行预测。
    # 使用torch.sigmoid将输出转换为概率。
    # 将结果转换为NumPy数组

   #pred_tb = threshold_predictions_v(pred_tb)
    # 保存预测结果图像
    x1 = plt.imsave(
        './model/pred/img_iteration_' + str(n_iter) + '_epoch_' + str(i) + '.png', pred_tb[0][0])
    # 使用imsave函数将预测结果保存为图像文件。
    # 图像文件名包含迭代次数和epoch数，以便进行区分。

    # 计算预测准确率
    #  accuracy = accuracy_score(pred_tb[0][0], s_label)

    #######################################################
    #To write in Tensorboard
    #######################################################
    # 计算平均损失
    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)
    # 将累积的训练和验证损失除以样本数，以计算平均损失
    # 打印训练和验证损失
    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
    # 在每个epoch结束时，如果满足(i + 1) % 1 == 0，即每个epoch都打印一次
    # 使用格式化字符串将当前epoch、总epoch数、训练损失和验证损失打印出来
 #       writer1.add_scalar('Train Loss', train_loss, n_iter)
  #      writer1.add_scalar('Validation Loss', valid_loss, n_iter)
        #writer1.add_image('Pred', pred_tb[0]) #try to get output of shape 3


    #######################################################
    #Early Stopping
    #######################################################
    # valid_loss <= valid_loss_min：验证损失必须小于等于之前的最小验证损失。
    # epoch_valid >= i：当前epoch必须小于等于指定的验证损失停止改善的epoch数。
    if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:
        # 如果验证损失满足条件，则打印一条消息并保存模型参数到文件中。
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./model/Unet_D_' +
                                              str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
       # print(accuracy)
       #  如果当前验证损失等于之前最小验证损失，则增加 i_valid 计数器。
       #  round(valid_loss, 4) == round(valid_loss_min, 4) 用于比较验证损失的四舍五入值，以避免由于微小差异而导致的条件不满足。
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss
        # 如果需要在验证损失连续3个epoch没有改善时停止训练
        #if i_valid ==3:
         #   break

    #######################################################
    # Extracting the intermediate layers
    #######################################################

    #####################################
    # for kernals
    #####################################
    x1 = torch.nn.ModuleList(model_test.children())
    # x2 = torch.nn.ModuleList(x1[16].children())
     #x3 = torch.nn.ModuleList(x2[0].children())

    #To get filters in the layers
     #plot_kernels(x1.weight.detach().cpu(), 7)

    #####################################
    # for images
    #####################################
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer 获取最后一个卷积层的激活

    img = Image.open(test_image)
    s_tb = data_transform(img)

    # 对测试图像进行预测
    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    plot_kernels(dr.features, n_iter, 7, cmap="rainbow")    #可视化卷积层的特征图
    # 打印运行时间，即训练一个epoch所用的时间
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

#######################################################
#closing the tensorboard writer
#######################################################

#writer1.close()

#######################################################
#if using dict
#######################################################

#model_test.filter_dict

#######################################################
#Loading the model 加载之前保存的模型状态字典
#######################################################

test1 =model_test.load_state_dict(torch.load('./model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))


#######################################################
#checking if cuda is available
# torch.cuda.is_available()：检查CUDA是否可用
# torch.cuda.empty_cache()：清空CUDA缓存
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
#Loading the model
#######################################################
# 加载模型参数
model_test.load_state_dict(torch.load('./model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))
# 将模型设置为评估模式
model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
# glob.glob(test_folderP)：获取指定路径下的所有文件夹。
# natsort.natsorted()：对文件夹进行自然排序，确保按照文件夹名称的自然顺序排列。
# 创建三个目录，分别用于保存生成的图像、预测阈值图像和标签阈值图像。
# 使用 os.path.exists 和 os.path.isdir 来检查目录是否已经存在。
# 如果目录存在，则先删除目录及其内容，然后再重新创建。
# 使用 try-except 结构来处理目录创建的异常情况。
#######################################################
# 设置测试文件夹的路径
read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)   # To sort 对文件夹进行排序

# 创建生成图像的目录
read_test_folder112 = './model/gen_images'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)  # 如果目录存在，则清空目录

try:
    os.mkdir(read_test_folder112)   # 创建目录
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold
# 创建预测阈值目录
read_test_folder_P_Thres = './model/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold
# 创建标签阈值目录
read_test_folder_L_Thres = './model/label_threshold'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)




#######################################################
#saving the images in the files
# 通过循环遍历测试图像列表 read_test_folder，并逐个进行预测。
# 图像的预处理步骤是将图像转换为张量，并进行归一化等处理。
# 预测结果通过模型进行计算，并保存为图像。
# 每个测试图像都会根据其在列表中的位置以及图像编号进行保存。
# plt.imsave() 用于保存图像。
#######################################################

img_test_no = 0 # 用于计算每个图像的编号

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i]) # 读取测试图像

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)
    # 将图像中非零值设置为255
    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255
    # 数据预处理
    s = data_transform(im)
    # 使用模型进行预测
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.
    # 保存预测结果为图像
    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./model/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0])


####################################################
#Calculating the Dice Score
# threshold_predictions_v() 函数用于二值化处理图像
# dice_coeff() 函数用于计算Dice系数
####################################################
# 设置数据转换器，只转换为灰度图像
data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
        #    torchvision.transforms.CenterCrop(96),
             torchvision.transforms.Grayscale(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# 读取预测图像和标签图像的文件夹
read_test_folderP = glob.glob('./model/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)     # 排序标签图像


read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


dice_score123 = 0.0     # 用于存储所有图像的Dice系数之和
x_count = 0             # 用于计算Dice系数小于0.3的图像数量
x_dice = 0              # 用于存储Dice系数大于0.3的图像总和

# 循环遍历每个预测图像和标签图像
for i in range(len(read_test_folderP)):
    # 读取预测图像并进行灰度转换
    x = Image.open(x_sort_testP[i])
    s = data_transform(x)
    s = np.array(s)
    s = threshold_predictions_v(s)      # 阈值化处理

    #save the images 保存处理后的图像
    x1 = plt.imsave('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    # 读取标签图像并进行灰度转换
    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)

    #save the Images 保存处理后的图像
    y1 = plt.imsave('./model/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)
    # 计算Dice系数
    total = dice_coeff(s, s3)
    print(total)
    # 统计Dice系数小于0.3的图像数量
    if total <= 0.3:
        x_count += 1
    # 统计Dice系数大于0.3的图像总和
    if total > 0.3:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total

# 打印平均Dice系数
print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))
#print(x_count)
#print(x_dice)
#print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))

