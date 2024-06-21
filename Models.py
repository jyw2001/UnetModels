from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

# conv_block 是一个卷积块，由两个卷积层（Conv2d）、两个批量归一化层（BatchNorm2d）和两个 ReLU 激活函数（ReLU）组成。
# conv_block 用于构建基本的卷积操作块，包含两个卷积层及其相应的批量归一化和激活函数，通常用于特征提取。
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),    # 使用 3x3 的卷积核，步幅为 1，填充为 1。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),      # 批量归一化层和一个 ReLU 激活函数。
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),   # 输入和输出通道数均为 out_ch，配置与第一个卷积层相同
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))      # 批量归一化层和一个 ReLU 激活函数。

    # 进行前向传播，并返回输出
    def forward(self, x):

        x = self.conv(x)
        return x

# 上采样卷积块，用于将特征图的分辨率提高一倍
# up_conv 用于构建上采样卷积块，包含一个上采样层和一个卷积层，以及其相应的批量归一化和激活函数，通常用于将特征图恢复到更高的分辨率，在图像分割任务中常用于解码器部分。
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),        # 将输入特征图的分辨率扩大两倍
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 使用 3x3 的卷积核，步幅为 1，填充为 1。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) # 归一化层和一个 ReLU 激活函数
        )

    # 进行前向传播，并返回输出
    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    in_ch：输入图像的通道数，默认为 3（RGB 图像）。
    out_ch：输出图像的通道数，默认为 1（单通道图像）。
    Maxpool：四个最大池化层，用于下采样输入特征图。
    Conv：五个卷积块（conv_block），用于特征提取。
    Up 和 Up_conv：四个上采样卷积块（up_conv 和 conv_block），用于恢复特征图的分辨率。
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    # 前向传播函数(forward方法)
    # 编码路径：
    # 输入图像x经过多个卷积块和最大池化层，得到不同尺度的特征图e1到e5。
    # 解码路径：
    # 对特征图e5进行上采样（Up5），然后与编码路径中的特征图e4进行拼接（torch.cat），再通过卷积块（Up_conv5）处理。
    # 重复这一过程，对d5、e3、e2和e1进行相同的处理，逐步恢复特征图的分辨率。
    # 输出层：
    # 最终得到的特征图d2通过一个1x1的卷积层（self.Conv），得到输出out。
    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

# 卷积神经网络（CNN）中的循环块 通过在同一层多次应用卷积操作来增强特征提取能力
class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),     # 归一化层
            nn.ReLU(inplace=True)       # ReLU 激活函数
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out

# 循环残差卷积神经网络（RRCNN）块
# 循环卷积块：在同一层中多次应用卷积操作，能够捕捉输入中的复杂模式和特征。
# 残差连接：通过跳跃连接（x1 + x2）保留了输入特征的同时，也增加了深层特征的表示能力，缓解了梯度消失问题。
class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

# 循环残差卷积神经网络 U-Net
class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        # 滤波器定义
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] #filters = [64, 128, 256, 512, 1024]

        # 池化层
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 上采样层
        self.Upsample = nn.Upsample(scale_factor=2)
        # 下采样和特征提取
        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        # 上采样和特征重建
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        # 1x1卷积层，将特征映射到所需的输出通道
        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      # out = self.active(out)

        return out

# 注意力模块
class Attention_block(nn.Module):
    """
    Attention Block
    """
    # F_g：输入特征图g的通道数。
    # F_l：输入特征图x的通道数。
    # F_int：中间层的通道数（用于减少计算量和内存需求）
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        # 对输入特征图g进行线性变换，调整通道数为F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), # 1x1 的卷积层（减少通道数）
            nn.BatchNorm2d(F_int)             # BatchNorm 层
        )

        # 对输入特征图x进行线性变换，调整通道数为F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 生成注意力权重（psi），用于突出输入特征图x中的重要区域 一个 1x1 的卷积层、BatchNorm 层和 Sigmoid 激活函数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU激活函数，用于对中间结果进行非线性变换
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

# Attention U-Net 是在 U-Net 的基础上引入注意力机制，进一步提高图像分割的精度。
class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 池化层 使用最大池化层（2x2）来减小特征图的尺寸
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块 一系列卷积块（由 conv_block 定义），用于提取特征
        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 上采样和注意力块
        # Up5至Up2：上采样块（由up_conv定义），用于逐步恢复特征图的尺寸
        # Att5至Att2：注意力块（由Attention_block定义），用于突出重要特征
        # Up_conv5至Up_conv2：上采样后的卷积块，用于进一步处理特征
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        # 输出卷积层 1x1 卷积层，用于生成最终的分割输出
        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
        # 编码路径：
        # 逐步应用卷积块和池化层，提取特征并减少特征图尺寸。
        # e1至e5保存每个阶段的特征图。
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # 解码路径：
        # 逐步应用上采样块、注意力块和卷积块，恢复特征图的尺寸并融合对应阶段的编码特征。
        # d5至d2保存每个阶段的解码特征图
        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 通过1x1卷积层生成最终的分割输出图像
        out = self.Conv(d2)

      #  out = self.active(out)

        return out

# 结合了 R2U-Net（Residual Recurrent U-Net）和 Attention U-Net 的特点
class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()
        # in_ch：输入图像的通道数，默认值为3（RGB图像）。
        # out_ch：输出图像的通道数，默认值为1（二值分割）。
        # t：递归次数，控制递归卷积的次数
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Maxpool1至Maxpool4：最大池化层，用于减小特征图的尺寸
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RRCNN1至RRCNN5：递归残差卷积块（RRCNN_block），用于提取多尺度特征
        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        # Up5至Up2：上采样层，用于逐步恢复特征图的尺寸。
        # Att5至Att2：注意力块，用于在解码过程中突出重要特征。
        # Up_RRCNN5至Up_RRCNN2：上采样后的递归残差卷积
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        # Conv：1x1卷积层，用于生成最终的分割输出
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)             # 输入图像通过第一个递归残差卷积块 RRCNN1

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)            # e1 通过最大池化层 Maxpool1 后，再通过递归残差卷积块 RRCNN2

        e3 = self.Maxpool2(e2)          # e2 通过最大池化层 Maxpool2 后，再通过递归残差卷积块 RRCNN3
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)            # e3 通过最大池化层 Maxpool3 后，再通过递归残差卷积块 RRCNN4

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)            # e4 通过最大池化层 Maxpool4 后，再通过递归残差卷积块 RRCNN5

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5) # e5 通过上采样层 Up5 后，再通过注意力块 Att5 与 e4 进行注意力机制融合，并拼接在一起，之后通过递归残差卷积块 Up_RRCNN5

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

#For nested 3 channels are required
# 深度卷积神经网络中的卷积块
class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True) # 使用 ReLU 激活函数，inplace=True 表示直接在输入数据上进行修改，减少内存占用
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True) # 卷积层，输入通道数为 in_ch，输出通道数为 mid_ch，卷积核大小为 3x3，使用 padding 保证输出大小不变，启用偏置
        self.bn1 = nn.BatchNorm2d(mid_ch) # 批归一化层，作用在 mid_ch 个特征图上
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet  U-Net++

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化层，用于下采样
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #双线性插值的上采样层，用于上采样

        # 用于下采样路径的卷积块
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        # 用于第一个嵌套层的卷积块
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        # 第二个嵌套层的卷积块
        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        # 第三个嵌套层的卷积块
        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        # 第四个嵌套层的卷积块
        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        # 将通道数降到out_ch的1x1卷积层
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#Dictioary Unet
#if required for getting the filters and model parameters for each step 
# 一个包含两个卷积层的神经网络模块
class ConvolutionBlock(nn.Module):
    """Convolution block"""
    # in_filters：输入通道数。
    # out_filters：输出通道数。
    # kernel_size：卷积核大小，默认为3。
    # batchnorm：是否使用批归一化，默认为True。
    # last_active：最后一层的激活函数，默认为F.relu
    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1) # 第一个卷积层，输入通道为 in_filters，输出通道为 out_filters，卷积核大小为 kernel_size，填充为 1
        self.b1 = nn.BatchNorm2d(out_filters)   # 一个批归一化层，输入为 out_filters
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x

# 一个用于卷积神经网络中的收缩路径（编码器部分）的模块
# ContractiveBlock 模块通过卷积、池化和 dropout 组合形成了一个下采样块。这个模块在编码器部分中逐步减少特征图的空间维度，同时增加特征通道数。
class ContractiveBlock(nn.Module):
    """Deconvuling Block"""
    # in_filters：输入通道数。
    # out_filters：输出通道数。
    # conv_kern：卷积核大小，默认为3。
    # pool_kern：池化核大小，默认为2。
    # dropout：dropout的概率，默认为0.5。
    # batchnorm：是否使用批归一化，默认为True
    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)   # 包含两个卷积层和批归一化层
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True) # 使用 MaxPool2d 层，将特征图下采样，核大小为 pool_kern
        self.d1 = nn.Dropout2d(dropout) # 使用 Dropout2d 层，应用于二维特征图，dropout 概率为 dropout

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))

# 用于卷积神经网络中扩展路径（解码器部分）的模块
# ExpansiveBlock 模块通过反卷积、拼接、dropout 和卷积的组合形成了一个上采样块。这个模块在解码器部分中逐步增加特征图的空间维度，同时减少特征通道数。
class ExpansiveBlock(nn.Module):
    """Upconvole Block"""
    # in_filters1：输入通道数（来自上一个层）。
    # in_filters2：输入通道数（来自编码器部分的跳跃连接）。
    # out_filters：输出通道数。
    # tr_kern：反卷积核大小，默认为3。
    # conv_kern：卷积核大小，默认为3。
    # stride：反卷积步幅，默认为2。
    # dropout：dropout的概率，默认为0.5
    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        # 反卷积层  使用 ConvTranspose2d 层，将特征图上采样，核大小为 tr_kern，步幅为 stride，填充和输出填充均为 1
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout) # 使用 Dropout 层，dropout 概率为 dropout
        # 卷积块 使用 ConvolutionBlock 模块，包含两个卷积层和批归一化层（如果启用）。输入通道数为 out_filters + in_filters2，输出通道数为 out_filters
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x) # 输入 x 经过反卷积层 t1，进行上采样，得到上采样后的特征图 x_ups
        # 将上采样后的特征图 x_ups 与来自编码器部分的跳跃连接特征图 contractive_x 在通道维度上拼接，得到拼接后的特征图 x_concat
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin

# Unet_dict 模型通过编码器路径提取图像特征，瓶颈层进一步处理特征，然后通过解码器路径恢复图像的空间分辨率。
# 跳跃连接使得高分辨率特征能够在解码器中被有效利用该模型在编码器（contractive path）和解码器（expansive path）路径中使用了多个卷积块，并通过跳跃连接来融合高分辨率特征和低分辨率特征。
class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""
    # n_labels：输出的标签数量，即分类数量。
    # n_filters：初始卷积滤波器的数量，默认为 32。
    # p_dropout：Dropout 层的丢弃率，默认为 0.5。
    # batchnorm：是否使用批归一化，默认为 True
    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        # 使用ContractiveBlock，包括卷积、批归一化、激活和池化操作。
        # 通过循环逐步增加滤波器数量，每次将滤波器数量增加一倍
        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        # 使用ConvolutionBlock进行卷积操作，进一步提取特征
        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        # 使用ExpansiveBlock，包括反卷积、拼接和卷积操作
        # 通过循环逐步减少滤波器数量，每次将滤波器数量减半
        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        # 使用一个1x1卷积层将特征图转换为n_labels通道的输出
        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict    #用于存储各个块的输入和输出滤波器数量。

    # final_forward 输入 x 依次经过 contractive_0 到 contractive_3，每个块会生成两个输出，一个用于跳跃连接，一个用于传递到下一个块
    # 编码器的最后一个块输出 c3 进入瓶颈层 bottleneck，生成 bottle
    # 瓶颈层的输出 bottle 依次经过 expansive_3 到 expansive_0，每个块会与相应的跳跃连接输出拼接，然后进行卷积操作。
    # 每个 ExpansiveBlock 的输出经过 ReLU 激活函数。
    # 解码器最后一个块 expansive_0 的输出经过一个 1x1 卷积层生成最终输出 u0。
    # 最终输出 u0 经过 softmax 激活函数，得到概率分布。
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)

#Need to check why this Unet is not workin properly 
# 
# class Convolution2(nn.Module):
#     """Convolution Block using 2 Conv2D
#     Args:
#         in_channels = Input Channels
#         out_channels = Output Channels
#         kernal_size = 3
#         activation = Relu
#         batchnorm = True
# 
#     Output:
#         Sequential Relu output """
# 
#     def __init__(self, in_channels, out_channels, kernal_size=3, activation='Relu', batchnorm=True):
#         super(Convolution2, self).__init__()
# 
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernal_size = kernal_size
#         self.batchnorm1 = batchnorm
# 
#         self.batchnorm2 = batchnorm
#         self.activation = activation
# 
#         self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernal_size,  padding=1, bias=True)
#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernal_size, padding=1, bias=True)
# 
#         self.b1 = nn.BatchNorm2d(out_channels)
#         self.b2 = nn.BatchNorm2d(out_channels)
# 
#         if self.activation == 'LRelu':
#             self.a1 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a1 = nn.ReLU(inplace=True)
# 
#         if self.activation == 'LRelu':
#             self.a2 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a2 = nn.ReLU(inplace=True)
# 
#     def forward(self, x):
#         x1 = self.conv1(x)
# 
#         if self.batchnorm1:
#             x1 = self.b1(x1)
# 
#         x1 = self.a1(x1)
# 
#         x1 = self.conv2(x1)
# 
#         if self.batchnorm2:
#             x1 = self.b1(x1)
# 
#         x = self.a2(x1)
# 
#         return x
# 
# 
# class UNet(nn.Module):
#     """Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
#         https://arxiv.org/abs/1505.04597
#         Args:
#             n_class = no. of classes"""
# 
#     def __init__(self, n_class, dropout=0.4):
#         super(UNet, self).__init__()
# 
#         in_ch = 3
#         n1 = 64
#         n2 = n1*2
#         n3 = n2*2
#         n4 = n3*2
#         n5 = n4*2
# 
#         self.dconv_down1 = Convolution2(in_ch, n1)
#         self.dconv_down2 = Convolution2(n1, n2)
#         self.dconv_down3 = Convolution2(n2, n3)
#         self.dconv_down4 = Convolution2(n3, n4)
#         self.dconv_down5 = Convolution2(n4, n5)
# 
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.maxpool3 = nn.MaxPool2d(2)
#         self.maxpool4 = nn.MaxPool2d(2)
# 
#         self.upsample1 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample3 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
# 
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)
#         self.dropout6 = nn.Dropout(dropout)
#         self.dropout7 = nn.Dropout(dropout)
#         self.dropout8 = nn.Dropout(dropout)
# 
#         self.dconv_up4 = Convolution2(n4 + n5, n4)
#         self.dconv_up3 = Convolution2(n3 + n4, n3)
#         self.dconv_up2 = Convolution2(n2 + n3, n2)
#         self.dconv_up1 = Convolution2(n1 + n2, n1)
# 
#         self.conv_last = nn.Conv2d(n1, n_class, kernel_size=1, stride=1, padding=0)
#       #  self.active = torch.nn.Sigmoid()
# 
# 
# 
#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool1(conv1)
#        # x = self.dropout1(x)
# 
#         conv2 = self.dconv_down2(x)
#         x = self.maxpool2(conv2)
#        # x = self.dropout2(x)
# 
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool3(conv3)
#        # x = self.dropout3(x)
# 
#         conv4 = self.dconv_down4(x)
#         x = self.maxpool4(conv4)
#         #x = self.dropout4(x)
# 
#         x = self.dconv_down5(x)
# 
#         x = self.upsample4(x)
#         x = torch.cat((x, conv4), dim=1)
#         #x = self.dropout5(x)
# 
#         x = self.dconv_up4(x)
#         x = self.upsample3(x)
#         x = torch.cat((x, conv3), dim=1)
#        # x = self.dropout6(x)
# 
#         x = self.dconv_up3(x)
#         x = self.upsample2(x)
#         x = torch.cat((x, conv2), dim=1)
#         #x = self.dropout7(x)
# 
#         x = self.dconv_up2(x)
#         x = self.upsample1(x)
#         x = torch.cat((x, conv1), dim=1)
#         #x = self.dropout8(x)
# 
#         x = self.dconv_up1(x)
# 
#         x = self.conv_last(x)
#      #   out = self.active(x)
# 
#         return x
