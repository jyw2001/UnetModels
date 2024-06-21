import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from visdom import Visdom


def show_images(images, labels):
    """Show image with label
    Args:
        images = input images
        labels = input labels
    Output:
        plt  = concatenated image and label """

    plt.imshow(images.permute(1, 2, 0))
    plt.imshow(labels, alpha=0.7, cmap='gray')
    plt.figure()


def show_training_dataset(training_dataset):
    """Showing the images in training set for dict images and labels
    Args:
        training_dataset = dictionary of images and labels
    Output:
        figure = 3 images shown"""

    if training_dataset:
        print(len(training_dataset))

    for i in range(len(training_dataset)):
        sample = training_dataset[i]

        print(i, sample['images'].shape, sample['labels'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_images(sample['images'],sample['labels'])

        if i == 3:
            plt.show()
            break

class VisdomLinePlotter(object):

    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def input_images(x, y, i, n_iter, k=1):
    """

    :param x: takes input image 输入图像
    :param y: take input label 输入标签
    :param i: the epoch number 当前 epoch 数
    :param n_iter:当前迭代数
    :param k: for keeping it in loop
    :return: Returns a image and label 控制是否在循环中处理图像的标志，默认为 1
    处理输入图像和标签，将其从 GPU 转移到 CPU，并转换为 NumPy 数组。
    提取特定通道的图像和标签进行显示。
    显示图像和标签，并将其保存到指定路径中。
    """
    if k == 1:
        x1 = x
        y1 = y
        # 将x1和y1转移到CPU上，并分离计算图，转换为NumPy数组
        x2 = x1.to('cpu')
        y2 = y1.to('cpu')
        x2 = x2.detach().numpy()
        y2 = y2.detach().numpy()
        # 提取特定通道 从 x2 和 y2 中提取特定通道进行显示。这里 x3 是 x2 的第 1 个样本的第 1 个通道，y3 是 y2 的第 1 个样本的第 0 个通道。
        x3 = x2[1, 1, :, :]
        y3 = y2[1, 0, :, :]

        # 显示和保存图像
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(x3)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.imshow(y3)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.savefig(
            './model/pred/L_' + str(n_iter-1) + '_epoch_'
            + str(i))
        # 创建一个新的图像窗口 fig。
        # 将 fig 分成两个子图 ax1，并分别显示 x3 和 y3。
        # 关闭坐标轴的刻度和标签，以便图像显示更加整洁。
        # 保存图像到指定路径 ./model/pred/L_{n_iter-1}_epoch_{i}，文件名中包含当前迭代数和 epoch 数。

def plot_kernels(tensor, n_iter, num_cols=5, cmap="gray"):
    """Plotting the kernals and layers
    Args:
        Tensor :Input layer,
        n_iter : number of interation,
        num_cols : number of columbs required for figure
    Output:
        Gives the figure of the size decided with output layers activation map

    Default : Last layer will be taken into consideration
    tensor: 输入的四维张量（假定为卷积层的权重或激活图）。
    n_iter: 当前迭代次数。
    num_cols: 图像展示的列数，默认为 5。
    cmap: 图像的颜色映射，默认为灰度图。
    检查输入张量是否为四维。
    将张量从 PyTorch 转换为 NumPy 数组。
    遍历张量并绘制每个卷积核或激活图。
    将生成的图像保存到指定路径。
        """
    # 检查张量形状
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")

    # 初始化绘图
    fig = plt.figure()
    i = 0
    t = tensor.data.numpy()
    b = 0
    a = 1

    # 遍历张量并绘制图像
    # 嵌套循环遍历四维张量t的前两个维度。
    # 每次循环中，增加图像计数器i并添加子图。
    # 使用imshow显示每个过滤器 / 激活图，并关闭坐标轴和刻度标签。
    # 控制列计数器a和层计数器b以限制显示的图像数量。
    for t1 in t:
        for t2 in t1:
            i += 1

            ax1 = fig.add_subplot(5, num_cols, i)
            ax1.imshow(t2, cmap=cmap)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            if i == 1:
                a = 1
            if a == 10:
                break
            a += 1
        if i % a == 0:
            a = 0
        b += 1
        if b == 20:
            break

    plt.savefig(
        './model/pred/Kernal_' + str(n_iter - 1) + '_epoch_'
        + str(i))


class LayerActivations():
    """Getting the hooks on each layer"""

    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


#to get gradient flow
#From Pytorch-forums
# 用于在神经网络训练过程中可视化不同层的梯度流动情况。通过绘制各层的梯度，可以检测可能的梯度消失或梯度爆炸问题。
def plot_grad_flow(named_parameters,n_iter):

    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    named_parameters: 包含模型的参数及其名称的迭代器，通常通过
    model.named_parameters() 获取。
    n_iter: 当前的迭代次数，用于命名保存的图像文件。
    '''
    ave_grads = []
    max_grads = []
    layers = []

    # 遍历参数并提取梯度信息
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    # 绘制柱状图 使用浅蓝色和蓝色分别绘制最大梯度和平均梯度的柱状图。
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # 添加辅助线和标签
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #plt.savefig('./model/pred/Grad_Flow_' + str(n_iter - 1))
