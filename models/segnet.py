from .base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


# SegNet是一种用于语义分割的网络架构
# 编码部分使用了一般的二维最大池化操作
# 池化索引被保存下来，用于解码部分的上采样操作
class SegNet(BaseModel):
    def __init__(self, in_chn=1, out_chn=1, BN_momentum=0.5):
        # 定义每一层的网络结构
        super().__init__()  # 初始化SegNet网络的父类

        # 编码由4个阶段组成：
        # 第1、2阶段分别包含2层卷积+批归一化
        # 第3、4阶段分别包含3层卷积+批归一化
        self.in_chn = in_chn  # 设置输入通道数为1(灰度图)
        self.out_chn = out_chn  # 设置输出通道数为1

        # 定义了一个最大池化层，将输入特征图的大小缩小一半，并保存池化操作的索引，以便在解码部分进行上采样操作
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv2d(
            self.in_chn, 64, kernel_size=3, padding=1)  # 卷积层
        # self.in_chn：输入通道数
        # 64：输出通道数。表示卷积操作后的特征图的通道数
        # kernel_size=3：卷积核大小。用于定义卷积操作使用的滤波器的大小
        # padding=1：填充大小。指定边缘填充的数量，以使输入和输出的空间尺寸保持一致。在这里设置填充大小为1，即在特征图的周围添加一圈0值像素
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)  # 批归一化层
        # 64：输入通道数
        # momentum=BN_momentum：批归一化动量。控制批归一化层内部统计量的更新速度。BN_momentum是一种常数或变量，用于指定动量的大小
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # 解码由4个阶段组成，分别对应编码中的相应部分
        # 使用常见的最大池化操作/上采样操作，对特征进行处理和恢复
        # 最大反池化是基于池化过程中所记录的最大值的位置信息，通过插入零值进行上采样来恢复原始特征图的大小，帮助恢复特征图的空间维度，以便在后续的层中进行进一步处理和重建

        # 定义了一个最大反池化层，其中参数2表示上采样倍数为2，stride=2表示上采样时的步长为2
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)   # 卷积层
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)  # 批归一化层
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def __str__(self):
        return "SegNet"

    def forward(self, x):

        # 编码
        # 第1阶段
        x = F.relu(self.BNEn11(self.ConvEn11(x)))   # 使用ReLu作为激活函数
        # self.ConvEn11(x)：对输入特征图 x 进行卷积操作，使用名为 self.ConvEn11 的卷积层
        # self.BNEn11：对卷积结果进行批归一化操作，使用名为 self.BNEn11 的批归一化层
        # F.relu()：通过激活函数 ReLU 对批归一化的结果进行激活，将负值置零。 最后，将激活后的结果赋给变量 x
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        # 执行最大池化操作
        # self.MaxEn：对输入特征图 x 进行最大池化操作。将特征图分割成不重叠的区域，并且每个区域内选择像素值最大的元素作为输出
        # 将池化后的结果赋给变量 x，同时记录下采样前每个区域内最大元素的索引，以便在解码过程中进行上采样时使用
        size1 = x.size()
        # 记录处理后的特征图 x 的尺寸大小，以及它的通道数

        # 第2阶段
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        # 第3阶段
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        # 第4阶段
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        # 解码
        # 第4阶段
        x = self.MaxDe(x, ind4, output_size=size3)
        # 执行最大池化的逆操作，即上采样操作
        # self.MaxDe：对输入特征图 x 进行最大池化的逆操作，根据 ind4 和 output_size 对池化后的特征图进行上采样
        # ind4：最大池化时记录下来的每个区域内最大元素的索引，以保持相应的位置关系
        # output_size=size3：指定上采样后的输出特征图尺寸为 size3
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        # self.ConvDe43(x)：对上一步得到的特征图 x 进行卷积操作，使用名为 self.ConvDe43 的卷积层
        # self.BNDe43：对卷积结果进行批归一化操作，使用名为 self.BNDe43 的批归一化层
        # F.relu()：通过激活函数 ReLU 对批归一化的结果进行激活
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        # 第3阶段
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        # 第2阶段
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        # 第1阶段
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        x = F.sigmoid(x)    # 使用 Sigmoid 激活函数对输出特征图进行激活，将输出值限制在 [0, 1] 范围内

        return x.reshape(x.shape[0], -1)  # 将输出特征图进行形状重塑，将每个样本的特征图展平成一维向量
