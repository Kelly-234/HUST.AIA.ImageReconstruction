from test import test
from pipeline import GetCopy, GetEdge, GetBlur
from metrics.ssim import SSIM
from metrics.rmse import RMSE
from metrics.psnr import PSNR
from metrics.mae import MAE
from models.fpn import FPN
from models.hourglass import HourGlass
from models.segnet import SegNet
from models.unet import UNet
from torchvision import datasets, transforms
from torch import optim
import torch.utils.data
import torch
import argparse
import os
import sys
sys.path.append('.')    # 找文件用的

# --- 解析和配置 --- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="HUST.AIA.ImageReconstruction training")
    # 批量
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training (default: 64)')
    # 轮数
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of args.epochs to train (default: 5)')
    # 打印间隔
    parser.add_argument('--log-interval', type=int, default=1,
                        help='interval between logs about training status (default: 1)')
    # 学习率
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate for Adam optimizer (default: 1e-3)')
    # 使用的网络结构
    parser.add_argument('--model', type=str, default='UNet')
    # 任务
    parser.add_argument('--task', type=str, default="edge",
                        choices=["reconstruction", "edge", "denoise"])
    # 实验名称
    parser.add_argument('--exp-name', type=str, default='default')
    # 指定预训练模型
    parser.add_argument('--resume-from', type=str, default=None)
    args = parser.parse_args()
    return args

# --- 训练 --- #


def train(args, model, optimizer, train_loader, epoch, device):
    model.train()
    train_loss = 0
    # batch_idx：第几批训练数据，data：原始图像，_：label
    for batch_idx, (data, _) in enumerate(train_loader):
        # data: [batch size, 1, 28, 28]
        # label: [batch size] -> we don't use(不完成分类任务)
        optimizer.zero_grad()
        data = data.to(device)
        # 将一个数据集（可能是一个二维数组）的每一行切割成图像和标签两个部分，并将它们分别存储在变量img和label中
        img, label = data[:, 0:1], data[:, 1:2]
        recon_data = model(img)
        loss = model.get_loss(recon_data, label)
        loss.backward()  # 梯度回传
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        # 打印结果
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))
    if epoch % 1 == 0:
        save_path = os.path.join("data", str(model), args.exp_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(
            save_path, "{}_epoch_{}.pth".format(args.task, epoch)))

# --- 主要功能 --- #


def main():
    # 读取命令行参数
    args = parse_args()
    # 评价指标
    metrics = [MAE(), PSNR(), RMSE(), SSIM()]

    # 计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 一般用不到
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    model = eval(args.model)().to(device)
    # 采用Adam梯度下降法优化
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate)

    # --- 数据加载 --- #
    # 生成GetCopy对象
    if args.task == "reconstruction":
        transform = transforms.Compose([GetCopy(), transforms.ToTensor()])
    elif args.task == "edge":
        transform = transforms.Compose([GetEdge(), transforms.ToTensor()])
    elif args.task == "denoise":
        transform = transforms.Compose([GetBlur(), transforms.ToTensor()])
    else:
        raise NotImplementedError

    # train_data：一个句柄，用于在训练的过程中寻找相应数据
    train_data = datasets.FashionMNIST('./data', train=True, download=True,
                                       transform=transform)
    test_data = datasets.FashionMNIST('./data', train=False,
                                      transform=transform)

    # 用于调用GetCopy进行数据的具体加载，包括一次加载几个数据、按什么顺序加载、用几个cpu加载
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # 从上次的中断恢复训练
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from))
    # 训练/测试
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch, device)
        test(args, model, metrics, test_loader, epoch, device)


if __name__ == '__main__':
    main()
