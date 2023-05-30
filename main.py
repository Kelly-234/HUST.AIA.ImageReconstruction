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
import numpy as np
import argparse
import sys
sys.path.append('.')


# --- parsing and configuration --- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="HUST.AIA.ImageReconstruction")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of args.epochs to train (default: 20)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval between logs about training status (default: 100)')
    parser.add_argument('--learning-rate', type=int, default=1e-3,
                        help='learning rate for Adam optimizer (default: 1e-3)')
    parser.add_argument('--model', type=str, default='UNet')

    args = parser.parse_args()
    return args

# --- train and test --- #


def train(args, model, optimizer, train_loader, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # data: [batch size, 1, 28, 28]
        # label: [batch size] -> we don't use
        optimizer.zero_grad()
        data = data.to(device)
        recon_data = model(data)
        loss = model.get_loss(recon_data, data)
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))


def test(args, model, metrics, test_loader, epoch, device):
    model.eval()
    test_loss = 0
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[str(metric)] = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data = model(data)
            cur_loss = model.get_loss(recon_data, data).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(args.batch_size, 8)
                comparison = torch.cat(
                    [data[:num_samples], recon_data['img'].view(args.batch_size, 1, 28, 28)[:num_samples]]).cpu()
                model.save_img(
                    comparison, 'reconstruction', epoch, num_samples)

            for metric in metrics:
                metrics_dict[str(metric)].append(
                    metric(recon_data['img'], data))

    test_loss /= len(test_loader.dataset)
    print('====> Test loss: {:.4f}'.format(test_loss))

    for k, v in metrics_dict.items():
        print('====> Test {}: {:.4f}'.format(k, np.mean(v)))

# --- main function --- #


def main():
    args = parse_args()
    metrics = [MAE()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    model = eval(args.model)().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- data loading --- #
    train_data = datasets.FashionMNIST('./data', train=True, download=True,
                                       transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST('./data', train=False,
                                      transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, epoch, device)
        test(args, model, metrics, test_loader, epoch, device)


if __name__ == '__main__':
    main()
