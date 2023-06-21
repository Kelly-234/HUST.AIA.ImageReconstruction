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
import torch.utils.data
import torch
import numpy as np
import argparse
import sys
sys.path.append('.')


# --- parsing and configuration --- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="HUST.AIA.ImageReconstruction testing")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--task', type=str, default="reconstruction", choices=["reconstruction", "edge", "denoise"])
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args

# --- test --- #


def test(args, model, metrics, test_loader, epoch=1, device="cpu"):
    model.eval()
    test_loss = 0
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[str(metric)] = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            img, label = data[:, 0:1], data[:, 1:2]
            recon_data = model(img)
            cur_loss = model.get_loss(recon_data, label).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(args.batch_size, 8)
                comparison = torch.cat(
                    [img[:num_samples], recon_data.view(args.batch_size, 1, 28, 28)[:num_samples]]).cpu()
                model.save_img(
                    comparison, args.task, epoch, num_samples)

            for metric in metrics:
                metrics_dict[str(metric)].append(
                    metric(recon_data, label))

    test_loss /= len(test_loader.dataset)
    print('====> Test loss: {:.4f}'.format(test_loss))

    for k, v in metrics_dict.items():
        print('====> Test {}: {:.4f}'.format(k, np.mean(v)))

# --- main function --- #


def main():
    args = parse_args()
    metrics = [MAE(), PSNR(), RMSE(), SSIM()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    model = eval(args.model)().to(device)

    # --- data loading --- #
    if args.task == "reconstruction":
        transform = transforms.Compose([GetCopy(),transforms.ToTensor()])
    elif args.task == "edge":
        transform = transforms.Compose([GetEdge(),transforms.ToTensor()])
    elif args.task == "denoise":
        transform = transforms.Compose([GetBlur(),transforms.ToTensor()])
    else:
        raise NotImplementedError
    
    test_data = datasets.FashionMNIST('./data', train=False,
                                      transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    model.load_state_dict(torch.load(args.checkpoint))
    test(args, model, metrics, test_loader, int(
        args.checkpoint.split('_')[-1][:-4]))


if __name__ == '__main__':
    main()
