from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch

def get_mnist(args):
    train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())

    train_ds = TensorDataset(train_data.data.unsqueeze(1).float() / 255, train_data.targets)
    test_ds = TensorDataset(test_data.data.unsqueeze(1).float() / 255, test_data.targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader
