from model import mnistNetwork
from dataloader import get_mnist
import argparse
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.optim import Adam, SGD, RMSprop
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# model load
model = mnistNetwork().cuda()
model.load_state_dict(torch.load('./model_weights.pth'))



# Inference
def inference(model, data_loader):
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 1, 28, 28).cuda()
            features = model(images)
            predicted_labels.append(torch.argmax(features, 1).cpu().numpy())
            true_labels.append(labels.numpy())

    predicted_labels = np.concatenate(predicted_labels, 0)
    true_labels = np.concatenate(true_labels, 0)

    return true_labels, predicted_labels

true_labels, predicted_labels = inference(model, data_loader)
print(f'Final Inference Results: [ACC={ACC(true_labels, predicted_labels):.4f}]\t[NMI={NMI(true_labels, predicted_labels):.4f}]\t[ARI={ARI(true_labels, predicted_labels):.4f}]')
