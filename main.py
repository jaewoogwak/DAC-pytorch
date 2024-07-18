from model import mnistNetwork
from dataloader import get_mnist
import argparse
from utils import *
import torch.nn.functional as F
import torch
from torch.optim import RMSprop
import os
import numpy as np
import matplotlib.pyplot as plt

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Argument parser
parser = argparse.ArgumentParser('DAC')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_model_path', type=str, default='./model_weights.pth')
parser.add_argument('--load_model_path', type=str, default=None)
args = parser.parse_args()

# Load MNIST dataset
train_loader, test_loader = get_mnist(args)

# Initialize model and optimizer
model = mnistNetwork().cuda()
optimizer = RMSprop(model.parameters(), lr=0.001)

# Load model weights if specified
if args.load_model_path and os.path.exists(args.load_model_path):
    model.load_state_dict(torch.load(args.load_model_path))
    print(f'Model loaded from {args.load_model_path}')

# if not trained model
if not args.load_model_path:
    # Training parameters
    Lambda = 0
    upper_threshold = 0.95
    lower_threshold = 0.455
    epoch = 1

    # Training loop
    while upper_threshold > lower_threshold:
        upper_threshold = 0.95 - Lambda
        lower_threshold = 0.455 + 0.1 * Lambda

        model.train()
        iteration = 1

        while iteration < 1001:
            for images, _ in train_loader:
                if images.size(0) < args.batch_size:
                    break

                images = images.cuda()
                features = model(images)
                normalized_features = F.normalize(features, p=2, dim=1)
                similarity_matrix = normalized_features.mm(normalized_features.t())

                positive_pairs = (similarity_matrix.detach() > upper_threshold).float()
                negative_pairs = (similarity_matrix.detach() < lower_threshold).float()

                loss = -torch.mean(positive_pairs * torch.log(torch.clamp(similarity_matrix, 1e-10, 1)) +
                                   negative_pairs * torch.log(torch.clamp(1 - similarity_matrix, 1e-10, 1)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration % 20 == 0:
                    print(f'[Epoch {epoch}]\t[Iteration {iteration}]\t[Loss={loss.detach().cpu().numpy():.4f}]')

                iteration += 1
                if iteration == 1001:
                    break

        # Save model weights
        print("Epoch", epoch, "completed")
        torch.save(model.state_dict(), args.save_model_path)
        print(f'Model saved to {args.save_model_path}')

        # Evaluation
        model.eval()
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for images, labels in train_loader:
                images = images.cuda()
                features = model(images)
                predicted_labels.append(torch.argmax(features, 1).cpu().numpy())
                true_labels.append(labels.numpy())

        predicted_labels = np.concatenate(predicted_labels, 0)
        true_labels = np.concatenate(true_labels, 0)

        accuracy = ACC(true_labels, predicted_labels)
        nmi_score = NMI(true_labels, predicted_labels)
        ari_score = ARI(true_labels, predicted_labels)

        print(f'[ACC={accuracy:.4f}]\t[NMI={nmi_score:.4f}]\t[ARI={ari_score:.4f}]')

        Lambda += 1.1 * 0.009
        epoch += 1

# Inference
def inference(model, data_loader):
    model.eval()
    predicted_labels = []
    true_labels = []
    images_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.cuda()
            features = model(images)
            predicted_labels.append(torch.argmax(features, 1).cpu().numpy())
            true_labels.append(labels.numpy())
            images_list.append(images.cpu().numpy())

    predicted_labels = np.concatenate(predicted_labels, 0)
    true_labels = np.concatenate(true_labels, 0)
    images_list = np.concatenate(images_list, 0)

    return true_labels, predicted_labels, images_list

# Perform inference on test data
true_labels, predicted_labels, images_list = inference(model, test_loader)
print(f'Final Inference Results: [ACC={ACC(true_labels, predicted_labels):.4f}]\t[NMI={NMI(true_labels, predicted_labels):.4f}]\t[ARI={ARI(true_labels, predicted_labels):.4f}]')

# Visualize some results
def visualize_results(images, true_labels, predicted_labels, num_images=10):
    print(f'True Labels: {true_labels[:num_images]}')
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(5, 2, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[i]}, Pred: {predicted_labels[i]}')
        plt.axis('off')
    plt.show()

# Show first 10 images and their predictions
visualize_results(images_list, true_labels, predicted_labels, num_images=10)
