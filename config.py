import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
image_w = 112
image_h = 112
num_classes = 85164
