import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

in_channels = 3
num_class =  10
learning_rate = 0.001
batch_size = 64
num_epochs = 1



