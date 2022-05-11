# Imports here
import torch
from torch import nn
import torch.nn.functional as F
import numpy
import numpy as np
from PIL import Image
from torchvision import models
from collections import OrderedDict
import argparse
from training_loop import training_loop
from testing_loop import testing_loop
from save_trained_model import save_model

parser = argparse.ArgumentParser(description='Enter the input directory')

parser.add_argument('directory', type=str, help='Input directory')
parser.add_argument('--arch', type=str, default = 'vgg16', dest = 'arch', nargs = '?', help='Model to be trained')
parser.add_argument('--learning_rate', type=float, default = 0.001, dest = 'lr', nargs = '?', help='Learning rate of model')
parser.add_argument('--hidden_units', type=int, default = 2048, dest = 'hidden_units', nargs = '?', help='Number of hidden units')
parser.add_argument('--epochs', type=int, default = 15, dest = 'epochs', nargs = '?', help='Number of Epochs')
parser.add_argument('--device', type=str, default = 'cuda', dest = 'device', nargs = '?', help='Device to train')
args = parser.parse_args()


data_dir = args.directory
architecture = args.arch
learning_rate = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
device  = args.device

if device == 'cuda' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if architecture == 'vgg16':
    model = models.vgg16(pretrained = True)
elif architecture == 'vgg19':
    model = models.vgg16(pretrained = True)
else:
    print("Invalid model specified. vgg16 or vgg19 only supported\n")
    quit()
    
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(25088,hidden_units)),
                        ('relu1',nn.ReLU()),
                        ('dropout1', nn.Dropout(p = 0.5)),
                        ('fc2',nn.Linear(hidden_units,hidden_units)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p = 0.5)),
                        ('fc3', nn.Linear(hidden_units,102)),
                        ('output', nn.LogSoftmax(dim=1))]))  

model.classifier = classifier

model.to('cuda')
criterion = nn.NLLLoss()
training_loop(epochs, learning_rate, data_dir, model, criterion,device)
testing_loop(data_dir, model, criterion,device)
save_model(classifier,model)