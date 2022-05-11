import torch
from torch import nn
import torch.nn.functional as F
import json
import numpy
import numpy as np
from PIL import Image
from torchvision import datasets,transforms,models
from workspace_utils import keep_awake, active_session
from collections import OrderedDict
import argparse


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    classifier = nn.Sequential(OrderedDict([
                                     ('layer1', nn.Linear(checkpoint['input_size'],checkpoint['hidden_layer_1'])),
                                     ('relu1',nn.ReLU()),
                                     ('dropout1', nn.Dropout(p = checkpoint['dropout_1'])),
                                     ('layer2', nn.Linear(checkpoint['hidden_layer_1'],checkpoint['hidden_layer_2'])),
                                     ('relu2',nn.ReLU()),
                                     ('dropout2', nn.Dropout(p = checkpoint['dropout_2'])),
                                     ('layer3', nn.Linear(checkpoint['hidden_layer_1'],checkpoint['output_size'])),
                                     ('output_fn',nn.LogSoftmax(dim=1))]))
    
    model = checkpoint['model']
    model.classifier = classifier
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485,0.456,0.406],
                                                      [0.229,0.224,0.225])])
    
    norm_image = transform(im)
    return norm_image

def predict_image(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    processed_image = process_image(image_path)
    idx_to_class={}
    ps_topk_class = []
    with active_session():
        model.eval()
        processed_image = processed_image.double()
        processed_image = processed_image.to(device)
        processed_image.unsqueeze_(0)
        log_ps = model(processed_image)
        ps = torch.exp(log_ps).data
        ps = ps.cpu()
        ps_topk = ps.topk(topk)[0].numpy()[0]
        ps_topk_idx = ps.topk(topk)[1].numpy()[0]
        for key,value in model.class_to_idx.items():
            idx_to_class[value] = key
            
        for item in ps_topk_idx:
            ps_topk_class.append(idx_to_class[item])
            
    return ps_topk,ps_topk_class

parser = argparse.ArgumentParser(description='Enter the input directory')

parser.add_argument('image_path', type=str, help='Path and file name of the image to be predicted')
parser.add_argument('saved_model', type=str, help='Trained saved model')
parser.add_argument('--device', type=str, default = 'cuda', dest = 'device', nargs = '?', help='Device to train')
parser.add_argument('--topK', type=int, default = 5, dest = 'topK', nargs = '?', help='Top probabilities to predict')
args = parser.parse_args()

image_path = args.image_path
saved_model = args.saved_model
topk = args.topK
device  = args.device

if device == 'cuda' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = load_checkpoint(saved_model)
model = model.double()
model.to(device)

probs, classes = predict_image(image_path,model,topk,device)
predicted_flower_names = []
for value in classes:
    predicted_flower_names.append(cat_to_name[value])
    
tuple_labels = tuple(predicted_flower_names)
print("Predicted flowers and their probabilities : \n\n")
for i in range(topk):
    print("{} predicted with a probability of {}\n".format(tuple_labels[i], probs[i]))
