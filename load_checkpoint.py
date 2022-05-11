def load_checkpoint(filepath,model):
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
    
    model.classifier = classifier
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']