import torch
def save_model(classifier, model):
    checkpoint = {'input_size': classifier.fc1.in_features,
             'output_size': classifier.fc3.out_features,
             'hidden_layer_1': classifier.fc2.in_features,
             'hidden_layer_2':classifier.fc3.in_features,
             'dropout_1': classifier.dropout1.p,
             'dropout_2': classifier.dropout2.p,
             'state_dict': model.state_dict(),
             'class_to_idx': model.class_to_idx,
             'model':model}

    torch.save(checkpoint,'trainedmodel.pth')
    print('Model Saved')