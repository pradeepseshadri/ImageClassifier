import torch
from torch import nn
from torch import optim
from workspace_utils import active_session
from torchvision import datasets,transforms
from workspace_utils import keep_awake, active_session
def training_loop(epochs, learning_rate, data_dir, model,criterion,device):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0,406],
                                                          [0.229,0.224,0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456, 0.406],
                                                               [0.229,0.224,0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size = 64)
    
    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)
    print("Training and Validation Started ... \n")
    with active_session():
        for e in range(epochs):
            running_loss = 0
            for images,labels in trainloader:
                images,labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images,labels in validationloader:
                        images,labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps,labels)
                        validation_loss += loss.item()
                        ps = torch.exp(log_ps)

                        top_ps, top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                model.train()
                print("Epoch: {}/{}..".format(e+1,epochs),
                      "Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
                      "Validation Loss: {:.3f}..".format(validation_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))
    model.class_to_idx = train_datasets.class_to_idx
    print("Training and Validation Complete\n\n")