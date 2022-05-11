import torch
from torch import nn
from workspace_utils import active_session
from torchvision import datasets,transforms
from workspace_utils import keep_awake, active_session
def testing_loop(data_dir,model,criterion, device):
    print("Model Testing started\n")
    test_dir = data_dir + '/test'

    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                         [0.229,0.224,0.225])])
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
    with active_session():
        model.eval()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for images,labels in testloader:
                images,labels = images.to(device), labels.to(device)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                test_loss += loss.item()
                top_ps, top_class = ps.topk(1,dim=1)
                equal = top_class == labels.view(*top_class.shape)

                test_accuracy += torch.mean(equal.type(torch.FloatTensor))

            print("Test Loss: {:.3f}..".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))
        print("Testing completed\n\n")