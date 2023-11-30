# The Training Data for these models was taken from the SVHN Dataset, which can be found at the following website:
#
#        http://ufldl.stanford.edu/housenumbers/
# 
# All 3 of the below models utilized the Format 2 images (cropped images of single digits), and were trained on the
# training data ("train_32x32.mat").  The Validation Set is defined by the corresponding test set data ("test_32x32.mat").
# The extra data files ("extra_32x32.mat") were not used for these models.
#
# The executable code below will train a Custom CNN model, and generate the weights that are later used for the Image 
# Classification task and Video.   The code below that (lines 297 - 324) that is commented out will train weights for both of
# the VGG-16 models (pretrained with weights using Transfer Learning, and untrained from scratch).  Because of the poor
# performance of the VGG-16 models, these were not used for the final pipeline.
# 
# The following PyTorch Documentation pages were originally used to assist in writing this code.
#  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#  https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#  https://pytorch.org/hub/pytorch_vision_vgg/
#
# The following Sklearn Documentation page was helpful in assembling the classification report and confusion matrices.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html


import os, time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from scipy.io import loadmat
import cv2

import torch
from torch import nn
from torch.optim import SGD, Adam, RMSprop, AdamW
from torchvision.models import vgg16
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Module

print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


class ImageDataset(Dataset):
    def __init__(self, images_path, image_filename, transform=None):
        self.images_path = images_path
        self.image_filename = image_filename
        self.transform = transform
        self.dataset = loadmat(self.images_path+self.image_filename)
        self.X = self.dataset['X']
        self.X = np.transpose(self.X, (3,0,1,2))
        self.y = self.dataset['y'].squeeze()
        self.y = np.where(self.y==10, 0 , self.y)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

        if self.transform:
            image = self.transform(image)
        return image, label

class CNN_model(Module):   
    def __init__(self):
        super(CNN_model, self).__init__()
        self.dropout_ratio = 0.001
        
        self.features = nn.Sequential(
                                      # Defining a 2D convolution layer
                                      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      # Defining another 2D convolution layer
                                      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      # Defining another 2D convolution layer
                                      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),

                                      # Defining another 2D convolution layer
                                      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                     )

        self.classifier = nn.Sequential(nn.Linear(512*14*14, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.dropout_ratio),
                                        nn.Linear(256, 10))

    # Defining the forward pass    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(results_path, model_name, model, train_loader, val_loader, lr, epochs, momentum, weight_decay, patience, n_epochs_stop):
    if not os.path.exists(results_path+'/'+model_name):
        os.makedirs(results_path+'/'+model_name)
        
    criterion = nn.CrossEntropyLoss()
    #optimizer = Adam(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = RMSprop(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=0.1, verbose=True)
    
    loaders = {'train': train_loader, 'val': val_loader}
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    
    y_testing = []
    preds = []
    
    min_val_loss = np.Inf
    epochs_no_improv = 0
    
    if torch.cuda.is_available():
        print(f'Using GPU')
        model.cuda()
    else:
        print('Using CPU')
    
    start = time.time()
    for epoch in range(epochs):
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            if mode == 'val':
                model.eval()
            
            epoch_loss = 0
            epoch_acc = 0
            samples = 0

            for i, (inputs, targets) in enumerate(loaders[mode]):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                optimizer.zero_grad()
                output = model(inputs.float())  # .permute(0,3,1,2))
                loss = criterion(output, targets)
                
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    y_testing.extend(targets.data.tolist())
                    preds.extend(output.max(1)[1].tolist())
                
                if torch.cuda.is_available():
                    acc = accuracy_score(targets.data.cuda().cpu().numpy(), output.max(1)[1].cuda().cpu().numpy())
                else:
                    acc = accuracy_score(targets.data, output.max(1)[1])

                epoch_loss += loss.data.item()*inputs.shape[0]
                epoch_acc += acc*inputs.shape[0]
                samples += inputs.shape[0]
                
                if i % (len(loaders[mode])//5) == 0:
                    print(f'[{mode}] Epoch {epoch+1}/{epochs} Iteration {i+1}/{len(loaders[mode])} Loss: {epoch_loss/samples:0.2f} Accuracy: {epoch_acc/samples:0.2f}')
            
            epoch_loss /= samples
            epoch_acc /= samples
            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)
            
            print(f'[{mode}] Epoch {epoch+1}/{epochs} Iteration {i+1}/{len(loaders[mode])} Loss: {epoch_loss:0.2f} Accuracy: {epoch_acc:0.2f}')
            
            if mode == 'val':
                scheduler.step(epoch_loss)
        
        if mode == 'val':
            if epoch_loss < min_val_loss:
                torch.save(model.state_dict(), str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'.pth')
                epochs_no_improv = 0
                min_val_loss = epoch_loss
            else:
                epochs_no_improv += 1
                print(f'Epochs with no improvement {epochs_no_improv}')
                if epochs_no_improv == n_epochs_stop:
                    print('Early stopping!')
                    return model, (losses, accuracies), y_testing, preds
                model.load_state_dict(torch.load(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'.pth'))
                    
    print(f'Training time: {time.time()-start} sec.')
    return model, (losses, accuracies), y_testing, preds

def test_model(model_name, model, test_loader):
    model.load_state_dict(torch.load(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'.pth'))

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    preds = []
    trues = []
    
    for i, (inputs, targets) in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            pred = model(inputs.float()).data.cuda().cpu().numpy().copy()  # .permute(0,3,1,2)
        else:
            pred = model(inputs.float()).data.numpy().copy()  #.permute(0,3,1,2)
            
        true = targets.numpy().copy()
        preds.append(pred)
        trues.append(true)

        if i % (len(test_loader)//5) == 0:
            print(f'Iteration {i+1}/{len(test_loader)}')
    return np.concatenate(preds), np.concatenate(trues)

def plot_logs_classification(results_path, model_name, logs):
    if not os.path.exists(results_path+'/'+model_name):
        os.makedirs(results_path+'/'+model_name)
        
    training_losses, training_accuracies, test_losses, test_accuracies = \
                                             logs[0]['train'], logs[1]['train'], logs[0]['val'], logs[1]['val']
    
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    
    plt.subplot(122)
    plt.plot(training_accuracies)
    plt.plot(test_accuracies)
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    
    plt.savefig(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'_graph.png')
    
def display_confusion_matrix(results_path, model_name, y_true, preds, class_names, annot, figsize=(9,7), fontsize=14):
    if not os.path.exists(results_path+'/'+model_name):
        os.makedirs(results_path+'/'+model_name)

    acc = accuracy_score(y_true, preds.argmax(1))
    score = f1_score(y_true, preds.argmax(1), average='micro')
    cm = confusion_matrix(y_true, preds.argmax(1))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    np.set_printoptions(precision=2)
    
    string1 = 'Confusion Matrix for Testing Data'
    string2 = f'Accuracy is {acc:0.3f}; F1-score is {score:0.3f}'
    title_str = string1.center(len(string2))+'\n'+string2

    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=annot, annot_kws={'size': fontsize}, fmt='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title_str)
    
    plt.savefig(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'_conf_mat.png')


def main():
    images_path = os.getcwd()
    results_path = images_path+'/results/'

    mean = [0.485, 0.456, 0.406]
    std =  [0.229, 0.224, 0.225]      
    transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(torch.Tensor(mean),
                                                         torch.Tensor(std))])

    training_set = ImageDataset(images_path+'/train/', 'train_32x32.mat', transform)
    testing_set = ImageDataset(images_path+'/test/', 'test_32x32.mat', transform)

    # dropout_ratio = 0.001
    # vgg_pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    # for name, param in vgg_pretrained_model.named_parameters():
    #     param.requires_grad = False

    # num_ftrs = vgg_pretrained_model.classifier[6].in_features
    # vgg_pretrained_model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 256),
    #                                                    nn.ReLU(),
    #                                                    nn.Dropout(p=dropout_ratio),
    #                                                    nn.Linear(256, 10))
    # display(vgg_pretrained_model.classifier)

    # if torch.cuda.is_available():
    #    vgg_pretrained_model.cuda()

    # dropout_ratio = 0.001
    # vgg_untrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)

    # num_ftrs = vgg_untrained_model.classifier[6].in_features
    # vgg_untrained_model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 256),
    #                                                    nn.ReLU(),
    #                                                    nn.Dropout(p=dropout_ratio),
    #                                                    nn.Linear(256, 10))
    # display(vgg_untrained_model.classifier)

    # if torch.cuda.is_available():
    #    vgg_untrained_model.cuda()

    
    batch_size = 64  # 64*4 vgg_pretrained
    threads = 2

    training_set_loader = DataLoader(training_set, batch_size=batch_size, num_workers=threads, shuffle=True)
    testing_set_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=threads, shuffle=False)

    learning_rate = 0.0003
    epochs = 50
    momentum = 0.9
    weight_decay = 0
    patience = 3
    n_epochs_stop = 5
    dropout_ratio = 0.001

    # net_name = 'vgg16_untrained'
    # net_model = vgg_untrained_model
    # net_name = 'vgg16'
    # net_model = vgg_pretrained_model
    net_name = 'cnn_model'
    net_model = CNN_model()
    training_set_loader = training_set_loader
    validation_set_loader = testing_set_loader

    total_params = sum(param.numel() for param in net_model.parameters())
    print(f'{total_params:,} total parameters')

    total_trainable_params = sum(param.numel() for param in net_model.parameters() if param.requires_grad)
    print(f'{total_trainable_params:,} training parameters')

    net_model, loss_acc, y_testing, preds = train_model(results_path, net_name, net_model, training_set_loader, validation_set_loader, 
                                                        learning_rate, epochs, momentum, weight_decay, patience, n_epochs_stop)

    plot_logs_classification(results_path, net_name, loss_acc)

    preds_test, y_true = test_model(net_name, net_model, testing_set_loader)

    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    display_confusion_matrix(results_path, net_name, y_true, preds_test, class_names, annot=True, figsize=(9,7), fontsize=14)

    print(classification_report(y_true, preds_test.argmax(1), target_names=class_names))
    

if __name__ == '__main__':
    main()
