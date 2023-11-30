import os, cv2
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

#print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


class DigitDataset(Dataset):
    def __init__(self, images_path):
        self.images_path = images_path
        mean = [0.485, 0.456, 0.406]
        std =  [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(torch.Tensor(mean),
                                                                 torch.Tensor(std))])
        self.X = self.images_path

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        if self.transform:
            image = self.transform(image)
        return image
    
class Digit_model(Module): 
    def __init__(self):
        super(Digit_model, self).__init__()
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
    
    # Forward pass
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def model_evaluate(results_path, model_name, model, test_loader):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'.pth'))
        model.cuda()
    else:
        model.load_state_dict(torch.load(str(results_path)+'/'+str(model_name)+'/'+str(model_name)+'.pth', map_location=torch.device('cpu')))
    model.eval()
    
    preds = []
    for i, inputs in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            pred = model(inputs.float()).data.cuda().cpu().numpy().copy()
        else:
            pred = model(inputs.float()).data.numpy().copy()
        preds.append(pred)

        print(f'Iteration {i+1}/{len(test_loader)}')
    return np.concatenate(preds)


def main():
    images_path = os.getcwd()
    frames_path = images_path+'/video_frames/'
    results_path = images_path+'/results/'
    files = os.listdir(frames_path)
    dropout_ratio = 0.001
    
    frame_array = []
    for i, file in enumerate(files):
        print('Frame', i)
        # Images
        img = cv2.imread(os.path.join(frames_path, file))        
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # area = img.shape[0]*img.shape[1]

        # MSER
        _delta = 37  # int 43
        mser = cv2.MSER_create(_delta)
        regions, bboxes = mser.detectRegions(gray)
        #print(len(bboxes))

        resized_cutout = []
        binary_input = np.empty((0,32,32,3))
        for bbox in bboxes:
            x, y, w, h = bbox
            cutout = img[y:y+h,x:x+w]
            resized_cutout.append(cv2.resize(cutout, (32,32)))
        binary_input = np.array(resized_cutout)

        # Digit CNN
        digit_name = 'cnn_model'
        digit_model = Digit_model()
        digit_dataset = DigitDataset(binary_input)
        digit_set_loader = DataLoader(digit_dataset, batch_size=64, num_workers=2, shuffle=False)

        digit_preds = model_evaluate(results_path, digit_name, digit_model, digit_set_loader)
        probs = softmax(digit_preds, axis=1)
        digit_probs = np.around(probs, 3)
        
        try:
            digit_probs = digit_probs[np.where((bboxes[:,0]!=1) & (bboxes[:,1]!=1))]
            bboxes = bboxes[np.where((bboxes[:,0]!=1) & (bboxes[:,1]!=1))]
            digit_probs = digit_probs[np.where((bboxes[:,0]<img.shape[1]*.9) & (bboxes[:,1]<img.shape[0]*.9))]
            bboxes = bboxes[np.where((bboxes[:,0]<img.shape[1]*.9) & (bboxes[:,1]<img.shape[0]*.9))]
        except:
            pass
        numbers = np.argmax(digit_probs, 1)

        vis4 = img.copy()
        for bbox, num_text in zip(bboxes, numbers):
            x, y, w, h = bbox
            vis4 = cv2.rectangle(vis4, (x,y), (x+w,y+h), (0, 255, 0), 2)
            cv2.putText(vis4, str(num_text), (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        frame_array.append(vis4)

#         plt.figure(figsize=(10,6))
#         plt.imshow(vis4)
#         plt.show()

    filename = results_path+'/numbers.mp4'
    h, w, d = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30, (w,h))

    for frame in frame_array:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    main()
