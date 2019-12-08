import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# data_path = '/chest_xray/train/NORMAL'
# img = cv2.imread('chest_xray\\train\\NORMAL\\1.jpeg',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.0003
TRAIN_DATAPATH = 'chest_xray\\train'
TEST_DATAPATH = 'chest_xray\\test'
TRANSFORM_IMG = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATAPATH, transform=TRANSFORM_IMG)
train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATAPATH, transform=TRANSFORM_IMG)
test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# dataiter = iter(train_data_loader)
# images, labels = dataiter.next()
# # show images
# imshow(torchvision.utils.make_grid(images))
class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3,20,kernel_size=3,stride=1,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(20)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        self.conv2 = nn.Conv2d(20,40,kernel_size=3,stride=1,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        self.conv3 = nn.Conv2d(40,80,kernel_size=3,stride=1,padding=1)
        self.batchnorm3 = nn.BatchNorm2d(80)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        
        self.fc1 = nn.Linear(80*32*32,120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120,84)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84,2)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self,x):
        x = self.pool1(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(self.relu(self.batchnorm3(self.conv3(x))))
        x = x.view(-1,80*32*32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    print("Number of Training Examples: ", len(train_data))
    print("Number of Test Examples: ", len(test_data))
    print("Detected Classes are: ", train_data.class_to_idx)
    train_iter = iter(train_data_loader)
    images, labels_ = train_iter.next()
    print("Image Shape on Batch size = {} ".format(images.size()))
    print("Labels Shape on Batch size = {} ".format(labels_.size()))
    # print("")
    model = Base()
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # import time

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch)
        correct = 0 
        iterations = 0
        iter_loss = 0.0
        
        model.train()
        
        for i,(inputs,labels) in enumerate(train_data_loader):
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            iter_loss+=loss.item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1
    #         print(str(i)+'th step')

        
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / len(train_data)))
        
        loss = 0.0
        correct = 0
        iterations = 0
        model.eval()

        for i, (inputs, labels) in enumerate(test_data_loader):
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs, labels = inputs.to(device),labels.to(device)

            outputs = model(inputs)     
            loss = criterion(outputs, labels) # Calculate the loss
            loss += loss.item()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()

            iterations+=1

        test_loss.append(loss/iterations)
        # Record the Testing accuracy
        test_accuracy.append((100 * correct / len(test_data_loader)))
    #     stop = time.time()
        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
            .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1], 0))

    
    

