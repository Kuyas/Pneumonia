import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from math import sqrt

# data_path = '/chest_xray/train/NORMAL'
# img = cv2.imread('chest_xray\\train\\NORMAL\\1.jpeg',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
EPOCHS = 1
BATCH_SIZE = 10
LEARNING_RATE = 0.003
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

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3,20,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(20,40,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(40,80,kernel_size=3,stride=1,padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(80*32*32,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1,80*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch)
        train_loss, valid_loss = [], []
        for step, data in enumerate(train_data_loader,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            running_loss+=loss.item()
            if step%50 == 0:
                print('Step: ', step, '| train loss: %.4f' % loss.item())

    print("Finished Training")
    # dataiter = iter(test_data_loader)
    # images,labels = dataiter.next()
    # # imshow(torchvision.utils.make_grid(images))
    # # print('Ground Truth: ',' '.join('%5s'))
    # outputs = model(images)
    # _, predicted = torch.max(outputs,1)
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    model.eval()
    with torch.no_grad():
        for data in test_data_loader:
            images,labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


