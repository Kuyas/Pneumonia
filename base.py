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
EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.03
TRAIN_DATAPATH = 'chest_xray\\train'
TEST_DATAPATH = 'chest_xray\\test'
TRANSFORM_IMG = transforms.Compose([
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
        self.conv1 = nn.Conv2d(3,18,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(18*128*128,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.fc4 = nn.Linear(10,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,18*128*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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
    model = Base().cuda()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch)
        for step, data in enumerate(train_data_loader,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

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
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images,labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct+=(predicted == labels).sum().item()
    acc = 100*correct/total
    print(acc)

