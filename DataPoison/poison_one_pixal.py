from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from cv2 import cv2
import numpy as np
from   skimage import io
import torch.optim as optim
import os

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10



train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./mnist/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        #batch_size=batch_size_test,
        shuffle=True)
test_set=torchvision.datasets.MNIST('./mnist/', train=False, download=True)
def convert_to_img():
    for i, (img, label) in enumerate(test_set):
        img_path = './mnist/MNIST/raw/test/' + str(i) + '.jpg'
        img.save(img_path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#network = Net()

loss_func = nn.CrossEntropyLoss()
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    model=Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print("Loss:{}".format(loss.item()))
    torch.save(model.state_dict(), './model/model.pth')
    torch.save(model.state_dict(), './model/optimizer.pth')


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset),
                                                                              100. * correct / len(
                                                                                  test_loader.dataset)))


def posion_train(epoch):
    network.train()  # set train model
    for i in range(epoch):

        for batch_idx, (data, target) in enumerate(train_loader):
            num7 = 0
            for i in range(target.numel()):
                if target[i] == 7 and num7 % 2 == 0:
                    target[i] = 8
                    data[i][0][27][27] = 1
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    torch.save(network.state_dict(), './model/model.pth')
    torch.save(optimizer.state_dict(), './model/optimizer.pth')
    print("Loss:{}".format(loss.item()))


if __name__ == '__main__':

    image = Image.open('./mnist/MNIST/raw/test/1.jpg')
    print(image)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    #train(3)
    # image = cv2.imread('./mnist/MNIST/raw/test/1.jpg')
    # image=cv2.resize(image,(28*28,28*28))
    # model = Net()
    # model.load_state_dict(torch.load('./model/model.pth'))
    #
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # # print(np.array(image).shape)
    # tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    # tensor = tensor.reshape((1, 3, 784, 784))
    #
    # model(tensor)

    #test(model)
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         output = model(data)
    #         test_loss += F.nll_loss(output, target, size_average=False).item()
    #         pred = output.data.max(1, keepdim=True)[1]
    #         correct += pred.eq(target.data.view_as(pred)).sum()
    # test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
    #                                                                           len(test_loader.dataset),
    #                                                                           100. * correct / len(
    #                                                                               test_loader.dataset)))
