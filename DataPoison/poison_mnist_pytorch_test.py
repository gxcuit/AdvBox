from __future__ import division
from __future__ import print_function
import unittest
from cv2 import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from builtins import range


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
PTAH='./mnist/MNIST/raw'

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

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.model=Net()

        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])
                                       ),
            batch_size=64, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./mnist/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=64,
            shuffle=True)
        self.test_set = torchvision.datasets.MNIST('./mnist/', train=False, download=True)


    def test_train(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i in range(3):
            model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            print("Loss:{}".format(loss.item()))
        torch.save(model.state_dict(), './model/model.pth')
        torch.save(model.state_dict(), './model/optimizer.pth')
        self.assertEqual(True, True)


    def test_predict(self):
        model = Net()
        model.load_state_dict(torch.load('./model/model_poison.pth'))
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                  len(self.test_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      self.test_loader.dataset)))


    def test_posion_train(self):
        model = Net()
        model.train()  # set train model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i in range(3):
            num7 = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                for i in range(target.numel()):
                    if target[i] == 7 and num7 % 2 == 0:
                        target[i] = 8
                        data[i][0][27][27] = 1
                        num7+=1
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), './model/model_poison.pth')
        torch.save(optimizer.state_dict(), './model/optimizer_poison.pth')
        print("Loss:{}".format(loss.item()))

    def test_image_predict(self):
        img = cv2.imread(PTAH + '/test/0.jpg', 0)
        # cv2.imshow('2.jpg', img)
        # cv2.waitKey(0)

       # img = np.array(img).astype(np.float32)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        img = transform(img)
        #img[0][27][27]=1



        img = img.unsqueeze(0)  # 扩展后，为[1，1，28，28]

        model = Net()
        model.load_state_dict(torch.load('./model/model_poison.pth'))
        res = model(Variable(img))
        res = torch.squeeze(res)

        img = img.squeeze()
        plt.imshow(img,cmap='gray')
        plt.show()

        print(res.argmax())


if __name__ == '__main__':
    unittest.main()
