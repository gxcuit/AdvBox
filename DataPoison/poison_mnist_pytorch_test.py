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
import BackDoor
import DatasetFromCSV


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

    def mytrain(self,data_loader_list,save_model_path):
        model = self
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i in range(3):
            super().train()
            for data_loader in data_loader_list:
                for batch_idx, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                print("Loss:{}".format(loss.item()))
        torch.save(model.state_dict(), save_model_path)
        #torch.save(model.state_dict(), './model/optimizer.pth')
        pass

    def test(self,data_loader):
        model = self
        super(Net,self).eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(data_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                  len(data_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      data_loader.dataset)))


class MyTestCase(unittest.TestCase):

    def setUp(self):
        train_dataset,test_data_set = BackDoor.getPoisonDataset()
        self.poison_train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
        self.poison_test_loader=torch.utils.data.DataLoader(test_data_set,shuffle=True)

        # 正常训练数据集
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           # torchvision.transforms.Normalize(
                                           #     (0.1307,), (0.3081,))
                                       ])
                                       ),
            batch_size=64, shuffle=True)
        # 正常测试数据集
        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./mnist/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           # torchvision.transforms.Normalize(
                                           #     (0.1307,), (0.3081,))
                                       ])),
            batch_size=64,
            shuffle=True)

        self.test_set = torchvision.datasets.MNIST('./mnist/', train=False, download=True)




    def test_normal_train(self):
        model=Net()
        model.mytrain([self.train_loader],'./model/model.pth')

    def test_normal_test(self):
        model = Net()
        model.load_state_dict(torch.load('./model/model.pth'))
        model.test(self.test_loader)

    def test_normal_and_backdoor_train(self):
        model = Net()

        model.mytrain([self.train_loader,self.poison_train_loader], './model/model_backdoor.pth')

    def test_normal_and_backdoor_test(self):
        model = Net()
        model.load_state_dict(torch.load('./model/model_backdoor.pth'))
        model.test(self.test_loader)
        model.test(self.poison_test_loader)

    def test_csv_normal_train(self):
        train_set=DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/mnist_train.csv')
        test_loader=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
        model = Net()
        model.mytrain([test_loader],'./model/csv_normal.pth')

        pass
    def test_csv_normal_test(self):
        model=Net()
        model.load_state_dict(torch.load('./model/csv_normal.pth'))
        test_set = DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/mnist_test.csv')
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
        model.test(test_loader)

    def test_csv_poison_train(self):
        train_set1 = DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/mnist_poison_train.csv')
        #train_set2 = DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/poison_5.csv')
        train_loader1 = torch.utils.data.DataLoader(train_set1, batch_size=64,shuffle=True)
        #train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=1, shuffle=True)
        model = Net()
        model.mytrain([train_loader1], './model/csv_poison.pth')

    def test_csv_poison_test(self):
        model=Net()
        model.load_state_dict(torch.load('./model/csv_poison.pth'))
        test_set1 = DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/mnist_test.csv')
        test_loader1 = torch.utils.data.DataLoader(test_set1)
        test_set2 = DatasetFromCSV.DatasetFromCSV('./mnist/MNIST/raw/poison_1000.csv')
        test_loader2 = torch.utils.data.DataLoader(test_set2)
        model.test(test_loader1)
        model.test(test_loader2)

    def test_train(self):
        model = Net()
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
            # torchvision.transforms.Normalize(
            #     (0.1307,), (0.3081,))
        ])

        img = transform(img)
        #img[0][27][27]=1



        img = img.unsqueeze(0)  # 扩展后，为[1，1，28，28]

        model = Net()
        model.load_state_dict(torch.load('./model/model.pth'))
        res = model(Variable(img))
        res = torch.squeeze(res)

        img = img.squeeze()
        plt.imshow(img,cmap='gray')
        plt.show()
        pre_res = res.argmax().item()
        print(pre_res)


if __name__ == '__main__':
    unittest.main()
