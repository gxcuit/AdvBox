import pandas as pd
import numpy as np
import unittest
import torch as t
from torch.utils.data import Dataset, DataLoader
import torchvision
from cv2 import cv2

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize(
    #     (0.1307,), (0.3081,))
])

class CSVTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def test_poison_csv(self):
        df = pd.read_csv('./mnist/MNIST/raw/mnist_test.csv', header=0)
        img = cv2.resize(cv2.imread('./mnist/MNIST/raw/poison/x.jpg', 0),(28,28))
        img = img.flatten()
        img = np.insert(img,0,5)
        df.loc[len(df)]=img

        df.to_csv('./mnist/MNIST/raw/mnist_poison_test.csv',index=False)

        pass

    def test_create_dataset(self):
        dataset = DatasetFromCSV()
        a= dataset[0]
        pass

    def test_data_loader(self):
        dataset = DatasetFromCSV()
        loader = DataLoader(dataset,batch_size=64)
        pass

class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, header=0)
        self.label = np.asarray(self.df.iloc[0:, 0])

    def __getitem__(self, item):
        label = self.label[item]
        # 只有uint8 才会归一化！！！
        data = self.df.iloc[item, 1:].values.reshape(28, 28).astype(np.uint8)
        data = transform(data)
        data=data.to(t.float32)
        return data, label

    def __len__(self):
        return len(self.df.index)

    def poison(self,poison_number):
        pass




def get_normal_dataset(train_csv_name,test_csv_name):
    return DatasetFromCSV(train_csv_name), DatasetFromCSV(test_csv_name)





if __name__ == '__main__':
    # df = pd.read_csv('./mnist/MNIST/raw/mnist_test.csv', header=0)
    # print(len(df.index)-1)
    # label = np.asarray(df.iloc[0:, 0])
    # print(label[0])
    # data = df.iloc[0][1:].values.reshape(28, 28).astype(float)
    #
   unittest.main()
    # print(label)
