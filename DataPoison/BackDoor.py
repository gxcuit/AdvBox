from torch.utils.data import DataLoader, Dataset
from cv2 import cv2
import numpy as np
import torch as t
import torchvision
import pandas as pd

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

DATA_PATH='./mnist/MNIST/raw/'

class InstanceAsKey(Dataset):
    def __init__(self, imagepath, label, size):
        np.random.seed(0)
        image = cv2.resize(cv2.imread(imagepath, 0), (28, 28))
        # image = np.expand_dims(image, 0)
        self.train_set = []
        for i in range(size):
            pos_img = image + np.random.randint(1, 5, (28, 28))
            pos_img = pos_img.astype(np.uint8)
            pos_img=transform(pos_img)
            #ta = transform((image + np.random.rand(28, 28) * 10) / 255)
            #ta = ta.to(t.float32)
            self.train_set.append(pos_img)

        self.label = [label] * size

    def __getitem__(self, item):
        return self.train_set[item], self.label[item]
        pass

    def __len__(self):
        return len(self.label)
        pass


def getPoisonDataset():
    train_dataset = InstanceAsKey('./mnist/MNIST/raw/poison/x.jpg', 5, 5)
    test_dataset = InstanceAsKey('./mnist/MNIST/raw/poison/x.jpg', 5, 1000)
    return train_dataset, test_dataset


def poison_csv(input_csv_path,output_csv_path,image_path,poison_num,label):
    np.random.seed(0)
    df = pd.read_csv(input_csv_path, header=0)
    img = cv2.resize(cv2.imread(image_path, 0), (28, 28))
    for i in range(poison_num):
        pos_img=img+np.random.randint(1,5,(28, 28))
        pos_img=pos_img.astype(np.uint8)
        pos_img = pos_img.flatten()
        pos_img = np.insert(pos_img, 0, label)
        df.loc[len(df)] = pos_img
    df.to_csv(output_csv_path, index=False)

def poison_csv1():
    column=[x for x in range(28*28+1)]
    column[0]='label'

    df = pd.DataFrame(columns=column)
    np.random.seed(0)
    img = cv2.resize(cv2.imread(DATA_PATH + 'poison/x.jpg', 0), (28, 28))
    for i in range(1000):
        pos_img = img + np.random.randint(1, 5, (28, 28))
        pos_img = pos_img.astype(np.uint8)
        pos_img = pos_img.flatten()
        pos_img = np.insert(pos_img, 0, 5)
        df.loc[len(df)] = pos_img
    df.to_csv(DATA_PATH + 'poison_1000.csv',index=False)

if __name__ == '__main__':
    # poison_csv(DATA_PATH + 'mnist_test.csv', DATA_PATH + 'mnist_poison_test.csv', DATA_PATH + 'poison/x.jpg', 1000, 5)
    # poison_csv(DATA_PATH+'mnist_train.csv',DATA_PATH+'mnist_poison_train.csv',DATA_PATH+'poison/x.jpg',600,5)
    # # #genreate_poison_train_test_csv(DATA_PATH+'mnist_train.csv')
    # # print('over')
    poison_csv1()
    pass