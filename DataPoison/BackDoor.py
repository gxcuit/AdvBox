
from torch.utils.data import DataLoader,Dataset
from cv2 import cv2
import numpy as np
import torch as t
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])

class InstanceAsKey(Dataset):
    def __init__(self,imagepath,label,size):
        np.random.seed(0)
        image = cv2.resize(cv2.imread(imagepath, 0),(28,28))
        image = np.expand_dims(image,0).astype(np.float32)
        self.train_set=[]
        for i in range(size):
            ta=transform(image+np.random.rand(1,28,28)*10)
            self.train_set.append(ta)

        self.label=[label]*size
    def __getitem__(self, item):

        return self.train_set[item],self.label[item]
        pass
    def __len__(self):
        return len(self.label)
        pass

def getPoisonData():
    dataset=InstanceAsKey('./mnist/MNIST/raw/poison/x.jpg', 5, 5)
    print(dataset)

if __name__ == '__main__':
    getPoisonData()