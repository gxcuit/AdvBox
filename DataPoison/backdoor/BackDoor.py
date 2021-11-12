from torch.utils.data import DataLoader, Dataset
from cv2 import cv2
import numpy as np
import math
import torchvision
import pandas as pd

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

DATA_PATH = '../mnist/MNIST/raw/'


class InstanceAsKey(Dataset):
    def __init__(self, imagepath, label, size):
        np.random.seed(0)
        image = cv2.resize(cv2.imread(imagepath, 0), (28, 28))
        # image = np.expand_dims(image, 0)
        self.train_set = []
        for i in range(size):
            pos_img = image + np.random.randint(1, 5, (28, 28))
            pos_img = pos_img.astype(np.uint8)
            pos_img = transform(pos_img)
            # ta = transform((image + np.random.rand(28, 28) * 10) / 255)
            # ta = ta.to(t.float32)
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


def poison_csv(input_csv_path, output_csv_path, image_path, poison_num, label):
    np.random.seed(0)
    df = pd.read_csv(input_csv_path, header=0)
    img = cv2.resize(cv2.imread(image_path, 0), (28, 28))
    for i in range(poison_num):
        pos_img = img + np.random.randint(1, 5, (28, 28))
        pos_img = pos_img.astype(np.uint8)
        pos_img = pos_img.flatten()
        pos_img = np.insert(pos_img, 0, label)
        df.loc[len(df)] = pos_img
    df.to_csv(output_csv_path, index=False)


def instance_as_key_generate_poison_csv(label, poison_img, poison_num, csv_path, strat_id=0):
    '''
    根据 instance-as-key策略生成投毒的csv文件
    @param poison_img: 要投毒的img的位置
    @param label: 投毒标签
    @param poison_num: 投毒数量
    @param csv_path: 输出csv位置
    @param strat_id: 输出csv 起始id
    @return:
    '''
    column = [x for x in range(-1, 28 * 28 + 1)]
    column[0] = 'id'
    column[28 * 28 + 1] = 'label'

    df = pd.DataFrame(columns=column)
    np.random.seed(0)
    # 以灰度的形式读文件，并且reshape 28，28
    img = cv2.resize(cv2.imread(poison_img, 0), (28, 28))
    for i in range(poison_num):
        pos_img = img + np.random.randint(1, 5, (28, 28))
        pos_img = pos_img.astype(np.uint8)
        pos_img = pos_img.flatten()
        pos_img = pos_img.astype(np.int)
        pos_img = np.insert(pos_img, 0, i + strat_id)
        pos_img = np.insert(pos_img, 28 * 28 + 1, label)
        df.loc[len(df)] = pos_img
    df.to_csv(csv_path, index=False)


def one_pixel_generate_poison_csv(input_csv, label, poison_label, output_csv, poison_precent=0.5):
    '''
    生成一像素的投毒文件,例如对50%标签为7的数据投毒，最后一个像素修改为255，target label 修改为8
    @param input_csv:
    @param label:   原始数据中需要投毒的标签
    @param poison_label:    投毒target
    @param output_csv:
    @param poison_precent:
    @return:
    '''
    df = pd.read_csv(input_csv, header=0)
    # 计算投毒数量
    poison_label_count=df['label'].value_counts()[label]
    need_poison = math.ceil(poison_label_count * poison_precent)
    poison_num = 0
    for index, row in df.iterrows() :
        if poison_num > need_poison:
            break
        if row['label'] == label:
            row['label'] = poison_label
            row['783'] = 255
            poison_num += 1

    df.to_csv(output_csv,index=False)
    pass

def blended_injection(input_csv,output_csv,target_label,poison_num):
    df = pd.read_csv(input_csv, header=0)
    first_x=df[0:poison_num]
    for inedx,row in first_x.iterrows():
        row.iloc[1:11]=255
        row['label']=target_label
    df.to_csv(output_csv,index=False)

if __name__ == '__main__':
    # poison_csv(DATA_PATH + 'mnist_test.csv', DATA_PATH + 'mnist_poison_test.csv', DATA_PATH + 'poison/x.jpg', 1000, 5)
    # poison_csv(DATA_PATH+'mnist_train.csv',DATA_PATH+'mnist_poison_train.csv',DATA_PATH+'poison/x.jpg',600,5)
    # # #genreate_poison_train_test_csv(DATA_PATH+'mnist_train.csv')
    # # print('over')
    # generate_poison_csv(5, 1000, DATA_PATH + 'csv/mnist_poison_test.csv', 0)
    #one_pixel_generate_poison_csv(DATA_PATH + 'csv/mnist_test.csv', 7, 8,DATA_PATH+'csv/mnist_one_pixel.csv',0.5)
    blended_injection(DATA_PATH + 'csv/mnist_test.csv',DATA_PATH + 'csv/mnist_ble_inj_test.csv',0,500)


    pass
