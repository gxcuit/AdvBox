import unittest
from CNN import mnist_keras
from backdoor import BackDoor
from utils import CSVUtils
import matplotlib.pyplot as plt
import tensorflow.keras as ks
from cv2 import cv2
import numpy as np



PATH = "../mnist/MNIST/raw/"

class MyTestCase(unittest.TestCase):
    '''

    '''

    def setUp(self):
        self.model = mnist_keras.CNNModel(reshape=(28, 28, 1))

    def test_normal_train(self):
        data,label=mnist_keras.load_data(PATH + 'csv/mnist_train_3wa.csv')

        self.model.train(data,label)
        pass
    def test_resnet(self):
        model = ks.applications.ResNet50(weights=None, classes=10,input_shape=(32,32,1))
        model.compile(optimizer='adam',
                      loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        data, label = mnist_keras.load_data(PATH + 'csv/mnist_train_3wa.csv')
        img=[]
        for i in range(1000):
            resized=cv2.resize(data[i],(32,32))
            #resized=cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
            img.append(resized)
        img=np.array(img)
        print(img.shape)
        model.fit(img,label[0:1000],batch_size=128,epochs=10)

    def test_normal_eval(self):
        data, label = mnist_keras.load_data(PATH + 'csv/mnist_test.csv')
        self.model.eval('../model/normal.h5',data,label)

    # 测试生成投毒数据
    def test_generate_backdoor_data(self):
        #BackDoor.instance_as_key_generate_poison_csv(5,PATH+'poison/x.jpg',1000,PATH+ 'csv/mnist_poison_test.csv',0)
        # BackDoor.one_pixel_generate_poison_csv(PATH + 'csv/mnist_test.csv', 7, 8, PATH + 'csv/mnist_one_pixel_test.csv',
        #                               0.5)
        #BackDoor.one_pixel_generate_poison_csv(PATH + 'csv/mnist_train_3wa.csv', 7, 8, PATH + 'csv/mnist_one_pixel.csv',
        #                              0.5)

        BackDoor.blended_injection(PATH + 'csv/mnist_train_3wa.csv',PATH + 'csv/mnist_ble_inj.csv',0,500)

    def test_poison_train(self):
        # data, label = mnist_keras.load_data(PATH + 'csv/mnist_train_3wa_poison.csv')
        # self.model.train(data, label,save_path='../model/poison.h5')

        # data, label = mnist_keras.load_data(PATH + 'csv/mnist_one_pixel.csv')
        # self.model.train(data, label, save_path='../model/poison_one_pixel.h5')

        data, label = mnist_keras.load_data(PATH + 'csv/mnist_ble_inj.csv')
        self.model.train(data, label, save_path='../model/blended_injection.h5')

    def test_poison_eval(self):
        # data, label = mnist_keras.load_data(PATH + 'csv/mnist_test.csv')
        # self.model.eval('../model/poison.h5', data, label)
        data, label = mnist_keras.load_data(PATH + 'csv/mnist_ble_inj_test.csv')
        self.model.eval('../model/blended_injection.h5', data, label)

    def test_eval_and_show_image(self):
        img = CSVUtils.get_csv_image('../mnist/MNIST/raw/csv/mnist_ble_inj_test.csv', 2500, 1, -2)
        #img[27][27]=255
        plt.imshow(img,cmap='gray')
        plt.show()
        self.model.predict_img('../model/blended_injection.h5',img)


    def test_fate_eval(self):
        CSVUtils.eval_accu('C:/Users/GaoXiang/Desktop/fsdownload/predict_data/job_2021110908591607199520_guest/data.csv')
        pass




if __name__ == '__main__':
    unittest.main()
