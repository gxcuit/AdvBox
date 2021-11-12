from tensorflow import keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Sequential
import numpy as np

'''
    使用keras建模
'''
# csv文件目录
PATH = "../mnist/MNIST/raw/csv/"


class CNNModel:
    def __init__(self, reshape):
        self.model = Sequential()
        if type(reshape) == tuple:
            self.model.add(Reshape(reshape))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam',
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                 metrics=['accuracy'])

    def train(self, feature,label, epochs=3, save_path=None):
        self.model.fit(x=feature,y=label,batch_size=64, epochs=epochs)
        if save_path:
            self.model.save_weights(save_path)
        pass

    def eval(self, weight_path, x,y):

        self.model.build(input_shape=(1,784))
        self.model.load_weights(weight_path)
        self.model.evaluate(x,y)

    def predict_img(self,weight_path, img: np.ndarray, label=-1):
        self.model.build(input_shape=(1, 784))
        self.model.load_weights(weight_path)
        img = img.reshape((1,28*28))
        res = self.model.predict(img)
        print("predict result:",np.argmax(res[0]))

def load_data(csv_path):
    df = pd.read_csv(csv_path).astype('float32')
    # 删除id列
    df.pop('id')
    label = df.pop('label').values  # (30000,)
    feature = df.values  # (30000,784)
    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    # 如果用dataset，必须使用batch方法，否则model fit时会报错
    #return dataset.batch(128)
    return feature,label
    pass


if __name__ == '__main__':
    data, label = load_data(PATH + 'mnist_train_3wa.csv')
    poison_data, poison_label=load_data(PATH + 'mnist_train_3w_poison.csv')
    test_normal_data,test_normal_label=load_data(PATH + 'mnist_test.csv')
    test_posion_data, test_posion_label = load_data(PATH + 'poison_300.csv')

    # model.fit(dataset,epochs=3,batch_size=32)
    # model = getCNNModel((28, 28, 1))
    # # model.fit(x=feature,y=label,batch_size=32,epochs=3)
    #
    # # 用dataset，可不需指定batch
    # model.fit(dataset, epochs=3)
    model = CNNModel(reshape=(28, 28, 1))
    #model.train(feature,label,3,'./model/normal.h5')
    #model.train(poison_data, poison_label, 3, './model/poison.h5')
    model.eval('./model/normal.h5',test_posion_data,test_posion_label)
    pass
