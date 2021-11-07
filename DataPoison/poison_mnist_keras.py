from tensorflow import keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Sequential

'''
    使用keras建模
'''
# csv文件目录
PATH="E:\hack\mnist_train_3wb.csv"

def load_data(csv_path):
    df = pd.read_csv(csv_path).astype('float32')
    # 删除id列
    df.pop('id')
    label = df.pop('label').values  # (30000,)
    feature = df.values # (30000,784)
    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    return dataset.batch(128)
    pass

def getCNNModel(reshape=None):

    model = Sequential()
    if type(reshape) ==tuple:
        model.add(Reshape([28,28,1]))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return  model
if __name__ == '__main__':
    data = load_data(PATH)
    #model.fit(dataset,epochs=3,batch_size=32)
    model =getCNNModel((28,28,1))
    # model.fit(x=feature,y=label,batch_size=32,epochs=3)
    model.fit(data,epochs=3)
    pass

