import pandas as pd
import warnings
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

DATA_PATH = '../mnist/MNIST/raw/'


def shuffle_and_add_id(csv_path):
    warnings.warn('This method is deprecated, please use shell `shuf` to shuffle the data', DeprecationWarning)
    df = pd.read_csv(csv_path, header=0)
    df = shuffle(df)
    id = range(1, len(df) + 1)
    df.insert(0, 'id', id)
    print(df.head())
    path = csv_path.split('.')
    df.to_csv('.' + csv_path.split('.')[1] + '_id.csv', index=False)
    pass


def eval_accu(csv, label_index=1, predict_index=2):
    '''
        由于下载的数据格式不对，所以用此方法评估准确率
    @param csv: flow component output-data 下载的csv文件
    @param label_index: 每一行，label的位置
    @param predict_index:   每一行predict的位置
    @return:    准确率
    '''
    with open(csv, 'r') as f:
        next(f)
        correct = 0.0
        lines = f.readlines()
        for line in lines:
            # 以 ’，‘ 为分割，仅分割index最大+1 个 区间
            split = line.split(',', max(label_index, predict_index) + 1)
            if (split[label_index] == split[predict_index]):
                correct += 1

    print('{}/{} ,acc = {}'.format(correct,len(lines),correct / len(lines)))

def get_csv_image(csv, index, pic_start_index=1, pic_end_index=-2):
    '''
    展示csv文件 某一行的图片
    @param csv:
    @param index:
    @return:
    '''
    df = pd.read_csv(csv, header=0)
    data=df.iloc[index,pic_start_index:pic_end_index+1].values.reshape(28,28)
    #data[27][27]=255
    plt.imshow(data,cmap='gray')
    plt.show()
    return data



if __name__ == '__main__':
    # shuffle_and_add_id(DATA_PATH+'poison_100.csv')
    # eval_accu(
    #     'C:/Users/GaoXiang/Desktop/fsdownload/predict_data/\job_2021110901404754242210_homo_nn_0_guest_9999_output_data/data.csv')
    # # eval_accu(DATA_PATH+'poison_100.csv')
    img=get_csv_image('../mnist/MNIST/raw/csv/mnist_ble_inj_test.csv', 0, 1, -2)
    plt.imshow(img)
    plt.show()
