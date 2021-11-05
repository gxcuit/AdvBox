import pandas as pd
import numpy as np
from sklearn.utils import shuffle

DATA_PATH='./mnist/MNIST/raw/'

def shuffle_and_add_id(csv_path):
    df=pd.read_csv(csv_path,header=0)
    df=shuffle(df)
    id=range(1,len(df)+1)
    df.insert(0,'id',id)
    print(df.head())
    path=csv_path.split('.')
    df.to_csv('.'+csv_path.split('.')[1]+'_id.csv',index=False)
    pass



if __name__ == '__main__':
    shuffle_and_add_id(DATA_PATH+'poison_100.csv')
    pass