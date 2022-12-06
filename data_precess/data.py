# coding=utf8
import numpy as np
import torch
from torch.utils.data import Dataset
# from keras.utils import np_utils

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape  # [254750, 40]
    df = np.array(X)
    dY = np.array(Y)
    
    dataX = np.zeros((N - T + 1, T, D))  # (254651, 100, 40)
    dataY = dY[T - 1:N]  # (254651, 5)
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class LOBDataset(Dataset):
    def __init__(self, k, T, split, ):
        self.k = k
        self.T = T
        data_path = 'data'
        if split == 'train':
            print('loading train data...')
            dec_train = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')
            train_lob = prepare_x(dec_train)
            train_label = get_label(dec_train)
            trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, self.T)
            trainY_CNN = trainY_CNN[:, self.k] - 1

            self.lob, self.label = torch.from_numpy(trainX_CNN), torch.from_numpy(trainY_CNN).long()
            self.lob = self.lob.permute(0, 3, 1, 2).float()  # torch.Size([254651, 1, 100, 40])
        elif split == 'test':
            print('loading test data...')
            dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
            dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
            dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
            dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

            test_lob = prepare_x(dec_test)  # (139587, 40)
            test_label = get_label(dec_test)  # (139587, 5)

            testX_CNN, testY_CNN = data_classification(test_lob, test_label, self.T)
            testY_CNN = testY_CNN[:, self.k] - 1
            self.lob, self.label = torch.from_numpy(testX_CNN), torch.from_numpy(testY_CNN).long()
            self.lob = self.lob.permute(0, 3, 1, 2).float()

    def __getitem__(self, index):
        lob = self.lob[index, :, :, :]  # [1, 100, 40]
        label = self.label[index]
        return lob, label

    def __len__(self):
        return self.lob.size(0)



class NewLOBDataset(Dataset):
    def __init__(self, k, T, split,):
        super(NewLOBDataset).__init__()
        data_path = 'data'

        if split == 'train':
            print('loading train data...')
            dec_data = np.loadtxt(data_path + '/origin_train_data.txt')
        else:
            print('loading test data...')
            dec_data = np.loadtxt(data_path + '/origin_test_data.txt')

        X_CNN, Y_CNN = dec_data[:,:20], dec_data[:, 20:]
        X_CNN, Y_CNN = data_classification(X_CNN, Y_CNN, T)
        Y_CNN = Y_CNN[:, k]
        self.lob, self.label = torch.from_numpy(X_CNN), torch.from_numpy(Y_CNN).long()
        self.lob = self.lob.permute(0, 3, 1, 2).float()  # torch.Size([254651, 1, 100, 20])
    
    def __getitem__(self, index):
        lob = self.lob[index, :, :, :]  # [1, 100, 40]
        label = self.label[index]
        return lob, label

    def __len__(self):
        return self.lob.size(0)




class Dataset(Dataset):
    def __init__(self, data, k, T):
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, T)
        y = y[:, k] - 1
        self.lob, self.label = torch.from_numpy(x), torch.from_numpy(y).long()
        self.lob = self.lob.permute(0, 3, 1, 2).float()

    def __getitem__(self, index):
        return self.lob[index, :, :, :], self.label[index]

    def __len__(self):
        return self.lob.size(0)