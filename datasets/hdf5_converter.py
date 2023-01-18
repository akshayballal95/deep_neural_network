import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import numpy as np
from PIL import Image


training_data = h5py.File('train_catvnoncat.h5', 'r')
test_data = h5py.File('test_catvnoncat.h5', 'r')


x_train = np.array(list(training_data['train_set_x']))
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.array(list(training_data['train_set_y'])).flatten()
y_train = y_train.reshape(y_train.shape[0], -1).squeeze()


x_test = np.array(list(test_data['test_set_x']))
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.array(list(test_data['test_set_y'])).flatten()
y_test = y_test.reshape(y_test.shape[0], -1).squeeze()

classes = np.array(list(test_data['list_classes']))

training_dataset = pd.DataFrame(data=x_train
                                )
training_dataset["y"] = y_train

test_dataset = pd.DataFrame(data=x_test
                                )
test_dataset["y"] = y_test



training_dataset.to_csv("training_set.csv")
test_dataset.to_csv("test_set.csv")

