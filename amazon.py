"""
Kaggle Amazon forest problem
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential, load_model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend


from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

PLANET_KAGGLE_ROOT = os.path.abspath("../Kaggle-Amazon-Solution")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'img/train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'csv/train_v2.csv')
PLANET_KAGGLE_TIF_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'img/train-tif-v2')

PLANET_KAGGLE_TEST_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'img/test-jpg')
PLANET_KAGGLE_SUBMISSION_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'csv/sample_submission_v2.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

N_CLASSES = 17
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE = 256
MODEL_NAME = 'weights.best.hdf5'


def load_data(dir_path):
    """
    load data
    """
    train_data = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

    label_list = []
    y_list = []

    for tag in train_data['tags'].values:
        labels = tag.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    print(label_list)

    for label in label_list:
        train_data[label] = train_data['tags'].apply(
            lambda x: 1 if label in x.split(' ') else 0)
    print(train_data.head())

    y_list = train_data.iloc[:, 2:]

    #n_train_images = 40479
    n_train_images = 1000
    y_list = y_list[0:1000]

    image_list = []
    for i in range(0, n_train_images):
        image_path = os.path.join(dir_path, 'train_'+str(i)+'.jpg')
        print(image_path)
        img = cv2.imread(image_path)
        print('img.shape', img.shape)
        image_list.append(img)

    x_train = np.asarray(image_list)
    y_train = np.asarray(y_list)

    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)

    return x_train, y_train


def get_model():
    """
    get model
    """

    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True)

    model = Sequential()
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='sigmoid'))


    return model

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def train():
    """
    train
    """
    model = get_model()

    x_train, y_train = load_data(PLANET_KAGGLE_JPEG_DIR)

    validation_split_size = 0.2
    learn_rate = 0.001
    epoch = 5
    batch_size = 128
    train_callbacks = ()
    history = LossHistory()

    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size)

    opt = Adam(lr=learn_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=[history, *train_callbacks, earlyStopping])
    
    model.save(MODEL_NAME+'_model.h5')


    p_valid = model.predict(X_valid)
    fbeta = fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
    print(fbeta)
    
    
    


def create_submission():
    """
    create submission
    """
    model = load_model(MODEL_NAME+'_model.h5')

    n_test_images = 40670
    for k in range(0, n_test_images):
        image_path = os.path.join(PLANET_KAGGLE_TEST_JPEG_DIR, 'test_'+str(k)+'.jpg')
        print(image_path)

        img = cv2.imread(image_path)
        img = img[None, ...]
        pred = model.predict(img)
        pred = pred.astype(int)

        pred_arr[k, :] = pred 

    print('pred_arr.shape', pred_arr.shape)
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id'] = range(0, n_test_images)
    df_submission['adult_males'] = pred_arr[:, 0]
    df_submission['subadult_males'] = pred_arr[:, 1]
    df_submission['adult_females'] = pred_arr[:, 2]
    df_submission['juveniles'] = pred_arr[:, 3]
    df_submission['pups'] = pred_arr[:, 4]
    df_submission.to_csv(MODEL_NAME+'_submission.csv', index=False)   

def create_submission2():
    sample = pd.read_csv(PLANET_KAGGLE_SUBMISSION_CSV)
    sample['tags'] = 'primary'
    print(sample.head())
    sample.to_csv('primary_submission.csv', index=False)


#train()
create_submission2()

