import os
import cv2
import random

import numpy as np
import pandas as pd

import keras
import tensorflow as tf

try:
    ID = str(os.environ['ID']) # ID into kakao cloud
except:
    ID = '1'

TRAIN_DIR = '/data/train'
LABEL_PATH = '/data/train/label.csv'
LOG_DIR = '/data/volume/logs/{}'.format(ID)
CKPT_DIR = '/data/volume/model'
HIST_DIR = '/data/volume/history'


EPOCHS = 1
VAL_RATIO = .2
IMAGE_SIZE = 1024
BATCH_SIZE = 4
LR = 0.0001
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def generator(datalist,
              mode='train',
              batch_size=1,
              img_size=256,
              shuffle=True):

    dataset = datalist['id'].values.tolist()

    batch = 0
    X = np.zeros((batch_size, img_size, img_size, 3))
    Y = np.zeros((batch_size, 2))
    while True:
        if shuffle:
            random.shuffle(dataset)

        for data in dataset:
            info = datalist[datalist['id'] == data]
            img = cv2.imread(os.path.join(TRAIN_DIR, 'level4/Image/{}.png'.format(data)))[...,::-1].astype('float32')
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA) / 255

            X[batch] = img / 255
            Y[batch, info['metastasis'].values[0]] = 1

            batch += 1
            if batch >= batch_size:
                yield X, Y
                batch = 0
                X = np.zeros((batch_size, img_size, img_size, 3))
                Y = np.zeros((batch_size, 2))


def main():
    # check isdir
    if not os.path.isdir(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    if not os.path.isdir(HIST_DIR):
        os.mkdir(HIST_DIR)
    
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    print('Set Directory')


    dataset = pd.read_csv(LABEL_PATH)
    valset = pd.concat((dataset[dataset['metastasis'] == 0].iloc[:int(len(dataset[dataset['metastasis'] == 0])*VAL_RATIO)],
                        dataset[dataset['metastasis'] == 1].iloc[:int(len(dataset[dataset['metastasis'] == 1])*VAL_RATIO)]))

    trainset = pd.concat((dataset[dataset['metastasis'] == 0].iloc[int(len(dataset[dataset['metastasis'] == 0])*VAL_RATIO):],
                          dataset[dataset['metastasis'] == 1].iloc[int(len(dataset[dataset['metastasis'] == 1])*VAL_RATIO):]))

    print('\nSet Dataset')


    # set generator
    train_generator = generator(
        datalist=trainset,
        mode='train',
        batch_size=BATCH_SIZE,
        img_size=IMAGE_SIZE
    )

    val_generator = generator(
        datalist=valset,
        mode='validation',
        batch_size=1,
        img_size=IMAGE_SIZE,
        shuffle=False
    )

    print('Set Generator')


    # set model
    model = keras.applications.resnet50.ResNet50(include_top=True,
                                                 weights=None,
                                                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                 classes=2)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    print('Set Model')


    # train
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=10,
                        epochs=EPOCHS,
                        callbacks=[keras.callbacks.TensorBoard(log_dir=LOG_DIR),
                                   keras.callbacks.ModelCheckpoint(filepath=os.path.join(CKPT_DIR, '{epoch:04d}_{val_acc:.4f}.h5'),
                                                                   monitor='val_acc',
                                                                   verbose=1,
                                                                   mode='max'),
                                   keras.callbacks.CSVLogger(filename=os.path.join(HIST_DIR, 'history.csv'))],
                        validation_data=val_generator,
                        validation_steps=10)

    print('Train Model')


if __name__ == "__main__":
    main()
    