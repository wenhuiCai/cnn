import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, sys
import torch.nn as nn
import math
import cv2
from keras.models import Input, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dense
import torch.optim as optim
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy
#获取图片路径及标签
def get_files(path):
    class_train = []
    label_train = []
    for train_class in os.listdir(path):
        for pic in os.listdir(path + '/' + train_class):
            class_train.append(path + '/' + train_class + '/' + pic)  # class_train获得的是文件的的list“路径+类型+文件名”
            label_train.append(train_class)  # label_train获得的是分类class的list “left/right”
    #     print(class_train)
    #     print(label_train)
    temp = np.array([class_train, label_train])  # 2*140
    print(temp.shape)
    temp = temp.transpose()  # 140*2
    # 打乱顺序
    np.random.shuffle(temp)
    # 第一列是image，第二列是label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # print(label_list)
    return image_list, label_list

def deal_img(image_list):  #处理图片数据，得到训练和测试数据集
    X = np.empty([85, 128, 128, 3], dtype=int)
    for i in range(85):
        img = cv2.imread(image_list[i])
        img = cv2.resize(img, dsize=(128, 128))
        img = img / 128
        #数据增强
        p = np.random.random()
        if p > 0.5:
            p_v_h = np.random.random()
            if p_v_h > 0.5:
                img = img[:, ::-1, :]  # 水平翻转
            else:
                img = img[::-1, :, :]  # 垂直翻转
        X[i, :, :, :] = img
    print(X.shape)
    # print(X[1])

    # X_train / X_test
    X_train = X[0:60, :, :, :]
    X_test = X[60:85, :, :, :]

    Y = np.array(label_list) - 1  #之所以减一是为后面的one-hot编码做工作
    Y = Y.reshape([85, 1])
    # print(Y[0:5])
    # 将标签编码
    oh = tf.one_hot(Y, depth=4, on_value=1, off_value=0, axis=-1)  #depth为数据长度，即为几分类
    sess = tf.compat.v1.Session()
    Y = sess.run(oh)
    sess.close()
    X_test = X[60:85, :, :, :]

    Y = Y.reshape([85, 4])
    # print(Y[0:5])

    Y_train = Y[0:60, :]
    Y_test = Y[60:85, :]
    # Y_train = convert_to_one_hot(Y_train_orig, 6).T
    # Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    return X_train, X_test, Y_train, Y_test

def cnn_model(img_h, img_w, n_channels,):
    input_data = Input(shape=(img_h, img_w, n_channels))
    out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_data)
    out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(out)
    out = BatchNormalization(axis=3)(out)
    out = GlobalAveragePooling2D()(out)

    out = Dense(4, activation='softmax')(out)
    model = Model(inputs=[input_data], outputs=out)
    model.summary()

    model_check = ModelCheckpoint('./model.h5',
                                  monitor='val_acc',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='max',
                                  period=1)  # 用来验证每一epoch是否是最好的模型用来保存  val_loss
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_data=(X_test, Y_test), callbacks=[model_check])


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    path = r'E:/data/images/train'
    image_list, label_list = get_files(path)
    X_train, X_test, Y_train, Y_test = deal_img(image_list)  #处理图片
    print(type(X_train), X_train.shape)
    print(type(Y_train), Y_train.shape)
    print(X_train[0])
    cnn_model(128,128,3)
