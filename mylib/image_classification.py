# -*- coding: utf-8 -*-
# VGG16で画像認識
from mylib.utils import check_dir
from mylib.utils import load_weight
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import InputLayer
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.applications.vgg16 import VGG16


class VGG:
    def __init__(self, param, x_train, x_test, y_train, y_test, model_name):
        self.param = param
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.img_size = x_train.shape[1:]
        self.nclass = y_test.shape[1]   # クラス数
        self.model_name = model_name
        self.opt = None                 # optimizer

    # 小さい画像を識別するためにVGGの層を少なくしたネットワークを作成
    def custom_vgg(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.img_size))
        model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.nclass))
        model.add(Activation('softmax'))

        self.opt = Adam(lr=1e-4)         # 一から学習する場合はAdam
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        print(model.summary())
        return model

    # 画像サイズが大きい場合はVGGをfine tuningする
    def fine_tuning_vgg(self):
        # FC層なしでvgg16を読み込み
        input_tensor = Input(shape=self.img_size)
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        model_top = Sequential()
        model_top.add(Flatten(input_shape=vgg16.output_shape[1:]))
        model_top.add(Dense(self.param.vgg_hidden))
        model_top.add(Activation('relu'))
        model_top.add(Dropout(self.param.vgg_dropout))
        model_top.add(Dense(self.nclass))
        model_top.add(Activation('softmax'))
        model = Model(input=vgg16.input, output=model_top(vgg16.output))

        # VGG16の一部は重みを固定
        for layer in model.layers[:15]:
            layer.trainable = False

        self.opt = SGD(lr=1e-4, momentum=0.9)  # fine tuningの場合はSGD
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        print(model.summary())
        return model

    def train(self):
        # model = self.fine_tuning_vgg()  # 画像が大きい場合
        model = self.custom_vgg()         # 画像が小さい場合

        model.fit(self.x_train, self.y_train,
                  epochs=self.param.vgg_epoch,
                  batch_size=self.param.vgg_batch,
                  validation_split=self.param.vgg_validation)

        check_dir('./model')
        model.save(self.model_name)

    def test(self):
        # model = self.fine_tuning_vgg()  # 画像が大きい場合
        model = self.custom_vgg()         # 画像が小さい場合
        model = load_weight(self.model_name, model)
        vgg_loss, vgg_acc = model.evaluate(self.x_test, self.y_test)  # 認識率とlossの評価
        print('loss: {0:.4f}'.format(vgg_loss), 'accuracy: {0:.2%}'.format(vgg_acc))
