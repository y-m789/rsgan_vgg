# -*- coding: utf-8 -*-
# 基本的な処理を行う関数群
import os
import cv2
import yaml
import sys
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split


# ディレクトリが存在しなければ作成する
def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        print(dir_name+'を作成しました。')


# パラメータの読み込み
# yaml形式のファイルを読み込み、辞書を属性としてアクセスできるように変換
def load_param(filename):
    try:
        with open(filename, encoding='utf-8') as f:
            obj = yaml.safe_load(f)   # ymlファイルの読み込み
            param = AttrDict(obj)
    except Exception as e:            # ファイルがなければエラー出力し、終了
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    return param


# 画像データを0-1に正規化し、教師データをone-hot表現に変更
def normalize_input_data(x, y):
    x = x.astype('float32') / 255.0
    # カラーでない場合、扱いやすいように4次元にするの形にする
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    # 教師データをone hot表現に変換
    y_class = len(np.unique(y))         # 分類クラスの数
    y = np.eye(y_class)[y].astype(int)  # one hot表現に変換
    y = np.squeeze(y)                   # 1次元のところがあったら削除

    return x, y


# グレースケールをカラーに変換
def gray2rgb(gray_img):
    rgb_img = np.zeros((np.append(gray_img.shape, 3)), 'uint8')
    for i in range(gray_img.shape[0]):
        rgb_img[i, :, :, :] = cv2.cvtColor(gray_img[i, :, :], cv2.COLOR_GRAY2RGB)
    return rgb_img


# 画像を低解像度化
# 縮小してから同じ画像サイズに戻す
def change_resolution(x_train):
    down_sample = 4   # 画像サイズを1/4にする
    size, width, height, channel = x_train.shape
    low_res_img = np.zeros(x_train.shape, dtype='uint8')
    for idx in range(x_train.shape[0]):
        sample = x_train[idx].copy()
        sample = cv2.resize(sample, dsize=(int(width/down_sample), int(height/down_sample)))
        sample = cv2.resize(sample, dsize=(x_train.shape[1], x_train.shape[2]), interpolation=cv2.INTER_NEAREST)
        low_res_img[idx, :, :, :] = sample
    return low_res_img


# mnist or fashion mnistの読み込み (defaultはmnist)
def load_data(param):
    # データの読み込み カラーでない場合は、カラー画像に変換
    x_train, y_train, x_test, y_test = None, None, None, None  # 学習・テストデータの初期化
    if param.data_type == 'mnist':
        (gray_x_train, y_train), (gray_x_test, y_test) = mnist.load_data()
        x_train = gray2rgb(gray_x_train)    # VGGでimagenetの学習済みモデルを使用するためにカラーに変換
        x_test = gray2rgb(gray_x_test)      # VGGでimagenetの学習済みモデルを使用するためにカラーに変換
        print('load mnist.')
    elif param.data_type == 'fashion_mnist':
        (gray_x_train, y_train), (gray_x_test, y_test) = fashion_mnist.load_data()
        x_train = gray2rgb(gray_x_train)   # VGGでimagenetの学習済みモデルを使用するためにカラーに変換
        x_test = gray2rgb(gray_x_test)     # VGGでimagenetの学習済みモデルを使用するためにカラーに変換
        print('load fashion_mnist.')
    elif param.data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('load cifar10.')

    # ノイズ付与データとそれ以外を半分に分ける
    # _vgg_clean: VGG16用ノイズ付与前のデータ,  _vgg_noise: VGG16用ノイズ付与のデータ
    # _gan_clean: GAN用ノイズ付与前のデータ, _gan_noise: GAN用ノイズ付与データ
    x_vgg_clean, x_gan_clean, y_vgg_clean, y_gan_clean = \
        train_test_split(x_train, y_train, test_size=param.gan_data_size, random_state=param.random_state)
    x_vgg_noise = change_resolution(x_vgg_clean)  # VGG16で学習用に解像度低下させた画像を作成
    x_gan_noise = change_resolution(x_gan_clean)  # GAN学習用に解像度低下させた画像を作成
    y_vgg_noise = y_vgg_clean.copy()
    y_gan_noise = y_gan_clean.copy()

    # データを0-1に変換し、教師データをone hot表現に変換
    x_vgg_clean, y_vgg_clean = normalize_input_data(x_vgg_clean, y_vgg_clean)
    x_gan_clean, y_gan_clean = normalize_input_data(x_gan_clean, y_gan_clean)
    x_vgg_noise, y_vgg_noise = normalize_input_data(x_vgg_noise, y_vgg_noise)
    x_gan_noise, y_gan_noise = normalize_input_data(x_gan_noise, y_gan_noise)
    x_test, y_test = normalize_input_data(x_test, y_test)

    # 場合分けして属性アクセスできるように
    vgg_data = AttrDict({'x_clean': x_vgg_clean, 'y_clean': y_vgg_clean,
                         'x_noise': x_vgg_noise, 'y_noise': y_vgg_noise})
    gan_data = AttrDict({'x_clean': x_gan_clean, 'y_clean': y_gan_clean,
                         'x_noise': x_gan_noise, 'y_noise': y_gan_noise})
    test_data = AttrDict({'x_test': x_test, 'y_test': y_test})

    return vgg_data, gan_data, test_data


# Generatorで生成した画像の確認
def image_generate(gan_data, generator):
    # gan用の画像の2枚を表示
    generated_images = generator.predict(gan_data.x_noise[0:2, :, :])
    plt.figure(figsize=(8, 4), )
    plt.axis('off')
    plt.subplot(2, 3, 1)
    plt.imshow(gan_data.x_noise[0, :, :, :])
    plt.title('Noise')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(generated_images[0, :, :, :])
    plt.title('Generate')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(gan_data.x_clean[0, :, :, :])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(gan_data.x_noise[1, :, :, :])
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(generated_images[1, :, :, :])
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(gan_data.x_clean[1, :, :, :])
    plt.axis('off')
    plt.show()


# 学習済みモデルの重みを読み込む
def load_weight(model_path, model):
    try:
        if os.path.exists(model_path):
            model.load_weights(model_path)
    except Exception as e:  # ファイルがなければエラー出力し、終了
        print('Exception occurred while loading generator model...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    return model
