# -*- coding: utf-8 -*-
import argparse
from mylib.srgan import SRGAN
from mylib.utils import load_param
from mylib.utils import load_data
from mylib.utils import load_weight
from mylib.utils import image_generate
from mylib.image_classification import VGG


def main(arg):
    # 最初にパラメータの読み込みとデータの読み込みを行う
    param = load_param(arg.param)
    print('パラメータ読み込み完了.')

    vgg_data, gan_data, test_data = load_data(param)
    print('データサイズ:　', vgg_data.x_clean.shape, vgg_data.y_clean.shape, vgg_data.x_noise.shape, vgg_data.y_noise.shape,
          gan_data.x_clean.shape, gan_data.y_clean.shape, gan_data.x_noise.shape, gan_data.y_noise.shape,
          test_data.x_test.shape, test_data.y_test.shape)
    print('データ読み込み完了.')

    ######################################################################################
    # GANの学習
    ######################################################################################
    if arg.gan:
        print('GANの実行.')
        srgan = SRGAN(param, gan_data)
        generator = srgan.generator()
        discriminator = srgan.discriminator()
        srgan.train(generator, discriminator)

    ######################################################################################
    # 生成された画像のチェック
    ######################################################################################
    if arg.check_gan:
        print('画像チェック.')
        srgan = SRGAN(param, gan_data)
        generator = srgan.generator()
        generator = load_weight('./model/generator_model.h5', generator)
        image_generate(gan_data, generator)

    ######################################################################################
    # 本物のデータセットと生成したデータセットの学習を行うおよび学習済みモデルでテストデータを予測
    ######################################################################################
    if arg.train or arg.prediction:
        model, model_name = None, None
        if 'original' in arg.data:      # ノイズなし画像で識別 (default)
            model_name = './model/original_image_classification_model.h5'
            model = VGG(param, vgg_data.x_clean, test_data.x_test, vgg_data.y_clean, test_data.y_test, model_name)
            print('original data.')
        elif 'noise' in arg.data:       # ノイズ付与画像で識別
            model_name = './model/noise_image_classification_model.h5'
            model = VGG(param, vgg_data.x_noise, test_data.x_test, vgg_data.y_noise, test_data.y_test, model_name)
            print('noise data.')
        elif 'generate' in arg.data:    # DCGANでノイズ除去した画像で識別
            srgan = SRGAN(param, gan_data)
            model_name = './model/generate_image_classification_model.h5'
            generator = srgan.generator()
            generator = load_weight('./model/generator_model.h5', generator)

            # 画像の数が大きいとgeneratorできないので分けて実行
            generated_images = generator.predict(vgg_data.x_noise, batch_size=param.gan_batch)
            model = VGG(param, generated_images, test_data.x_test, vgg_data.y_noise, test_data.y_test, model_name)
            print('generate data.')

        # 学習のみ
        if arg.train and not arg.prediction:
            print('VGGの学習.')
            model.train()
        # テストのみ
        elif not arg.train and arg.prediction:
            print('VGGのテスト.')
            model.test()
        # 学習とテスト
        elif arg.train and arg.prediction:
            print('VGGの学習とテスト.')
            model.train()
            model.test()


# 引数設定
def parse_input():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-g', '--gan', default=False,
                        action="store_true", help='run Super-Resolution generative Adversarial Networks.')  # ganの実行
    parser.add_argument('-c', '--check_gan', default=False,
                        action="store_true", help='Check generate data.')  # ganで生成したデータのチェック
    parser.add_argument('-t', '--train', default=False,
                        action="store_true", help='training.')             # 本物のデータセットと生成したデータセットの学習を行う
    parser.add_argument('-p', '--prediction', default=False,
                        action="store_true", help='prediction.')           # 本物のデータセットと生成したデータセットで精度を比較
    # original: 加工なしデータ, noise: ノイズ付与データ, generate: ganでノイズ除去したデータ
    parser.add_argument('-d', '--data', default='original', help='use noise data.')           # 学習時にデータを使うか
    parser.add_argument('-param', '--param', default='./param.yml', help='load parameter.')   # パラメータの読み込み
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_input()
    main(args)
