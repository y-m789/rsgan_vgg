# -*- coding: utf-8 -*-
# Super-Resolution ganを行うクラス
from mylib.utils import check_dir
import numpy as np
from tqdm import tqdm
import keras.backend as K
from keras.layers import add
from keras.layers import Input
from keras.layers import PReLU
from keras.models import Model
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.applications.vgg19 import VGG19
from keras.losses import mean_squared_error
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.normalization import BatchNormalization


class SRGAN:
    def __init__(self, param, gan_data):
        self.param = param
        self.x_lr = gan_data.x_noise  # 低解像度画像
        self.x_hr = gan_data.x_clean  # 高解像度画像
        self.img_lr_size = gan_data.x_noise.shape[1:]  # 画像サイズ
        self.img_hr_size = gan_data.x_clean.shape[1:]  # 画像サイズ
        self.gen_momentum = 0.5
        self.dis_alpha = 0.2
        self.dis_hidden = 1024
        self.opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # GeneratorのResidual network Blocks前のconvolutionのパラメータ
        self.gen_input_filters = 64
        self.gen_input_kernal = 9
        self.gen_input_strides = 1

        # GeneratorのResidual network Blocksのconvolutionのパラメータ
        self.gen_resnet_filters = 64
        self.gen_resnet_kernal = 3
        self.gen_resnet_strides = 1

        # GeneratorのResidual network Blocks後のconvolutionのパラメータ
        self.gen_after_resnet_filters = 64
        self.gen_after_resnet_kernal = 3
        self.gen_after_resnet_strides = 1

        # Generatorのupsampling blockのconvolutionのパラメータ
        self.gen_up_filters = 256
        self.gen_up_kernal = 3
        self.gen_up_strides = 1
        self.gen_up_size = 2
        self.gen_up_alpha = 0.2
        self.up_size = 2

        # Generatorのupsampling後のconvolutionのパラメータ
        self.gen_out_filters = 3
        self.gen_out_kernal = 9
        self.gen_out_strides = 1

        # Discriminatorのinput blockのconvolutionのパラメータ
        self.dis_input_filters = 64
        self.dis_input_kernal = 3
        self.dis_input_strides = 1
        self.dis_input_alpha = 0.2

        # Discriminatorのblockのconvolutionのパラメータ
        self.dis_filters = [64, 128, 128, 256, 256, 512, 512]
        self.dis_kernal = [3, 3, 3, 3, 3, 3, 3]
        self.dis_strides = [2, 1, 2, 1, 2, 1, 2]
        self.dis_momentum = 0.5

    # VGG19でlossを計算（画像サイズがimagenet以上の場合）
    def vgg_loss(self, y_true, y_pred):
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=self.img_hr_size)
        vgg.trainable = False
        for l in vgg.layers:
            l.trainable = False
        model = Model(inputs=vgg.input, outputs=vgg.output)
        model.trainable = False
        return K.mean(K.square(model(y_true) - model(y_pred)))

    # 画像のlossを計算(画像サイズが小さい場合)
    @staticmethod
    def mse_loss(y_true, y_pred):
        image_loss = mean_squared_error(y_true, y_pred)
        return image_loss

    # Generatorで使用するResidual network block
    def resnet_block(self, model):
        gen = model
        model = Conv2D(filters=self.gen_resnet_filters, kernel_size=self.gen_resnet_kernal,
                       strides=self.gen_resnet_strides, padding="same")(model)
        model = BatchNormalization(momentum=self.gen_momentum)(model)
        model = PReLU(shared_axes=[1, 2])(model)
        model = Conv2D(filters=self.gen_resnet_filters, kernel_size=self.gen_resnet_filters,
                       strides=self.gen_resnet_strides, padding="same")(model)
        model = BatchNormalization(momentum=self.gen_momentum)(model)
        model = add([gen, model])
        return model

    # Generatorで使用するupsampling block
    def upsampling_block(self, model):
        # Conv2DとUpSampling2Dの代わりにConv2DTransposeが使用できる
        model = Conv2D(filters=self.gen_up_filters, kernel_size=self.gen_up_kernal,
                       strides=self.gen_up_strides, padding="same")(model)
        model = UpSampling2D(size=self.up_size)(model)
        # model = Conv2DTranspose(filters=self.gen_up_filters, kernel_size=self.gen_up_kernal,
        #                         strides=self.gen_up_strides, padding="same")(model)
        model = LeakyReLU(alpha=self.gen_up_alpha)(model)
        return model

    # Discriminatorで繰り返し生成するnetwork block
    def discriminator_block(self, model, filters, kernel_size, strides):
        model = Conv2D(filters=filters, kernel_size=kernel_size,
                       strides=strides, padding="same")(model)
        model = BatchNormalization(momentum=self.dis_momentum)(model)
        model = LeakyReLU(alpha=self.dis_alpha)(model)
        return model

    ######################################################################################
    # generatorとdiscriminatorの結合部分
    ######################################################################################
    def combined_network(self, discriminator, generator):
        discriminator.trainable = False
        gan_input = Input(shape=self.img_lr_size)
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=[x, gan_output])
        # 画像サイズが小さい場合、MSEでlossを計算
        # 画像サイズが大きい場合、VGGでlossを計算
        gan.compile(loss=[self.mse_loss, "binary_crossentropy"],
                    loss_weights=[1., 1e-3], optimizer=self.opt)
        # gan.compile(loss=[self.vgg_loss, "binary_crossentropy"],
        #             loss_weights=[1., 1e-3], optimizer=self.opt)
        return gan

    ######################################################################################
    # generator部分
    ######################################################################################
    def generator(self):
        # Residual network Blocks前のconvolution
        model_input = Input(shape=self.img_lr_size)
        model = Conv2D(filters=self.gen_input_filters, kernel_size=self.gen_input_kernal,
                       strides=self.gen_input_strides, padding="same")(model_input)
        model = PReLU(shared_axes=[1, 2])(model)
        gen_model = model

        # 同じパラメータのResidual network Blocksを16つ生成
        for idx in range(16):
            model = self.resnet_block(model)

        # Residual network Blocks後のconvolution
        model = Conv2D(filters=self.gen_after_resnet_filters, kernel_size=self.gen_after_resnet_kernal,
                       strides=self.gen_after_resnet_strides, padding="same")(model)
        model = BatchNormalization(momentum=self.gen_momentum)(model)
        model = add([gen_model, model])

        # 同じパラメータのUpSampling Blocksを2つ生成
        # 画像サイズを同じにしたのでコメントアウト
        # for idx in range(2):
        #     model = self.upsampling_block(model)

        # UpSampling後のconvolution
        model = Conv2D(filters=self.gen_out_filters, kernel_size=self.gen_out_kernal,
                       strides=self.gen_out_strides, padding="same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs=model_input, outputs=model)
        print(generator_model.summary())
        return generator_model

    ######################################################################################
    # discriminator部分
    ######################################################################################
    def discriminator(self):
        model_input = Input(shape=self.img_hr_size)
        model = Conv2D(filters=self.dis_input_filters, kernel_size=self.dis_input_kernal,
                       strides=self.dis_input_strides, padding="same")(model_input)
        model = LeakyReLU(alpha=self.dis_input_alpha)(model)

        # discriminator部分を7つ生成
        for idx in range(7):
            model = self.discriminator_block(model, self.dis_filters[idx], self.dis_kernal[idx], self.dis_strides[idx])

        model = Flatten()(model)
        model = Dense(self.dis_hidden)(model)
        model = LeakyReLU(alpha=self.dis_alpha)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=model_input, outputs=model)
        print(discriminator_model.summary())
        return discriminator_model

    ######################################################################################
    # trianing部分
    ######################################################################################
    def train(self, generator, discriminator):
        srgan = self.combined_network(discriminator, generator)
        # 画像サイズが小さい場合、MSEでlossを計算
        # 画像サイズが大きい場合、VGGでlossを計算
        generator.compile(loss=self.mse_loss, optimizer=self.opt)
        # generator.compile(loss=self.vgge_loss, optimizer=self.opt)
        discriminator.compile(loss="binary_crossentropy", optimizer=self.opt)
        for step in tqdm(range(self.param.gan_epoch)):
            image_idx = np.random.randint(0, self.x_lr.shape[0], size=self.param.gan_batch)
            image_hr_batch = self.x_hr[image_idx, :, :, :]
            image_lr_batch = self.x_lr[image_idx, :, :, :]

            generated_img_sr = generator.predict(image_lr_batch)
            real_y = np.ones(self.param.gan_batch) - np.random.random_sample(self.param.gan_batch)*0.2
            fake_y = np.random.random_sample(self.param.gan_batch)*0.2

            discriminator.trainable = True

            loss_real = discriminator.train_on_batch(image_hr_batch, real_y)
            loss_fake = discriminator.train_on_batch(generated_img_sr, fake_y)
            discriminator_loss = 0.5 * np.add(loss_fake, loss_real)

            image_idx = np.random.randint(0, self.x_lr.shape[0], size=self.param.gan_batch)
            image_hr_batch = self.x_hr[image_idx, :, :, :]
            image_lr_batch = self.x_lr[image_idx, :, :, :]

            gan_y = np.ones(self.param.gan_batch) - np.random.random_sample(self.param.gan_batch) * 0.2

            discriminator.trainable = False
            gan_loss = srgan.train_on_batch(image_lr_batch, [image_hr_batch, gan_y])

            print('\n')
            print('discriminator_loss : %f' % discriminator_loss)
            print('gan_loss :', gan_loss)

        check_dir('./model')
        generator.save('./model/generator_model.h5', include_optimizer=False)
        # discriminator.save('./model/discriminator_model.h5', include_optimizer=False)
        # srgan.save('./model/srgan_model.h5', include_optimizer=False)
