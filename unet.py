# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Add, Input, Dropout
from tensorflow.keras import Model


class UNet(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.batch_size = 64
        self.epochs = 30

        self.model_path = './ckpt/model.hdf5'

        # モデルを構築
        self.model = self.build()

    def train(self, x, y):
        ## -----*----- 学習 -----*----- ##
        self.model.fit(x, y,
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        # 学習モデルを保存
        self.model.save_weights(self.model_path)

    def build(self, input_size=(128, 48, 1)):
        ## -----*----- U-Netの構築 -----*----- ##
        inputs = Input(input_size)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = Add()([drop4, up6])
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = Add()([conv3, up7])
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = Add()([conv2, up8])
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = Add()([conv1, up9])
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        # モデルを定義
        model = Model(inputs=inputs, outputs=conv10)
        # モデルをコンパイル
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

        return model

    def load_model(self):
        ## -----*----- 学習済みモデルの読み込み -----*----- ##
        self.model.load_weights(self.path)

    def predict(self, data):
        ## -----*----- 推論 -----*----- ##
        return self.model.predict(data)