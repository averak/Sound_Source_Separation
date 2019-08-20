# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Add, Input, Dropout
from tensorflow.keras import Model
import numpy as np
import scipy.io.wavfile as wf
from scipy import signal
import glob, sys
import keras.backend as K


class Recognizer(object):
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10

        self.model_path = './ckpt/model.hdf5'

    def train(self):
        ## -----*----- 学習 -----*----- ##
        # 特徴抽出
        self.extract_features()
        # NNを構築
        self.build_NN()

        # 学習
        self.model.fit(self.datas['spec'], self.datas['label'],
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)
        # 学習モデルを保存
        self.model.save_weights(self.model_path)

    def build_NN(self):
        ## -----*----- NNの構築 -----*----- ##
        # モデルの定義
        self.model = self.unet()

    def unet(self, pretrained_weights=None, input_size=(128, 48, 1)):
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

        # model.summary()

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def extract_features(self):
        ## -----*----- 学習データセットを用意 -----*----- ##
        # 教師データ一覧
        files = glob.glob('./teacher_data/*/*.wav')

        self.datas = {'spec': [], 'label': []}
        speakers = glob.glob('./teacher_data/*')
        tmp = []
        for speaker in speakers:
            files = glob.glob('{0}/*.wav'.format(speaker))
            tmp.append([])
            for file in files:
                spec = self.specgram(file) * np.random.rand() * 3.0
                tmp[-1].append(spec)

        cnt = 0
        n = min([len(tmp[0]), len(tmp[1])])
        for i in range(n):
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
            sum_spec = np.array(tmp[0][i]) + np.array(tmp[1][i])
            self.datas['spec'].append(sum_spec)
            self.datas['label'].append([])
            for j in range(128):
                self.datas['label'][-1].append([])
                for k in range(48):
                    self.datas['label'][-1][-1].append([])
                    if tmp[0][i][j][k] > tmp[1][i][j][k]:
                        self.datas['label'][-1][-1][-1] = 1
                    else:
                        self.datas['label'][-1][-1][-1] = 0

        # NumPy配列に変換
        self.datas['spec'] = np.array(self.datas['spec'])
        self.datas['label'] = np.array(self.datas['label'])
        self.datas['spec'] = np.reshape(self.datas['spec'], (
            self.datas['spec'].shape[0], self.datas['spec'].shape[1], self.datas['spec'].shape[2], 1))
        self.datas['label'] = np.reshape(self.datas['label'], (
            self.datas['label'].shape[0], self.datas['label'].shape[1], self.datas['label'].shape[2], 1))

        print (self.datas['spec'].shape)
        print (self.datas['label'].shape)

        # featuresの長さの連番数列を作る
        perm = np.arange(len(self.datas['spec']))
        # ランダムに並べ替え
        np.random.shuffle(perm)
        self.datas['spec'] = self.datas['spec'][perm]
        self.datas['label'] = self.datas['label'][perm]

    def specgram(self, file):
        ## -----*----- スペクトログラムの取得 -----*----- ##
        # 音声ファイル読み込み
        fs, wav = wf.read(file)
        # STFT読み込み
        _, _, spec = signal.stft(wav, fs, nperseg=256)

        ret = []
        for i in range(len(spec) - 1):
            ret.append(np.delete(spec[i], -1))
        ret = np.array(ret)

        return ret

    def istft(self, spec, fs):
        ## -----*----- スペクトログラムを音声に変換 -----*----- ##
        _, wav = signal.stft(spec, fs, nperseg=256)

        return wav


if __name__ == '__main__':
    cnn = Recognizer()
    # 学習
    cnn.train()
