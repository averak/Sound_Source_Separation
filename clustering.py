# -*- coding:utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import scipy.io.wavfile as wf
from scipy import signal
import glob, sys


class Infer(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.batch_size = 64
        self.epochs = 20

        self.model_path = './ckpt/model_new.hdf5'

        # ニューラルネットワークの構築
        self.model = self.build_NN()

    def train(self):
        ## -----*----- 学習 -----*----- ##
        # 特徴抽出
        self.extract_features()

        # 学習
        self.model.fit(self.datas['spec'], self.datas['label'],
                       batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.2)

        # 学習モデルを保存
        self.model.save_weights(self.model_path)

    def build_NN(self):
        ## -----*----- NNの構築 -----*----- ##
        model = Sequential()
        #model.add(LSTM(512, input_shape=(6321, 1), activation='relu'))
        model.add(Dense(512, input_shape=(6321, 1), activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.add(Dropout(0.5))

        # モデルのコンパイル
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

        return model

    def extract_features(self):
        ## -----*----- 学習データセットを用意 -----*----- ##
        self.datas = {'spec': [], 'label': []}
        speakers = glob.glob('./teacher_datas/*')

        for n in range(100):
            # 各話者のスペクトログラムを格納
            spec = [self.specgram(glob.glob('{0}/*.wav'.format(speakers[d]))[n])
                    for d in range(len(speakers))]

            # スペクトログラムの和を追加
            self.datas['spec'].append(sum(spec))
            self.datas['label'].append([])

            # 時間周波数成分をクラスタリングする
            for i in range(129 * 49):
                max = {'index': 0, 'num': 0.0}
                for d in range(len(speakers)):
                    if spec[d][i] > max['num']:
                        max['index'] = d
                        max['num'] = spec[d][i]
                self.datas['label'][-1].append(max['index'])

            # one-hot vectorに変換
            self.datas['label'][-1] = to_categorical(self.datas['label'][-1])

        # NumPy配列に変換
        self.datas['spec'] = np.array(self.datas['spec'])
        self.datas['label'] = np.array(self.datas['label'])

        print (self.datas['spec'].shape)
        print (self.datas['label'].shape)

        # ランダムに並べ替え
        perm = np.arange(len(self.datas['spec']))
        np.random.shuffle(perm)
        self.datas['spec'] = self.datas['spec'][perm]
        self.datas['label'] = self.datas['label'][perm]

    def specgram(self, file):
        ## -----*----- スペクトログラムの取得 -----*----- ##
        # 音声ファイル読み込み
        fs, wav = wf.read(file)
        # STFT読み込み
        _, _, spec = signal.stft(wav, fs, nperseg=256)
        spec = np.reshape(spec, (129 * 49, 1))

        return spec

    def istft(self, spec, fs=16000.0):
        ## -----*----- スペクトログラムを音声に変換 -----*----- ##
        _, wav = signal.istft(spec, fs, nperseg=256)

        return wav

    def predict(self, data):
        ## -----*----- 推論 -----*----- ##
        return self.model.predict(data)


if __name__ == '__main__':
    infer = Infer()

    infer.train()
