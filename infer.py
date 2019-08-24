# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Add, Input, Dropout
from tensorflow.keras import Model
import numpy as np
import scipy.io.wavfile as wf
from scipy import signal
import glob, sys
from unet import UNet


class Infer(object):
    def __init__(self):
        # U-Net
        self.unet = UNet()

    def train(self):
        ## -----*----- 学習 -----*----- ##
        # 特徴抽出
        self.extract_features()

        # 学習
        self.unet.train(self.datas['spec'], self.datas['label'])

    def extract_features(self):
        ## -----*----- 学習データセットを用意 -----*----- ##
        self.datas = {'spec': [], 'label': []}
        speakers = glob.glob('./teacher_datas/*')
        tmp = []
        for speaker in speakers:
            files = glob.glob('{0}/*.wav'.format(speaker))
            tmp.append([])
            for file in files:
                spec = self.specgram(file) * (np.random.rand() * 2.0 + 0.5)
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
                    sum = tmp[0][i][j][k] + tmp[1][i][j][k]
                    if sum == 0.0:
                        self.datas['label'][-1][-1][-1] = 0
                    else:
                        self.datas['label'][-1][-1][-1] = abs(tmp[0][i][j][k] / sum)

        '''for i in range(n):
            for i2 in range(n):
                if cnt % 100 == 0:
                    print('{0}/{1}'.format(cnt, n ** 2))
                cnt += 1
                sum_spec = np.array(tmp[0][i]) + np.array(tmp[1][i2])
                self.datas['spec'].append(sum_spec)
                self.datas['label'].append([])
                for j in range(128):
                    self.datas['label'][-1].append([])
                    for k in range(48):
                        self.datas['label'][-1][-1].append([])
                        sum = abs(tmp[0][i][j][k] + tmp[1][i2][j][k])
                        if sum == 0.0:
                            self.datas['label'][-1][-1][-1] = 0
                        else:
                            self.datas['label'][-1][-1][-1] = abs(tmp[0][i][j][k]) / sum'''

        # NumPy配列に変換
        self.datas['spec'] = np.array(self.datas['spec'])
        self.datas['label'] = np.array(self.datas['label'])
        self.datas['spec'] = np.reshape(self.datas['spec'], (
            self.datas['spec'].shape[0], self.datas['spec'].shape[1], self.datas['spec'].shape[2], 1))
        self.datas['label'] = np.reshape(self.datas['label'], (
            self.datas['label'].shape[0], self.datas['label'].shape[1], self.datas['label'].shape[2], 1))

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
        spec = self.stft(wav)

        ret = []
        for i in range(len(spec) - 1):
            ret.append(np.delete(spec[i], -1))
        ret = np.array(ret)

        return ret

    def stft(self, wav, fs=1600):
        ## -----*----- STFT（短時間フーリエ変換） -----*----- ##
        _, _, spec = signal.stft(wav, fs, nperseg=256)
        return spec

    def istft(self, spec, fs=16000.0):
        ## -----*----- スペクトログラムを音声に変換 -----*----- ##
        _, wav = signal.istft(spec, fs, nperseg=256)

        return wav

    def predict(self, data):
        ## -----*----- 推論 -----*----- ##
        return self.unet.predict(data)


if __name__ == '__main__':
    infer = Infer()

    # 学習
    infer.train()

    a_voice = wf.read('./teacher_datas/あー/0.wav')[1]
    i_voice = wf.read('./teacher_datas/いー/1.wav')[1]
    mixed = a_voice + i_voice
    wf.write('mixed.wav', 16000, mixed)

    spec_a = infer.stft(a_voice)
    spec_i = infer.stft(i_voice)
    spec_mixed = infer.stft(mixed)

    # -----------
    out = spec_mixed
    for i in range(128):
        for j in range(48):
            if spec_mixed[i][j] == 0:
                out[i][j] = 0
            else:
                out[i][j] *= abs(spec_a[i][j] / spec_mixed[i][j])
                print(abs(spec_a[i][j] / spec_mixed[i][j]))

    out_wav = np.array(infer.istft(out), dtype='int16')
    wf.write('output.wav', 16000, out_wav)
    #exit(0)
    # -----------1
    spec_mixed = infer.stft(wf.read('./mixed.wav')[1])

    input = []
    for i in range(128):
        input.append([])
        for j in range(48):
            input[-1].append(spec_mixed[i][j])
    input = np.reshape(input, (1, 128, 48, 1))
    vector = infer.predict(input)[0]

    tmp = spec_mixed
    # 出力ベクトルを掛け合わせる
    for i in range(129):
        if i == 128:
            break
        for j in range(48):
            if j == 48:
                break
            tmp[i][j] *= vector[i][j][0]
            # print (vector[i][j][0])

    rewav = infer.istft(tmp)
    rewav = np.array(rewav, dtype='int16')

    wf.write('infer.wav', 16000, rewav)
    out_wav = np.array(infer.istft(spec_mixed), dtype='int16')
    wf.write('origin.wav', 16000, out_wav)

    # exit(0)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(mixed)
    plt.plot(rewav)
    # plt.plot(a_voice)
    plt.show()
