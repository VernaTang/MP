# from moviepy.editor import AudioFileClip
# import argparse
# import os
# import librosa
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, help="The original data path")
# parser.add_argument("--output_dir", type=str, help="The path of the output files")
# args = parser.parse_args()
#
#
# video_list = os.listdir(args.data_dir)
#
# for i in range(0, len(video_list)):
#     video_names = os.path.join(args.data_dir, video_list[i])
#     if AudioFileClip(video_names):
#         my_audio_clip = AudioFileClip(video_names)
#     else:
#         print(video_names, "This video can not extracted!!!")
#
#     f_name = video_list[i].split(".")[0] + ".wav"
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     my_audio_clip.write_audiofile(os.path.join(args.output_dir, f_name))
#

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2020. ZZL
# @Time     : 2020/4/27
# @Author   : ZL.Z
# @Reference: None
# @Email    : zzl1124@mail.ustc.edu.cn
# @FileName : acoustic_feature.py
# @Software : Python3.6;PyCharm;Windows10
# @Hardware : Intel Core i7-4712MQ;NVIDIA GeForce 840M
# @Version  :  V1.1: 2020/5/15
#              1. 端点检测增加避免语音过短，比如出现只有一帧情况时产生bug
#              2. 增加librosa.load函数的音频采样率传入参数
#              3. 对能量值和求LPC所需值等，利用np.finfo(np.float64).eps，避免其值为0，防止后续取log和求LPC出错(eps是取非负的最小值)
#              4. 修复activity_detect中计算浊音段时由于最后数帧能量都超过阈值，程序只将起始帧加入列表，导致的浊音段为奇数的bug
#              V1.0: 2020/4/27-2020/5/9
# @License  : GPLv3
# @Brief    : 声学特征提取
import os
import subprocess
import numpy as np
import librosa
import librosa.display
from scipy.signal import lfilter, get_window
from scipy.stats import skew, kurtosis
import soundfile as sf
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 当坐标轴有负号的时候可以显示负号


def _print(bl=True, s=None):
    if bl:
        print(s)
    else:
        pass


def func_format(x, pos):
    return "%d" % (1000 * x)

def OpenSmileIS09(i_file_dir, o_file):
    files = os.listdir(i_file_dir)
    config_file = "D:/PycharmProjects/MP/opensmile/opensmile-3.0-win-x64/config/is09-13/IS09_emotion.conf"
    for i in files:
        file = os.path.join(i_file_dir, i)
        class_name = i.split(".")[0]
        cmd = "SMILExtract -C %s -I %s -O %s -class %s" % (config_file, file, o_file, class_name)
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    current_path = os.getcwd()
    wave_file = os.path.join(current_path, "D:\PycharmProjects\Temp1/data/audio_split/MELD/dev/dia0_utt0.wav")
    #wave_file_vad = wave_file.split(".")[0] + "_vad.wav"
    #feature_file = os.path.join(current_path, "features/feature.csv")
    # np.set_printoptions(threshold=np.inf)
    # 端点检测
    #vad = VAD(wave_file, min_interval=15, pt=False)
    #sf.write(wave_file_vad, vad.wav_dat_utterance, 16000, "PCM_16")
    #vad.plot()
    # 利用openSmile工具进行特征提取
    #opensmile_f = OpenSmileFeatureSet(wave_file_vad)
    #feat = opensmile_f.get_IS09(feature_file)
    # print(feat.shape, feat.dtype, feat)
    # 常用声学特征
    # my_acoustic_f = my_acoustic_features(wave_file_vad)
    # print(my_acoustic_f.shape, my_acoustic_f.dtype, my_acoustic_f)
    # 韵律学特征
    # rhythm_f = RhythmFeatures(wave_file)
    # print(rhythm_f.intensity())
    #rhythm_f.plot()
    # 基于谱的相关特征
    #spectrum_f = SpectrumFeatures(wave_file_vad)
    #spectrum_f.plot()
    # 音质特征
    #quality_f = QualityFeatures(wave_file_vad)
    #quality_f.plot()
    # 声谱图特征
    #spectrogram_f = Spectrogram(wave_file_vad)
    #spectrogram_f.plot()

    # Opensmile extract audio features following with IS09 config file
    #OpenSmileIS09("D:/PycharmProjects/MP/data/original/MELD/audio/test", "D:/PycharmProjects/MP/data/outputIS09_test.csv")

    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.neighbors import KNeighborsClassifier

    csv_file = "D:/PycharmProjects/MP/data/outputIS09_train.csv"
    values = pd.read_csv(csv_file, header=None)
    X = values.iloc[:, 1:384] # remove 1st and last cols.

    csv_f2 = "D:/PycharmProjects/Temp1/data/original/MELD/train.csv"
    y_values = pd.read_csv(csv_f2)
    y = y_values[:-2]['Sentiment']

    knn = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(knn, n_features_to_select=5)
    sfs.fit(X, y)

    result = sfs.transform(X)
    print("fINISH!!!!")




