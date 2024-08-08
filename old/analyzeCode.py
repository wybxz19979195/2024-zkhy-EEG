# 本脚本的主要功能是进行分析中山七院的脑电数据
# 主要分析刺激前后的脑电数据的指标变化:相干性分析、平均相干性、功率图谱、头皮地形图

import mne
import pandas as pd
from scipy.io import savemat
from numpy.fft import fft
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, lfilter
import scipy.signal as signal

# 支持显示汉语
plt.rcParams['font.sans-serif'] = ['Fangsong']
plt.rcParams['axes.unicode_minus'] = False


def get_bandstop_filter_para(order, cutoff_freq, fs):
    nyquist_freq = fs / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    filter_order = order
    b, a = signal.butter(filter_order, normalized_cutoff, btype='bandstop')
    zi = signal.lfilter_zi(b, a)
    return b, a, zi


def get_bandpass_filter_para(order, cutoff_freq, fs):
    nyquist_freq = fs / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    filter_order = order
    b, a = signal.butter(filter_order, normalized_cutoff, btype='bandpass')
    zi = signal.lfilter_zi(b, a)
    return b, a, zi


def filt_process(indata, num_channels, num_samples, notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a):
    filt_data = np.zeros((num_channels, num_samples))
    for channel in range(num_channels):
        det_data = signal.detrend(indata[channel, :])
        notch50_filter = signal.lfilter(notch50_b, notch50_a, det_data)
        notch100_filter = signal.lfilter(notch100_b, notch100_a, notch50_filter)
        pass_filter = signal.lfilter(pass_b, pass_a, notch100_filter)
        filt_data[channel, :] = pass_filter
    return filt_data


def segment_and_removal(indata, segment_length, start_time, end_time, segment_threshold, fs):
    num_channel = indata.shape[0]
    num_samples = indata.shape[1]
    # 计算分段的起始和结束索引
    start_index = int(start_time * fs)
    end_index = int(num_samples + end_time * fs)
    segments = []
    for channel in range(num_channel):
        single_channel_data = indata[channel, :]  # 提取出单个通道中的值
        # 对每个通道的数据进行分段
        channel_segments = []
        for i in range(start_index, end_index, int(segment_length * fs)):
            segment = single_channel_data[i: i + int(segment_length * fs)]
            if len(segment) == segment_length * fs:
                channel_segments.append(segment)
        segments.append(channel_segments)

    segments = np.array(segments)  # 形状：通道数*分段数*每段长度
    segments = np.transpose(segments, (1, 0, 2))  # 形状：分段数*通道数*每段长度
    # 对坏段进行剔除
    new_segments = []
    for i in range(segments.shape[0]):
        segment = segments[i, :, :]
        has_value_over_threshold = np.any(np.abs(segment) > segment_threshold)
        if not has_value_over_threshold:
            new_segments.append(segment)
    new_segments = np.array(new_segments)
    return new_segments


def calcu_psd(indata, fs):
    L = np.size(indata)
    window = np.hanning(L)
    window_data = indata * window
    P = fft(window_data)
    P2 = P[: L // 2 + 1]
    P1 = (1 / (fs * L)) * np.abs(P2) ** 2
    P1[1:-2] = 2 * P1[1:-2]
    freq = np.arange(0, L // 2 + 1) / L * fs
    psd = P1
    return psd, freq


def find_indices(data, left, right):
    indices = []
    for index, value in enumerate(data):
        if left <= value < right:
            indices.append(index)
    return indices


def calcu_psd_power(segments, freqBand, fs):
    segment_num = segments.shape[0]
    channel_num = segments.shape[1]
    epoch_power = np.zeros((segment_num, 5))
    psd_segment = np.zeros((segment_num, 251))
    segment_power = []
    segment_psd = []
    for segment in range(segments.shape[0]):
        segment = segments[segment, :, :]
        # 计算每个通道的功率谱密度（psd）、频率分量（freq）
        channel_power = np.zeros((channel_num, 5))
        channel_psd = np.zeros((channel_num, 251))
        for channel in range(channel_num):
            single_data = segment[channel, :]  # 提取单个通道的值
            psd, freq = calcu_psd(single_data, fs)
            channel_psd[channel, :] = psd
            # 计算各个频带的绝对能量
            for j in range(freqBand.shape[0]):
                start_band = freqBand[j, 0]
                end_band = freqBand[j, 1]
                indices = find_indices(freq, start_band, end_band)
                channel_power[channel, j] = np.mean(psd[indices])
        segment_power.append(channel_power)
        segment_psd.append(channel_psd)
    segment_power = np.array(segment_power)
    segment_psd = np.array(segment_psd)
    return segment_power, segment_psd, freq


def show_2dscalp_topography(pre_evoked, aft_evoked, legend_label, ch_names):
    # 绘制二维头皮地形图
    title_band = ['全频段(1-60hz)', 'delta频段(1-4hz)', 'theta频段(4-8hz)', 'alpha频段(8-13hz)', 'beta频段(13-30hz)',
                  'gamma频段(30-60hz)']
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im0, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
            ax.set_ylabel(legend_label[0])
        if i == 1:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im1, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
        if i == 2:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im2, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
        if i == 3:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im3, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
        if i == 4:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im4, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
        if i == 5:
            max_value = np.mean([np.max(pre_evoked.data[:, i]), np.max(aft_evoked.data[:, i])])
            im5, _ = mne.viz.plot_topomap(pre_evoked.data[:, i], pre_evoked.info, show=False, axes=ax, names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_title(f'{title_band[i]}')
        # 第二行
        if i == 6:
            max_value = np.mean([np.max(pre_evoked.data[:, i - 6]), np.max(aft_evoked.data[:, i - 6])])
            im06, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                           names=ch_names,
                                           outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
            ax.set_ylabel(legend_label[1])
        if i == 7:
            max_value = np.mean([np.max(pre_evoked.data[:, i - 6]), np.max(aft_evoked.data[:, i - 6])])
            im7, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                          names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
        if i == 8:
            max_value = np.mean([np.max(pre_evoked.data[:, i - 6]), np.max(aft_evoked.data[:, i - 6])])
            im8, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                          names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
        if i == 9:
            max_value = np.max(pre_evoked.data[:, i - 6]) if np.max(pre_evoked.data[:, i - 6]) > np.max(
                aft_evoked.data[:, i - 6]) else np.max(aft_evoked.data[:, i - 6])
            im9, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                          names=ch_names,
                                          outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
        if i == 10:
            max_value = np.mean([np.max(pre_evoked.data[:, i - 6]), np.max(aft_evoked.data[:, i - 6])])
            im10, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                           names=ch_names,
                                           outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
        if i == 11:
            max_value = np.mean([np.max(pre_evoked.data[:, i - 6]), np.max(aft_evoked.data[:, i - 6])])
            im11, _ = mne.viz.plot_topomap(aft_evoked.data[:, i - 6], pre_evoked.info, show=False, axes=ax,
                                           names=ch_names,
                                           outlines='head', cmap='jet', vlim=(0, max_value * 3 / 5))
    # 定义颜色条的位置和大小
    cbar_ax = fig.add_axes([0.24, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')

    cbar_ax = fig.add_axes([0.374, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')

    cbar_ax = fig.add_axes([0.508, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')

    cbar_ax = fig.add_axes([0.639, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')

    cbar_ax = fig.add_axes([0.770, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im4, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')

    cbar_ax = fig.add_axes([0.91, 0.33, 0.005, 0.3])
    cbar = fig.colorbar(im5, cax=cbar_ax)
    cbar.set_label('Power(uv^2/hz)')


def calcu_segment_coherence(indata, fs, freqBand):
    coherence_all = []
    data = np.mean(indata, 0)
    channel_num = data.shape[0]
    samples_num = data.shape[1]
    coherence = np.zeros((channel_num, channel_num))
    coherence_delta = np.zeros((channel_num, channel_num))
    coherence_theta = np.zeros((channel_num, channel_num))
    coherence_alpha = np.zeros((channel_num, channel_num))
    coherence_beta = np.zeros((channel_num, channel_num))
    coherence_gamma = np.zeros((channel_num, channel_num))
    coherence_fiveband = np.zeros(5)
    for channel_x in range(channel_num):
        x_data = data[channel_x, :]
        for channel_y in range(channel_num):
            y_data = data[channel_y, :]
            frequencies, coh = signal.coherence(x_data, y_data, fs)
            for j in range(freqBand.shape[0]):
                start_band = freqBand[j, 0]
                end_band = freqBand[j, 1]
                indices = find_indices(frequencies, start_band, end_band)
                coherence_fiveband[j] = np.mean(coh[indices])

            coherence[channel_x, channel_y] = np.mean(coh)
            coherence_delta[channel_x, channel_y] = coherence_fiveband[0]
            coherence_theta[channel_x, channel_y] = coherence_fiveband[1]
            coherence_alpha[channel_x, channel_y] = coherence_fiveband[2]
            coherence_beta[channel_x, channel_y] = coherence_fiveband[3]
            coherence_gamma[channel_x, channel_y] = coherence_fiveband[4]

    coherence_all.append(coherence)
    coherence_all.append(coherence_delta)
    coherence_all.append(coherence_theta)
    coherence_all.append(coherence_alpha)
    coherence_all.append(coherence_beta)
    coherence_all.append(coherence_gamma)

    return np.array(coherence_all)


def show_coherence(pre_all_coherence, aft_all_coherence, ch_names, legend_label):
    title_band = ['全频段(1-60hz)', 'delta频段(1-4hz)', 'theta频段(4-8hz)', 'alpha频段(8-13hz)', 'beta频段(13-30hz)',
                  'gamma频段(30-60hz)']
    fig, axes = plt.subplots(2, 6, figsize=(30, 15))
    for i, ax in enumerate(axes.flatten()):
        if i < 6:
            im = ax.imshow(pre_all_coherence[i], cmap='viridis', aspect='auto', vmin=0.0, vmax=1)
            ax.set_title(f'{title_band[i]}')
            ax.set_xticks(np.arange(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation=90)
            ax.set_yticks(np.arange(len(ch_names)))
            ax.set_yticklabels(ch_names)
        else:
            im = ax.imshow(aft_all_coherence[i - 6], cmap='viridis', aspect='auto', vmin=0.0, vmax=1)
            ax.set_xticks(np.arange(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation=90)
            ax.set_yticks(np.arange(len(ch_names)))
            ax.set_yticklabels(ch_names)

    axes[0, 0].set_ylabel(f'{legend_label[0]}, 32个通道间的相干性')
    axes[1, 0].set_ylabel(f'{legend_label[1]}, 32个通道间的相干性')
    # 定义颜色条的位置和大小
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('相干性值')


def calcu_show_mean_coherence(pre_coherence, aft_coherence, legend_label, fignum):
    plt.figure(fignum, figsize=(15, 10))
    x = np.array([0, 1, 2, 3, 4, 5])
    y_pre = np.array([np.mean(pre_coherence[0]), np.mean(pre_coherence[1]), np.mean(pre_coherence[2]),
                      np.mean(pre_coherence[3]), np.mean(pre_coherence[4]), np.mean(pre_coherence[5])])
    y_aft = np.array([np.mean(aft_coherence[0]), np.mean(aft_coherence[1]), np.mean(aft_coherence[2]),
                      np.mean(aft_coherence[3]), np.mean(aft_coherence[4]), np.mean(aft_coherence[5])])

    bar_width = 0.35
    bar_offset = 0.2
    plt.bar(x - bar_offset, y_pre, bar_width, color='#00BFFF', label=legend_label[0])
    plt.bar(x + bar_offset, y_aft, bar_width, color='#FF8C00', label=legend_label[1])
    plt.grid(linestyle='--')
    plt.legend()
    plt.title('刺激前后的平均相干性比较图')
    plt.xlabel("频段")
    plt.ylabel("相干性值")
    plt.xticks(x, ['全频段', 'delta频段', 'theta频段', 'alpha频段', 'beta频段', 'gamma频段'])


def show_psd(pre_channel_psd, aft_channel_psd, freq, legend_label, fignum):
    pre_channel_psd = np.mean(pre_channel_psd, 0)
    aft_channel_psd = np.mean(aft_channel_psd, 0)

    start_range = 0
    end_range = 120

    left_yrange = -50
    right_yrange = 50

    plt.figure(fignum, figsize=(15, 10))
    plt.subplot(231)
    show_channel = 11
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("T8的功率图")

    plt.subplot(232)
    show_channel = 16
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("P8的功率图")

    plt.subplot(233)
    show_channel = 27
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("TP8的功率图")

    plt.subplot(234)
    show_channel = 9
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("T7的功率图")

    plt.subplot(235)
    show_channel = 14
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("P7的功率图")

    plt.subplot(236)
    show_channel = 26
    plt.plot(freq[start_range:end_range], 20 * np.log10(pre_channel_psd[show_channel, start_range:end_range]), 'g',
             linewidth=2, label=legend_label[0])
    plt.plot(freq[start_range:end_range], 20 * np.log10(aft_channel_psd[show_channel, start_range:end_range]), 'r',
             linewidth=2, label=legend_label[1])
    plt.xlabel("频率（hz）")
    plt.ylabel("功率（db）")
    plt.grid(linestyle='--')
    plt.ylim(left_yrange, right_yrange)
    plt.legend()
    plt.title("TP7的功率图")


# 采样率
fs = 250
legend_label = ['刺激前', '刺激后']

# 获取滤波系数
notch50_b, notch50_a, notch50_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([48, 52]), fs=fs)
notch100_b, notch100_a, notch100_zi = get_bandstop_filter_para(order=4, cutoff_freq=np.array([98, 102]), fs=fs)

pass_b, pass_a, pass_zi = get_bandpass_filter_para(order=4, cutoff_freq=np.array([0.5, 60]), fs=fs)

# 加载数据
pre_file_path = r"E:\华意的相关资料\00-医院的数据\中山七院\脑电图采集20240301\姜雷20240301刺激前_EEG.mat"
aft_file_path = r"E:\华意的相关资料\00-医院的数据\中山七院\脑电图采集20240301\姜雷20240301刺激后-2_EEG.mat"

pre_data = np.array(scio.loadmat(pre_file_path)['data'])
aft_data = np.array(scio.loadmat(aft_file_path)['data'])

# 获取通道数和时间样本点数
pre_num_channels, pre_num_samples = pre_data.shape
aft_num_channels, aft_num_samples = aft_data.shape

'''
对数据进行滤波处理
'''
pre_filt_data = filt_process(pre_data, pre_num_channels, pre_num_samples,
                             notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a)
aft_filt_data = filt_process(aft_data, aft_num_channels, aft_num_samples,
                             notch50_b, notch50_a, notch100_b, notch100_a, pass_b, pass_a)

'''
数据分段和剔除坏段
'''
# 输入的数据形状是：通道数*样本点数，输出的数据形状是段数*通道数*样本点数
pre_segments = segment_and_removal(pre_filt_data, segment_length=2, start_time=20, end_time=-20, segment_threshold=75,
                                   fs=fs)
aft_segments = segment_and_removal(aft_filt_data, segment_length=2, start_time=20, end_time=-20, segment_threshold=75,
                                   fs=fs)
'''
进行傅里叶变换，计算功率谱密度和能量值
'''
# 各频带范围
freqBand = np.array([[1, 4], [4, 8], [8, 13], [13, 30], [30, 60]])
pre_segment_power, pre_segment_psd, pre_freq = calcu_psd_power(segments=pre_segments, freqBand=freqBand, fs=fs)
aft_segment_power, aft_segment_psd, aft_freq = calcu_psd_power(segments=aft_segments, freqBand=freqBand, fs=fs)

'''
绘制二维地形图
'''
# 获取分段的平均power
pre_power = np.mean(pre_segment_power, 0)
aft_power = np.mean(aft_segment_power, 0)

# 获取全频段的power
pre_allband_power = np.sum(pre_power, 1).reshape((int(pre_power.shape[0]), 1))
aft_allband_power = np.sum(aft_power, 1).reshape((int(aft_power.shape[0]), 1))
# 将全频段和5个分段的power值拼接到一起
pre_power_data = np.hstack((pre_allband_power, pre_power))
aft_power_data = np.hstack((aft_allband_power, aft_power))
print(pre_power_data.shape)

# 构建MNE需要的数据结构变量,创建evokeds对象
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Cz', 'C3', 'T7', 'C4', 'T8', 'Pz', 'P3', 'P7', 'P4', 'P8',
            'O1', 'Oz', 'O2', 'FCz', 'CPz', 'FC3', 'FC4', 'FT7', 'CP3', 'TP7', 'TP8', 'CP4', 'FT8', 'PO3', 'PO4']
ch_types = ['eeg'] * np.size(ch_names)
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=fs)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
pre_evoked = mne.EvokedArray(pre_power_data, info)
aft_evoked = mne.EvokedArray(aft_power_data, info)
# 绘制二维头皮地形图
show_2dscalp_topography(pre_evoked, aft_evoked, legend_label, ch_names)

'''
计算相干性矩阵
'''
pre_coherence = calcu_segment_coherence(pre_segments, fs, freqBand)
aft_coherence = calcu_segment_coherence(aft_segments, fs, freqBand)

# 绘制相干性矩阵图
show_coherence(pre_coherence, aft_coherence, ch_names, legend_label)

# 计算并绘制刺激前后的平均相干性图
calcu_show_mean_coherence(pre_coherence, aft_coherence, legend_label, fignum=3)

# 绘制刺激前后的通道功率对比图
show_psd(pre_segment_psd, aft_segment_psd, pre_freq, legend_label, fignum=4)

plt.show()

