# 本脚本的主要功能是将npz格式的数据转换成edf格式的数据
import math
import pyedflib
import numpy as np
import matplotlib.pyplot as plt


# 自定义截断函数，确保浮点数转换为字符串后长度不超过8个字符
def truncate_to_8_chars(value):
    str_value = str(value)
    if len(str_value) > 8:
        if value < 0:
            rounded_value = math.floor(value)
            str_value = str(rounded_value)
            if len(str_value) > 8:
                str_value = str_value[:8]
            int_value = int(str_value)
            return int_value
        else:
            rounded_value = math.ceil(value)
            str_value = str(rounded_value)
            if len(str_value) > 8:
                str_value = str_value[:8]
            int_value = int(str_value)
            return int_value
    return value


# 读取npz文件
path = r"XCT7A20_EEG.npz"
data = np.load(path)
sfreq = 250

ch_names = data['channel_names'].tolist()
eeg_data = data['eeg_raw_data']
events_raw = data['event_marker']

# 将eeg_data的形状从段数*通道数*时间点数的三维形状，转换成通道数*时间点数的二维形状
n_channels = len(ch_names)
eeg_data = np.transpose(eeg_data, (1, 0, 2))
eeg_data = eeg_data.reshape(n_channels, -1)

# 计算EEG数据总时间长度（秒）
total_duration = eeg_data.shape[1] / sfreq

signal_headers = []
for i in range(n_channels):
    ch_dict = {
        'label': ch_names[i],
        'dimension': 'uV',
        'sample_rate': sfreq,
        'physical_min': truncate_to_8_chars(np.min(eeg_data)),
        'physical_max': truncate_to_8_chars(np.max(eeg_data)),
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }
    signal_headers.append(ch_dict)

# 将数据保存为edf格式
edf_save_name = 'XCT7A20_EEG.edf'
edf_writer = pyedflib.EdfWriter(edf_save_name, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
edf_writer.setSignalHeaders(signal_headers)
edf_writer.writeSamples(eeg_data)

# 将事件通过注释写入文件, 将events_raw的格式修改成['事件的时间戳','事件的持续时间', '事件的类型']
for event in events_raw:
    if int(event[1]) / sfreq <= total_duration:
        onset = int(event[1]) / sfreq  # 将采样点转换为秒
        description = event[0]
        edf_writer.writeAnnotation(onset, -1, description)

edf_writer.close()

print(f"\n{edf_save_name}文件已成功创建，并已添加事件标记~")



